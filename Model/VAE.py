import torch
import torch.nn as nn


class _Residual_Block(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(_Residual_Block, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))

        return output


class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512), image_size=256):
        super(Encoder, self).__init__()

        self.zdim = zdim
        cc = channels[0]                                      #  64 一开始是提升维度到了64
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),         ## 1 64 提升维度  5*5的卷积核
            nn.InstanceNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.fc = nn.Conv2d(cc, 2 * zdim, 3, 1, 1)

    def forward(self, x):
        y = self.main(x)
        y = self.fc(y)                    ##再次升通道的数目 变成原来的两倍，但是对应的图像的大小不做改变
        mu, logvar = y.chunk(2, dim=1)   ##在第一个维度上分成了两半，就是把通道数砍成了两半 最后的通道数都是512 512  尺寸是原始的尺寸除以16 就是传统意义上encoder的效果
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512), image_size=256):
        super(Decoder, self).__init__()

        assert (2 ** len(channels)) * 16 == image_size   ##这里声明一下必须是通道数目的长度和图像原始大小之间的关系
        cc = channels[-1]
        self.fc = nn.Sequential(
            nn.Conv2d(zdim, cc, 3, 1, 1),
            nn.ReLU(True),
        )

        sz = 16

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), _Residual_Block(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        y = self.fc(z)
        y1 = self.main[1](self.main[0](y))
        y2 = self.main[3](self.main[2](y1))
        y3 = self.main[5](self.main[4](y2))
        y4 = self.main[7](self.main[6](y3))
        y5 = self.main[8](y4)
        out = self.main[9](y5)

        return out


def reparameterize(mu, logvar):
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


if __name__ == '__main__':
    from torchsummary import summary

    enc = Encoder(cdim=1).to(torch.device('cuda'))
    dec = Decoder(cdim=1).to(torch.device('cuda'))

    print(summary(enc, (1, 192, 192)))
    print(summary(dec, (512, 32, 32)))
    #
    # from torchinfo import summary
    #
    # model = Encoder(cdim=1)
    # device = torch.device('cuda')
    # model = model.to(device)
    #
    # enc = Encoder(cdim=1).to(torch.device('cuda'))
    # dec = Decoder(cdim=1).to(torch.device('cuda'))
    # summary(enc,imput_size=(1, 64, 64))
    # summary(dec,imput_size=(512, 64, 64))
    # import timm
    # from torchinfo import summary
    #
    # net = timm.create_model('resnet18', pretrained=False, num_classes=120)
    # print(summary(net, input_size=(2, 3, 64, 64)))