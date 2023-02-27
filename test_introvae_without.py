import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchsummary import summary
from Model.intro_vae import Encoder, Decoder
from DataSet.DatasetintroVAE.dataset_introvae_without_config import get_trainloader
from Untils.utils import save_checkpoint, create_logger, setup_seed
from Core.function import train
from Core.config_introvae import config
import torchio as tio
from Model.intro_vae import reparameterize
import matplotlib.pyplot as plt


def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def main():
    enc = Encoder(cdim=1).to(torch.device(config.DEVICE))
    dec = Decoder(cdim=1).to(torch.device(config.DEVICE))                          ##原始的intro-vae都是 3,但是这里考虑到我们医学图像的特点这里是1

    pretrained_path = r"E:\NPCICdataset\Patient_Image\train_intro_vae_result\intro_vae_result\原始的网络架构\epoch38save_checkpoint.pth"
    weights = torch.load(pretrained_path)


    enc_weight_dict = weights['enc']
    dec_weight_dict = weights['dec']
    # enc_weight_dict = weights['enc'].state_dict()
    # dec_weight_dict = weights['dec'].state_dict()
    enc.load_state_dict(enc_weight_dict)
    dec.load_state_dict(dec_weight_dict)

    trainloader, traindataset = get_trainloader(root=config.TRAIN.ROOT, mod=config.MOD)
    rec_list = []
    real_list = []
    for idx, data in enumerate(trainloader):
        real = data['data'][tio.DATA].to(torch.device(config.DEVICE)).squeeze(4)

        real_mu, real_logvar = enc(real)
        real_code = reparameterize(real_mu, real_logvar)
        rec = dec(real_code)


        rec = rec.cpu().detach()
        real = real.cpu().detach()
        rec_list.append(rec)
        real_list.append(real)
        if idx == 5:
            break

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(real_list[0][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 2)
    plt.imshow(real_list[1][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 3)
    plt.imshow(real_list[2][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 4)
    plt.imshow(real_list[3][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 5)
    plt.imshow(real_list[4][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 6)
    plt.imshow(real_list[5][1, 0, :, :], cmap="gray")

    #plt.show()

    plt.figure(2)
    plt.subplot(2, 3, 1)
    plt.imshow(rec_list[0][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 2)
    plt.imshow(rec_list[1][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 3)
    plt.imshow(rec_list[2][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 4)
    plt.imshow(rec_list[3][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 5)
    plt.imshow(rec_list[4][1, 0, :, :], cmap="gray")

    plt.subplot(2, 3, 6)
    plt.imshow(rec_list[5][1, 0, :, :], cmap="gray")

    plt.show()


if __name__ == '__main__':
    main()
