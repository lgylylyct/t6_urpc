import torch
import os
from torchvision import models
from torchsummary import summary
import monai
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mod = 3

print(torch.cuda.current_device())  #这句必须加
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if mod == 1:
    model = monai.networks.nets.DenseNet264(init_features=64, growth_rate=32, block_config=(6, 12, 64, 48),
                                            pretrained=False,
                                            progress=True, spatial_dims=3, in_channels=1, out_channels=2)

if mod == 2:
    model = monai.networks.nets.ResNet([50], [3, 4, 6, 3], spatial_dims=3, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, num_classes=400, feed_forward=True)
    #model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

if mod == 3 :
    model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)

model.to(device)
#print(model.__dict__)
#print(model)
#print(summary(model, (1, 64, 64, 64)))
model_stat = model.state_dict()
model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2,pretrained = True)
print(1)