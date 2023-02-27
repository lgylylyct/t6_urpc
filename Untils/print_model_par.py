from torchsummary import summary
import torch
from Model.intro_vae import Encoder, Decoder

test_if = False

if test_if:

    enc = Encoder(cdim=1).to(torch.device(config.DEVICE))
    dec = Decoder(cdim=1).to(torch.device(config.DEVICE))
    print(summary(enc, (1, 64, 64)))
    print(summary(dec, (512, 64, 64)))
# input()