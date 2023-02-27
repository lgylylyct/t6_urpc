import torch
import torchio as tio
import torch.nn.functional as F

from Untils.utils import AverageMeter
from Untils.show_dataloader import show_pic
from Model.VAE import reparameterize
import matplotlib.pyplot as plt


def kl(mu, log_var):
    loss = - 0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var))
    return loss


def train(model, trainloader, optimE, optimD, logger, config, epoch):
    encoder, decoder = model                                                             ##在vae当中就没有必要进行对应的拆解开

    ls_vae = AverageMeter()

    for idx, data in enumerate(trainloader):
        real = data['data'][tio.DATA].to(torch.device(config.DEVICE)).squeeze(4)  ##这里的数据需要是一个二维的数据
        real = real.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        batch_size = real.size(0)
        optimE.zero_grad()
        optimD.zero_grad()

        real_mu, real_logvar = encoder(real)  #
        real_code = reparameterize(real_mu, real_logvar)
        rec = decoder(real_code)

        # model loss: regularization loss /reconstruction loss
        l_vae = kl(real_mu, real_logvar) + F.mse_loss(rec, real)
        l_vae.backward(retain_graph=True)
        optimE.step()
        ls_vae.update(l_vae, batch_size)

        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Enc {loss1.val:.3f} ({loss1.avg:.3f})'.format(
                epoch, idx, len(trainloader),
                loss1=ls_vae
            )
            logger.info(msg)
