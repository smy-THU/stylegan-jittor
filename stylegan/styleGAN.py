import jittor as jt
import numpy as np
from jittor import nn
from jittor.nn import BCEWithLogitsLoss

from .dis import Discriminator
from .gen import Generator



class StyleGAN:

    def __init__(self, resolution, num_channels, latent_size, gen_args, dis_args, learning_rate=0.01):
        self.max_depth = int(np.log2(resolution)) - 1
        self.latent_size = latent_size
        # self.device = device

        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=num_channels,
                             resolution=resolution,
                             **gen_args)
        self.dis = Discriminator(num_channels=num_channels,
                                 resolution=resolution,
                                 **dis_args)

        # define the optimizers for the discriminator and generator
        self.gen_optim = nn.Adam(self.gen.parameters(), lr=learning_rate)
        self.dis_optim = nn.Adam(self.dis.parameters(), lr=learning_rate)

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)
        loss_fuc = BCEWithLogitsLoss()
        real_loss = loss_fuc(jt.squeeze(r_preds, 1), jt.ones(real_samps.shape[0]))
        fake_loss = loss_fuc(jt.squeeze(f_preds, 1), jt.zeros(fake_samps.shape[0]))

        return (real_loss + fake_loss) / 2

    def gen_loss(self, fake_samps, height, alpha):
        preds = self.dis(fake_samps, height, alpha)
        loss_fuc = BCEWithLogitsLoss()
        return loss_fuc(jt.squeeze(preds, 1), jt.ones(fake_samps.shape[0]))

