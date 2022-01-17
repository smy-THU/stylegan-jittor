import random

import jittor as jt
import numpy as np
from jittor import nn

from .Layers import Truncation, GMapping, GSynthesis


class Generator(nn.Module):

    def __init__(self, resolution, latent_size=512, dlatent_size=512,
                 truncation_psi=0.7, truncation_cutoff=8, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **kwargs):

        super(Generator, self).__init__()

        self.style_mixing_prob = style_mixing_prob

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        if truncation_psi > 0:
            self.truncation = Truncation(avg_latent=jt.zeros(dlatent_size),
                                         max_layer=truncation_cutoff,
                                         threshold=truncation_psi,
                                         beta=dlatent_avg_beta)
        else:
            self.truncation = None

    def execute(self, latents_in, depth, alpha, labels_in=None):

        dlatents_in = self.g_mapping(latents_in)

        # if self.training:
        if self.is_training:

            if self.truncation is not None:
                self.truncation.update(dlatents_in[0, 0].detach())

            # Perform style mixing regularization.
            if self.style_mixing_prob is not None and self.style_mixing_prob > 0:
                latents2 = jt.random(latents_in.shape, 'float32', 'normal').stop_grad()
                dlatents2 = self.g_mapping(latents2)

                layer_idx = jt.array(np.arange(self.num_layers)[np.newaxis, :, np.newaxis])
                cur_layers = 2 * (depth + 1)
                mixing_cutoff = random.randint(1,
                                               cur_layers) if random.random() < self.style_mixing_prob else cur_layers

                mask_dlatents = layer_idx < mixing_cutoff
                dlatents_in = mask_dlatents * dlatents_in + (1 - mask_dlatents) * dlatents2

            if self.truncation is not None:
                dlatents_in = self.truncation(dlatents_in)

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images
