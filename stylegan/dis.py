import numpy as np
from jittor import nn

from .Layers import DisNeck, DisBackbone, MyConv2d


class Discriminator(nn.Module):

    def __init__(self, resolution, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, **kwargs):
        super(Discriminator, self).__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(scale=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            blocks.append(DisBackbone(nf(res - 1), nf(res - 2),
                                      gain=gain, use_wscale=use_wscale, activation_layer=act,
                                      blur_kernel=blur_filter))

            from_rgb.append(MyConv2d(num_channels, nf(res - 1), kernel_size=1,
                                     gain=gain, use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DisNeck(self.mbstd_group_size, self.mbstd_num_features,
                                   in_channels=nf(2), intermediate_channels=nf(2),
                                   gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(MyConv2d(num_channels, nf(2), kernel_size=1,
                                 gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        self.temporaryDownsampler = nn.Pool(kernel_size=2, op='mean')

    def execute(self, images_in, depth, alpha=1., labels_in=None):
        x = self.from_rgb[0](images_in)
        for i, block in enumerate(self.blocks):
            x = block(x)
        scores_out = self.final_block(x)
        return scores_out
