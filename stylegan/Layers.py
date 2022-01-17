from collections import OrderedDict

import jittor as jt
import jittor.nn as nn
import numpy as np


class ImgNormalizer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def execute(self, x):
        return x / jt.sqrt(jt.mean(x.sqr(), dim=1, keepdims=True) + self.epsilon)


class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.ndim == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(
                [shape[0], shape[1], shape[2], factor, shape[3], factor])
            x = x.view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def execute(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):

    def __init__(self, factor=2, gain=1):

        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = Blur(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def execute(self, x):
        assert x.ndim == 4
        if self.blur is not None and x.dtype == jt.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        return nn.pool(x, kernel_size=self.factor, op='mean', stride=self.factor)


class Blur(nn.Module):

    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):

        super().__init__()
        # breakpoint()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = jt.float32(kernel)
        kernel = kernel.unsqueeze(dim=1) * kernel.unsqueeze(dim=0)
        kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.kernel = kernel

        self.stride = stride

    def execute(self, x):

        size = self.kernel.shape
        kernel = self.kernel.expand([x.size(1), size[1], size[2], size[3]])
        x = nn.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((size[2] - 1) / 2),
            groups=x.size(1)
        )
        return x


class MyLinear(nn.Module):
    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()

        self.use_bias_ = bias
        he_std = gain * input_size ** (-0.5)  # He init

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.init.gauss([output_size, input_size], 'float32', std=init_std)
        if self.use_bias_:
            self.bias = nn.init.constant([output_size], 'float32', 0.0)
            self.b_mul = lrmul
        else:
            self.bias = None

    def execute(self, x):

        y = nn.matmul_transpose(x, self.weight * self.w_mul)
        if self.use_bias_:
            y += self.bias * self.b_mul
        return y


class MyConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5, use_wscale=False, lrmul=1,
                 bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()
        self.use_bias_ = bias
        self.upscale_ = upscale
        self.downscale_ = downscale

        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None

        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None

        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)
        self.kernel_size = kernel_size

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.init.gauss([output_channels, input_channels, kernel_size, kernel_size], 'float32',
                                    std=init_std)
        if self.use_bias_:
            self.bias = nn.init.constant([output_channels], 'float32', 0.0)
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def execute(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            # w = F.pad(w, [1, 1, 1, 1])
            w = nn.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = nn.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            # w = F.pad(w, [1, 1, 1, 1])
            w = nn.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            # x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            x = nn.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            # return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
            return nn.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            # x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)
            x = nn.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class Noise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.init.constant([channels], 'float32', 0.0)
        self.noise = None

    def execute(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = jt.random([x.size(0), 1, x.size(2), x.size(3)], x.dtype, 'normal')
        elif noise is None:
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)

    def execute(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.ndim - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class ChooseLayers(nn.Module):
    def __init__(self, channels, dlatent_size, use_wscale,
                 use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()

        layers = []
        if use_noise:
            layers.append(('noise', Noise(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', ImgNormalizer()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels, affine=False)))

        self.top_epi = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def execute(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def execute(self, x):
        return x.view(x.size(0), *self.shape)


class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def execute(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features, c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdims=True)
        y = (y.sqr()).mean(0, keepdims=True)
        y = (y + 1e-8).pow(0.5)
        y = y.mean([3, 4, 5], keepdims=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand([group_size, y.size(1), y.size(2), h, w]).clone().reshape(b, self.num_new_features, h, w)
        z = jt.concat([x, y], dim=1)
        return z


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta

        self.avg_latent = avg_latent
        self.avg_latent.stop_grad()

    def update(self, last_avg):
        self.avg_latent.update(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def execute(self, x):
        assert x.ndim == 3
        interp = self.avg_latent + self.threshold * (x - self.avg_latent)
        do_trunc = (jt.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return do_trunc * interp + (1 - do_trunc) * x


class BackboneBlock(nn.Module):

    def __init__(self, nf, dlatent_size, const_input_layer, gain,
                 use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf

        if self.const_input_layer:
            self.const = jt.ones([1, nf, 4, 4])
            self.bias = jt.ones(nf)
        else:
            self.dense = MyLinear(dlatent_size, nf * 16, gain=gain / 4, use_wscale=use_wscale)

        self.epi1 = ChooseLayers(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                 use_styles, activation_layer)
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = ChooseLayers(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                 use_styles, activation_layer)

    def execute(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        if self.const_input_layer:
            x = self.const.expand([batch_size, self.const.size(1), self.const.size(2), self.const.size(3)])
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)

        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])

        return x


class GSynBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain,
                 use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):

        super().__init__()

        if blur_filter:
            blur = Blur(blur_filter)
        else:
            blur = None

        self.conv0_up = MyConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                 intermediate=blur, upscale=True)
        self.epi1 = ChooseLayers(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                 use_styles, activation_layer)
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = ChooseLayers(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                 use_styles, activation_layer)

    def execute(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class DisNeck(nn.Sequential):
    def __init__(self,
                 mbstd_group_size,
                 mbstd_num_features,
                 in_channels,
                 intermediate_channels,
                 gain, use_wscale,
                 activation_layer,
                 resolution=4,
                 in_channels2=None,
                 output_features=1,
                 last_gain=1):

        layers = []
        if mbstd_group_size > 1:
            layers.append(('stddev_layer', StddevLayer(mbstd_group_size, mbstd_num_features)))

        if in_channels2 is None:
            in_channels2 = in_channels

        layers.append(('conv', MyConv2d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
                                        gain=gain, use_wscale=use_wscale)))
        layers.append(('act0', activation_layer))
        layers.append(('view', View(-1)))
        layers.append(('dense0', MyLinear(in_channels2 * resolution * resolution, intermediate_channels,
                                          gain=gain, use_wscale=use_wscale)))
        layers.append(('act1', activation_layer))
        layers.append(('dense1', MyLinear(intermediate_channels, output_features,
                                          gain=last_gain, use_wscale=use_wscale)))

        super().__init__(OrderedDict(layers))


class DisBackbone(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel):
        super().__init__(OrderedDict([
            ('conv0', MyConv2d(in_channels, in_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)),
            # out channels nf(res-1)
            ('act0', activation_layer),
            ('blur', Blur(kernel=blur_kernel)),
            ('conv1_down', MyConv2d(in_channels, out_channels, kernel_size=3,
                                    gain=gain, use_wscale=use_wscale, downscale=True)),
            ('act1', activation_layer)]))


class GMapping(nn.Module):

    def __init__(self, latent_size=512, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(scale=0.2), np.sqrt(2))}[mapping_nonlinearity]

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', ImgNormalizer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', MyLinear(self.latent_size, self.mapping_fmaps,
                                          gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 MyLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def execute(self, x):

        x = self.map(x)
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand([x.size(0), self.dlatent_broadcast, x.size(1)])
        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 use_styles=True, const_input_layer=True, use_noise=True, nonlinearity='lrelu',
                 use_wscale=True, use_pixel_norm=False, use_instance_norm=True, blur_filter=None,
                 **kwargs):

        super().__init__()

        if blur_filter is None:
            blur_filter = [1, 2, 1]

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(scale=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = BackboneBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                        use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # create the ToRGB layers for various outputs
        rgb_converters = [MyConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            blocks.append(GSynBackbone(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                       use_noise, use_pixel_norm, use_instance_norm, use_styles, act))
            rgb_converters.append(MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)
        self.temporaryUpsampler = lambda x: nn.interpolate(x, scale_factor=2, mode='nearest')

    def execute(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        x = self.init_block(dlatents_in[:, 0:2])
        for i, block in enumerate(self.blocks):
            x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
        images_out = self.to_rgb[-1](x)

        return images_out
