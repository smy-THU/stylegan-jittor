import os
os.environ['nvcc_path'] = '/home/shenmy/cuda/bin/nvcc'
import jittor as jt
import numpy as np
from stylegan.styleGAN import StyleGAN
from config import get_cfg_defaults

if __name__ == '__main__':
    os.makedirs('./style_mixing', exist_ok=True)
    jt.flags.use_cuda = 1
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./color_symbol.yaml')
    cfg.freeze()
    model = StyleGAN(resolution=128, num_channels=3, latent_size=512, gen_args=cfg.gen, dis_args=cfg.dis)
    model.gen.load('./checkpoint/generator.pkl')
    model.gen.eval()
    sourceA_inputs = jt.random([25, model.latent_size], 'float32', 'normal').stop_grad()
    sourceB_inputs = jt.random([25, model.latent_size], 'float32', 'normal').stop_grad()
    with jt.no_grad():

        A_middle = model.gen.g_mapping(sourceA_inputs)
        B_middle = model.gen.g_mapping(sourceB_inputs)
        # for i in range(12):
        #     mixing = B_middle
        #     mixing[:,0:i] = A_middle[:,0:i]
        #     for j in range(50):
        #         mixing[:,i] = (B_middle[:,i]*j+A_middle[:,i]*(50-j))/50
        #         imgs = model.gen.g_synthesis(mixing, 5, 1)
        #         jt.save_image(imgs, './style_mixing/%03d.jpg' % (i*50+j), nrow=5, normalize=True, scale_each=False, pad_value=128, padding=1)

        for i in range(500):
            mixing = (B_middle * i + A_middle * (500 - i)) / 500
            imgs = model.gen.g_synthesis(mixing, 5, 1)
            jt.save_image(imgs, './interp/%03d.jpg' % (i), nrow=5, normalize=True, scale_each=False, pad_value=128, padding=1)
