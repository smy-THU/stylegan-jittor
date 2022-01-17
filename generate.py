import os

os.environ['nvcc_path'] = '/home/shenmy/cuda/bin/nvcc'
import jittor as jt
import numpy as np
from stylegan.styleGAN import StyleGAN
from config import get_cfg_defaults

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./color_symbol.yaml')
    cfg.freeze()
    model = StyleGAN(resolution=128, num_channels=3, latent_size=512, gen_args=cfg.gen, dis_args=cfg.dis)
    model.gen.load('./checkpoint/generator.pkl')
    model.gen.eval()
    input = jt.random([16, model.latent_size], 'float32', 'normal').stop_grad()
    with jt.no_grad():
        imgs = model.gen(input, 5, 1).detach()
        jt.save_image(imgs, './sample.jpg', nrow=int(np.sqrt(len(imgs))), normalize=True,
                      scale_each=False, pad_value=128, padding=1)
