import os

os.environ['nvcc_path'] = '/home/shenmy/cuda/bin/nvcc'
from os.path import join
import jittor as jt
import numpy as np
from stylegan.styleGAN import StyleGAN
from dataset import ColorSymbol

from config import get_cfg_defaults


def train(model, dataloader, epochs, output, start_depth=0, is_resume=False, checkpoints=[]):

    model.gen.train()
    model.dis.train()

    # start_depth = model.depth - 1
    for current_depth in range(start_depth, model.max_depth):
        iter = 0

        if is_resume:
            model.gen.load(checkpoints[0])
            model.dis.load(checkpoints[1])

        for epoch in range(epochs[current_depth]):
            batch_per_epoch = len(dataloader)

            for i, batch in enumerate(dataloader):
                # calculate the alpha for fading in the layers
                alpha = iter / epochs[current_depth] * batch_per_epoch

                images = batch
                gan_input = jt.random([images.shape[0], model.latent_size], 'float32', 'normal').stop_grad()

                # train dis
                fake_samples = model.gen(gan_input, current_depth, alpha).detach()
                loss = model.dis_loss(images, fake_samples, current_depth, alpha)
                model.dis_optim.step(loss)

                # train gen
                fake_samples = model.gen(gan_input, current_depth, alpha)
                loss = model.gen_loss(fake_samples, current_depth, alpha)
                model.gen_optim.step(loss)

                iter += 1

            if epoch % 10 == 0:
                print("current depth %d, epoch %d" % current_depth, epoch)
                save_dir = join(output, 'models')
                os.makedirs(save_dir, exist_ok=True)
                gen_save_file = join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pkl")
                dis_save_file = join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pkl")

                model.gen.save(gen_save_file)
                model.dis.save(dis_save_file)


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    jt.flags.log_silent = 1
    cfg = get_cfg_defaults()
    cfg.merge_from_file('./color_symbol.yaml')
    cfg.freeze()
    train_loader = ColorSymbol(cfg.data_path, 'train').set_attrs(batch_size=64, shuffle=True, drop_last=True)
    model = StyleGAN(resolution=128, num_channels=3, latent_size=512, gen_args=cfg.gen, dis_args=cfg.dis)
    train(model, train_loader, epochs=cfg.epochs, output=cfg.output_dir)
