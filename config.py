from yacs.config import CfgNode as CN

_C = CN()

_C.output_dir = ''
_C.data_path = ''
_C.epochs = []

_C.gen = CN()
_C.gen.latent_size = 512
# 8 in original paper
_C.gen.mapping_layers = 4
_C.gen.blur_filter = [1, 2, 1]
_C.gen.truncation_psi = 0.7
_C.gen.truncation_cutoff = 8

_C.dis = CN()
_C.dis.use_wscale = True
_C.dis.blur_filter = [1, 2, 1]


def get_cfg_defaults():
    return _C.clone()
