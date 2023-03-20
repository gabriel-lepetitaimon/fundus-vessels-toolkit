from .backbones import *
from .hemeling import HemelingNet
from .steered import SteeredHemelingNet, SteeredUNet
from .old_hemeling import OldHemelingNet


def setup_model(cfg):
    if cfg.check('backbone', 'unet'):
        args = cfg.subset('n_in,n_out,nfeatures,'
                          'nscale,depth,kernel,padding,'
                          'batchnorm,downsampling,upsampling')
        if cfg.get('steered', default=False):
            if cfg.get('steering', 'attention') == 'attention' and not cfg.get('attention_base', False):
                cfg['attention_base'] = True
            net = SteeredUNet(normalize_steer=cfg.get('normalized', True),
                              base=cfg.get('base', None),
                              attention_mode=cfg.get('normalized', 'shared'),
                              attention_base=cfg.get('attention_base', False), **args)
        else:
            net = UNet(**args)
    return net

