from .steered import SteeredUNet, SteeredHemelingNet
from .backbones import UNet, HemelingNet


def setup_model(cfg, n_in, n_out):
    if cfg.check('backbone', ('unet', 'hemeling')):
        args = cfg.subset('nfeatures,'
                          'nscale,depth,padding,'
                          'batchnorm,downsampling,upsampling')
        if cfg.get('steered', default=False):
            NET = SteeredUNet if cfg.get('backbone') == 'unet' else SteeredHemelingNet
            steered = cfg.get('steered')
            if isinstance(steered, str):
                cfg['steered'] = {'steering': steered}
                steered = {'steering': steered}
            if steered.get('steering', 'attention') == 'attention':
                steered['steering'] = 'attention'
                if steered.get('attention_mode', False):
                    steered['attention_mode'] = 'shared'
            else:
                steered['attention_mode'] = False
            net = NET(n_in, n_out,
                      rho_nonlinearity=steered.get('rho_nonlinearity', None),
                      base=steered.get('base', None),
                      attention_mode=steered.get('attention_mode'),
                      attention_base=steered.get('attention_base', False), **args)
        else:
            NET = UNet if cfg.get('backbone') == 'unet' else HemelingNet
            net = NET(n_in, n_out, kernel=cfg.get('kernel',3), **args)
    return net
