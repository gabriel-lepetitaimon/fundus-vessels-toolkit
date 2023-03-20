import segmentation_models_pytorch as smp
from efficientnet_pytorch import EfficientNet


def setup_model(cfg, n_in, n_out, mode='segment'):
    if mode=='segment':
        return smp.Unet(encoder_name="efficientnet-b4", in_channels=n_in, classes=n_out, encoder_weights="imagenet")
    else:
        return EfficientNet.from_pretrained("efficientnet-b4", in_channels=n_in, num_classes=n_out)