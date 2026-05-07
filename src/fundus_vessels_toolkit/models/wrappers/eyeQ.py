import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import remove_small_objects

from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.models.generic_inference import fundus_inference
from fundus_toolkits.models.pre_postprocessing import DeviceLikeType, basic_fundus_pre_postprocessing

DEFAULT_AUTOMORPH_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "AutoMorph"


@lru_cache(1)
def eyeQ_preprocessing(standard_resolution: int = 512):
    prepost_process = basic_fundus_pre_postprocessing(
        standard_resolution=standard_resolution, rgb_to_bgr=False, pad_to_multiple=32
    )

    def preprocess(img: torch.Tensor, device: torch.device):
        (x,), preprocessing_info = prepost_process.preprocess(img, device=device)

        # Normalization
        for i, img in enumerate(x):
            not_null_img = img[img > 0.0]
            mean = not_null_img.mean()
            std = not_null_img.std()
            x[i] = (img - mean) / std

        return (x,), preprocessing_info

    def postprocess(*model_outputs: torch.Tensor, preprocessing_info):
        (y_proba,) = model_outputs
        return (y_proba.argmax(dim=-1),)

    return prepost_process.update(preprocess=preprocess, postprocess=postprocess)


@lru_cache(1)
def eyeQ_model(automorph_path: Path | None, device: torch.device):
    if automorph_path is None:
        automorph_path = DEFAULT_AUTOMORPH_PATH
    M1_PATH = automorph_path / "M1_Retinal_Image_quality_EyePACS"
    sys.path.append(str(M1_PATH.absolute()))

    from model import Efficientnet_fl

    def load_checkpoint(i):
        model_fl = Efficientnet_fl(pretrained=True).to(device=device)
        model_name = [
            "0_seed_42",
            "1_seed_40",
            "2_seed_38",
            "3_seed_36",
            "4_seed_34",
            "5_seed_32",
            "6_seed_30",
            "7_seed_28",
        ][i]
        checkpoint_path = (
            M1_PATH / "Retinal_quality" / "EyePACS_quality" / "efficientnet" / model_name / "best_loss_checkpoint.pth"
        )
        model_fl.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_fl.eval()
        return model_fl

    nets = [load_checkpoint(i) for i in range(8)]

    def model(x: torch.Tensor):
        with torch.no_grad():
            preds = [net(x).softmax(dim=1) for net in nets]
            return torch.stack(preds, dim=0).mean(dim=0)

    return model


def eye_Q(
    fundus: torch.Tensor,
    *,
    automorph_path: Path | None = None,
    device: DeviceLikeType | Literal["auto"] = "auto",
):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepost_process = eyeQ_preprocessing()
    net = eyeQ_model(automorph_path, device)

    x, preprocessing_info = prepost_process.preprocess(fundus, device=device)
    y = net(*x)
    (y_clean,) = prepost_process.postprocess(y, preprocessing_info=preprocessing_info)

    return y_clean
