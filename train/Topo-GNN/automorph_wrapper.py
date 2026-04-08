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

DEFAULT_AUTOMORPH_PATH = Path(__file__).parent.parent.parent.parent / "AutoMorph"


@lru_cache(1)
def automorph_prepost_processing(standard_resolution: int = 1024):
    prepost_process = basic_fundus_pre_postprocessing(
        standard_resolution=standard_resolution, rgb_to_bgr=False, pad_to_multiple=32
    )

    def preprocess(img: torch.Tensor, device: torch.device):
        x, preprocessing_info = prepost_process.preprocess(img, device=device)
        x = x[0] * 255

        # Normalization
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        for i, img in enumerate(x):
            mask = img[..., 0] > 0.0
            mean = img[mask].mean(dim=0)
            std = img[mask].std(dim=0)
            # AUTOMORPH erroneously normalizes the image by (img - mean) * std instead of (img - mean) / std...
            x[i] = (img - mean[None, None, :]) * std[None, None, :]
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        return (x,), preprocessing_info

    def postprocess(*model_outputs: torch.Tensor, preprocessing_info):
        (y_proba,) = prepost_process.postprocess(*model_outputs, preprocessing_info=preprocessing_info)
        y = y_proba.argmax(dim=-3).cpu().numpy()
        y_clean = np.zeros_like(y, dtype=np.uint8)
        y_clean[y == 3] = AVLabel.UNK
        y_clean[remove_small_objects(y == 1, 30, connectivity=5)] = AVLabel.ART
        y_clean[remove_small_objects(y == 2, 30, connectivity=5)] = AVLabel.VEI
        return (torch.from_numpy(y_clean),)

    return prepost_process.update(preprocess=preprocess, postprocess=postprocess)


@lru_cache(1)
def automorph_model(automorph_path: Path | None, device: torch.device):
    if automorph_path is None:
        automorph_path = DEFAULT_AUTOMORPH_PATH
    M2_PATH = automorph_path / "M2_Artery_vein"
    sys.path.append(str(M2_PATH.absolute()))

    from scripts.model import Generator_branch, Generator_main

    def load_checkpoint(i):
        net_G = Generator_main(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
        net_G_A = Generator_branch(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
        net_G_V = Generator_branch(input_channels=3, n_filters=32, n_classes=4, bilinear=False)
        path = M2_PATH / "ALL-AV" / f"20210724_ALL-AV_randomseed_{i}" / "Discriminator_unet"
        for name, module in {"all": net_G, "A": net_G_A, "V": net_G_V}.items():
            checkpoint_path = path / f"CP_best_F1_{name}.pth"
            module.load_state_dict(torch.load(checkpoint_path, map_location=device))
            module.eval()
            module.to(device=device)

        return net_G, net_G_A, net_G_V

    nets = [load_checkpoint(i) for i in (28, 30, 32, 34, 36, 38, 40, 42)]

    def model(x: torch.Tensor):
        with torch.no_grad():
            preds = []
            for net_G, net_G_A, net_G_V in nets:
                pred_A_part = net_G_A(x)[1].detach()
                pred_V_part = net_G_V(x)[1].detach()
                pred, _, _, _ = net_G(x, pred_A_part, pred_V_part)
                preds += [F.softmax(pred.clone().detach(), dim=1)]

            return torch.stack(preds, dim=0).mean(dim=0)

    return model


@fundus_inference("av")
def automorph_segment_av(
    fundus: torch.Tensor,
    *,
    automorph_path: Path | None = None,
    device: DeviceLikeType | Literal["auto"] = "auto",
):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepost_process = automorph_prepost_processing()
    net = automorph_model(automorph_path, device)

    x, preprocessing_info = prepost_process.preprocess(fundus, device=device)
    y = net(*x)
    (y_clean,) = prepost_process.postprocess(y, preprocessing_info=preprocessing_info)

    return y_clean
