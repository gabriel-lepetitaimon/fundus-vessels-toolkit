# This file is adapted from https://github.com/Eyened/rtnls_vascx_models/blob/main/vascx_models/inference.py

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import numpy.typing as npt
import torch
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from scipy.ndimage import gaussian_filter

from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.models.generic_inference import fundus_inference
from fundus_toolkits.models.pre_postprocessing import DeviceLikeType, basic_fundus_pre_postprocessing


def circular_mirror_pixel_and_mask(
    img: npt.NDArray[np.float32], radius: int, cx: Optional[int] = None, cy: Optional[int] = None
):
    shrink_ratio = 0.01
    h, w, c = img.shape
    cx = cx if cx is not None else w // 2
    cy = cy if cy is not None else h // 2
    d = int(np.round(shrink_ratio * radius))

    mirrored_image = img.copy()
    # === Image black strips ===
    img_mask = np.where(img[..., 0] > 0.1)
    min_y, max_y = np.min(img_mask[0]) + d, np.max(img_mask[0]) - d
    min_x, max_x = np.min(img_mask[1]) + d, np.max(img_mask[1]) - d

    # Flat mirror around strips
    mirrored_image[:min_y] = mirrored_image[2 * min_y - 1 : min_y - 1 : -1]
    mirrored_image[max_y:] = mirrored_image[max_y : 2 * max_y - h : -1]
    mirrored_image[:, :min_x] = mirrored_image[:, 2 * min_x - 1 : min_x - 1 : -1]
    mirrored_image[:, max_x:] = mirrored_image[:, max_x : 2 * max_x - w : -1]

    # === Circle ROI ===
    dx = np.arange(w)[None, :] - cx
    dy = np.arange(h)[:, None] - cy
    r = (1 - shrink_ratio) * w // 2
    dx_norm = dx / r
    dy_norm = dy / r
    r_squared_norm = dx_norm**2 + dy_norm**2

    mask_outside = r_squared_norm > 1
    y0, x0 = np.where(mask_outside)
    scale = 1 / r_squared_norm[mask_outside]

    x1 = np.round(cx + dx[0, x0] * scale).astype(int)
    y1 = np.round(cy + dy[y0, 0] * scale).astype(int)
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    mirrored_image[y0, x0] = mirrored_image[y1, x1]

    # === MASK ===
    mask = r_squared_norm < 1
    mask[:min_y] = False
    mask[max_y - d :] = False
    mask[:, : max_x + d] = False
    mask[:, max_x - d :] = False
    return mirrored_image, mask


@lru_cache(1)
def vascx_prepost_processing(standard_resolution: int = 1024):
    from rtnls_fundusprep.cfi_bounds import unsharp_masking

    prepost_process = basic_fundus_pre_postprocessing(
        standard_resolution=standard_resolution, rgb_to_bgr=False, pad_to_multiple=32
    )

    def preprocess(imgs: torch.Tensor, device: torch.device):
        (imgs,), preprocessing_info = prepost_process.preprocess(imgs, device=device)

        # Contrast Enhancement
        imgs_np: list[npt.NDArray[np.float32]] = list((imgs.permute(0, 2, 3, 1).numpy(force=True)).astype(np.float32))  # type: ignore
        for i, img in enumerate(imgs_np):
            assert img.shape[0] == img.shape[1], "Expected square images"
            img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            radius = 128
            img_mirrored, mask = circular_mirror_pixel_and_mask(img_256, radius=radius)
            blur_256 = gaussian_filter(img_mirrored, (0.05 * radius, 0.05 * radius, 0))
            blur = cv2.resize(blur_256, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0])) > 0.5
            ce_np = unsharp_masking(img, blur, contrast_factor=4, sharpen=False)

            preprocessing_info["ce"] = ce_np
            imgs_np[i] = np.concatenate([img, ce_np], axis=-1)  # (H, W, 6)

        x = torch.from_numpy(np.stack(imgs_np, axis=0)).permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return (x,), preprocessing_info

    return prepost_process.update(preprocess=preprocess)


@lru_cache(1)
def vascx_model(device: torch.device):
    ensemble_av = SegmentationEnsemble.from_huggingface("Eyened/vascx:artery_vein/av_july24.pt").to(device).eval()
    ensemble_av.predict_preprocessed

    def model(x: torch.Tensor):
        with torch.autocast(device_type=device.type):
            with torch.no_grad():
                proba = ensemble_av.forward(x.to(device))
                proba = torch.mean(proba, dim=1)  # average over models
                proba = torch.nn.functional.softmax(proba, dim=1)
                proba = proba[:, [1, 3, 2, 0]]
        return proba

    return model


@fundus_inference("av")
def vascx_segment_av(
    fundus: torch.Tensor,
    *,
    automorph_path: Path | None = None,
    device: DeviceLikeType | Literal["auto"] = "auto",
):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepost_process = vascx_prepost_processing()
    net = vascx_model(device)

    x, preprocessing_info = prepost_process.preprocess(fundus, device=device)
    y = net(*x)
    (y_clean,) = prepost_process.postprocess(y, preprocessing_info=preprocessing_info)

    return y_clean, preprocessing_info["ce"]
