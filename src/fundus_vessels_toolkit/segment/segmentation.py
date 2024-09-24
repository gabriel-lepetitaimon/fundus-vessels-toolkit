########################################################################################################################
#   *** VESSELS SEGMENTATION ***
#   This module provides function and pretrained models for vessel segmentation on fundus images.
#
########################################################################################################################
__all__ = ["segmentation_model", "segment", "SegmentModel"]

import os
import warnings
from enum import Enum
from pickle import UnpicklingError
from typing import NamedTuple, Optional
from urllib.error import HTTPError

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils import model_zoo

from steered_cnn.utils.torch import crop_pad

from ..utils.math import ensure_superior_multiple
from ..utils.torch import img_to_torch


class SegmentModel(str, Enum):
    """
    The available pretrained models for vessel segmentation.
    """

    resnet34 = "resnet34"


class ModelCache(NamedTuple):
    name: Optional[SegmentModel]
    model: Optional[torch.nn.Module]


_last_model: ModelCache = ModelCache(None, None)


def clear_gpu_cache():
    """
    Clears the GPU cache.
    """
    _last_model = ModelCache(None, None)
    torch.cuda.empty_cache()


def segment_vessels(
    x, model_name: SegmentModel = SegmentModel.resnet34, roi_mask="auto", device: torch.device = "cuda"
):
    """
    Segments the vessels in a fundus image.

    Parameters
    ----------
    x : numpy.ndarray
        The fundus image to segment.

    model_name : SegmentModel, optional
        The model to use for segmentation. Defaults to SegmentModel.resnet34.

    roi_mask : numpy.ndarray | torch.Tensor | str, optional
        The region of interest mask. If "auto", the ROI mask is computed automatically.

    device : torch.device, optional
        The device to use for computation. Defaults to "cuda"..

    Returns
    -------
    numpy.ndarray
        The segmentation mask.
    """
    global _last_model
    if _last_model.name == model_name:
        model = _last_model.model
    else:
        model = segmentation_model(model_name).to(device=device)
        _last_model = ModelCache(model_name, model)

    raw = x

    match model_name:
        case SegmentModel.resnet34:
            with torch.no_grad():
                x = img_to_torch(x, device)

                final_shape = x.shape
                if not (1000 < final_shape[3] < 1500):
                    warnings.warn(
                        f"Image size {x.shape[-2:]} is not optimal for {model_name}.\n"
                        f"Consider resizing the image to a size close to 1024x1024.",
                        stacklevel=1,
                    )

                x = torch.flip(x, [1])  # RGB to BGR
                padded_shape = [ensure_superior_multiple(s, 32) for s in final_shape]
                x = crop_pad(x, padded_shape)
                y = model(x)
                y = crop_pad(y, final_shape)[:, 1] > 0.5
                y = y.cpu().numpy()

    if raw.ndim == 3:
        y = y[0]

    if isinstance(roi_mask, str) and roi_mask == "auto":
        from ..utils.fundus import fundus_ROI

        if raw.ndim == 4:
            roi_mask = np.stack([fundus_ROI(_) for _ in raw])
        else:
            roi_mask = fundus_ROI(raw)
    if roi_mask is not None:
        if roi_mask.ndim == 2 and y.ndim == 3:
            roi_mask = roi_mask[None]
        y *= roi_mask

    return y


def segmentation_model(model_name: SegmentModel = SegmentModel.resnet34):
    """
    Returns a pretrained model for vessel segmentation on fundus images.

    Parameters
    ----------
    model_name : SegmentModel, optional
        The model to use for segmentation. Defaults to SegmentModel.resnet34.

    Returns
    -------
    torch.nn.Module
        The pretrained model.
    """

    match model_name:
        case SegmentModel.resnet34:
            model = smp.Unet("resnet34", classes=2, activation="sigmoid")
            url = "https://huggingface.co/gabriel-lepetitaimon/FundusVessel/resolve/main/Segmentation/resnet34.pt?download=true"
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\n" f"Available models are: {', '.join(_.value for _ in SegmentModel)}."
            )

    model_dir = torch.hub.get_dir() + "/checkpoints/fundus_vessels_toolkit/segmentation"
    try:
        state_dict = model_zoo.load_url(
            url, map_location="cpu", file_name=model_name + ".pth", model_dir=model_dir, check_hash=True, progress=True
        )
    except UnpicklingError:
        os.remove(model_dir + "/" + model_name + ".pth")
        raise RuntimeError(
            "The requested model file is corrupted, it has been removed. Please retry.\n"
            "If the problem persists contact the author: the url might be invalid."
        ) from None
    except HTTPError as e:
        raise RuntimeError(
            f"An error occurred while downloading the model: {e}\n"
            "Please retry. If the problem persists contact the author."
        ) from None

    model.load_state_dict(state_dict)
    return model.eval()
