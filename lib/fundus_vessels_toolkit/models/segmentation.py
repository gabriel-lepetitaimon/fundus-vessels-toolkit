########################################################################################################################
#   *** VESSELS SEGMENTATION ***
#   This module provides function and pretrained models for vessel segmentation on fundus images.
#
########################################################################################################################
__all__ = ["segmentation_model", "segment", "SegmentModel"]

import warnings
from enum import Enum
from typing import NamedTuple, Optional

import torch
from torch.utils import model_zoo

from steered_cnn.utils.torch import crop_pad

from .utils import ensure_superior_multiple, img_to_torch


class SegmentModel(str, Enum):
    """
    The available pretrained models for vessel segmentation.
    """

    resnet34 = "resnet34"


class ModelCache(NamedTuple):
    name: Optional[SegmentModel]
    model: Optional[torch.nn.Module]


_last_model: ModelCache = (None, None)


def segment(x, model_name: SegmentModel = SegmentModel.resnet34, roi_mask="auto", device: torch.device = "cuda"):
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
        _last_model = (model_name, model)

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
                        stacklevel=None,
                    )

                padded_shape = [ensure_superior_multiple(s, 32) for s in final_shape]
                x = crop_pad(x, padded_shape)
                y = model(x)
                y = crop_pad(y, final_shape)[0, 1] > 0.5
                y = y.cpu().numpy()

    if roi_mask == "auto":
        from ..fundus_utilities import compute_ROI_mask

        roi_mask = compute_ROI_mask(raw)
    if roi_mask is not None:
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
            import segmentation_models_pytorch as smp

            model = smp.Unet("resnet34", classes=2, activation="sigmoid")
            url = "https://drive.google.com/uc?export=download&id=1IMW3m-tig0Hd0MW46Ula_K61N8I37nHS&confirm=t&uuid=45f08800-5c34-4dd4-98f7-6a9cc338d2e6"
        case _:
            raise ValueError(f"Unknown model: {model_name}.\n" f"Available models are: resnet34")

    model_dir = torch.hub.get_dir() + "/checkpoints/fundus_vessels_toolkit/segmentation"
    model.load_state_dict(
        model_zoo.load_url(
            url, map_location="cpu", file_name=model_name + ".pth", model_dir=model_dir, check_hash=True, progress=True
        )
    )
    return model.eval()
