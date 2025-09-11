########################################################################################################################
#   *** VESSELS SEGMENTATION ***
#   This module provides function and pretrained models for vessel segmentation on fundus images.
#
########################################################################################################################
from __future__ import annotations

__all__ = ["segmentation_model", "segment_vessels", "SegmentModel"]

import typing
import warnings
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt
import segmentation_models_pytorch as smp
import torch

from fundus_toolkits.utils.fundus import fundus_ROI
from fundus_toolkits.utils.image import PathLikeType, crop_pad, read_image
from fundus_toolkits.utils.models import ModelCache, PrePostProcessing, TensorSpec, download_state_dict

from ..utils.math import ensure_superior_multiple
from ..utils.torch import TensorArray, img_to_torch

if TYPE_CHECKING:
    from torch._prims_common import DeviceLikeType


class SegmentModel(str, Enum):
    """
    The available pretrained models for vessel segmentation.
    """

    RESNET34 = "RESNET34"


_last_model: ModelCache[SegmentModel] = ModelCache()

AUTO: Literal["auto"] = "auto"


@overload
def segment_vessels(
    fundus_image: npt.NDArray | PathLikeType | Sequence[PathLikeType],
    model_name: SegmentModel = SegmentModel.RESNET34,
    roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> npt.NDArray[np.bool_]: ...
@overload
def segment_vessels(
    fundus_image: torch.Tensor,
    model_name: SegmentModel = SegmentModel.RESNET34,
    roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> torch.Tensor: ...
def segment_vessels(
    fundus_image: torch.Tensor | npt.NDArray | PathLikeType | Sequence[PathLikeType],
    model_name: SegmentModel = SegmentModel.RESNET34,
    roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> torch.Tensor | npt.NDArray[np.bool_]:
    """
    Segments the vessels in a fundus image.

    Parameters
    ----------
    fundus_image :
        One or multiple fundus images on which to segment the vessels.

        - Images can be provided as numpy array or torch tensor, there shape must be ``(C, H, W)`` or ``(B, C, H, W)``.
          With: ``C`` = channels, ``H`` = height, ``W`` = width, ``B`` = batch size.
        - They may also be provided as a path to an image file or a list of paths to image files.

    model_name : SegmentModel, optional
        The model to use for segmentation. Defaults to SegmentModel.resnet34.

    roi_mask : numpy.ndarray | torch.Tensor | str, optional
        The region of interest mask. If "auto", the ROI mask is computed automatically.

    device : torch.device, optional
        The device to use for computation. Defaults to "cuda"..

    Returns
    -------
    torch.Tensor | numpy.ndarray
        The segmentation mask as a torch tensor or numpy array, depending on the input type.

        - If the input is a torch tensor, the output will also be a torch tensor.
        - Otherwise, the output will be a numpy array.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Fetch model by name ---
    global _last_model
    pre_post_process = basic_fundus_pre_postprocessing(model_name, None)
    if _last_model.is_cached():
        # Attempt to load the model from cache
        model = _last_model.model_to(device=device)
        if _last_model.has_pre_post_processing():
            pre_post_process = _last_model.pre_post_processing
    else:
        model = segmentation_model(model_name).to(device=device)
        pre_post_process = segmentation_pre_post_processing(model_name)
        _last_model.set_model(model_name, model, pre_post_process)

    # --- Read images if provided as path ---
    result_to_numpy = not isinstance(fundus_image, torch.Tensor)
    if isinstance(fundus_image, (str, Path)):
        fundus_image = read_image(fundus_image)
    elif isinstance(fundus_image, Sequence):
        if len(fundus_image) == 0:
            raise ValueError("Empty list of images.")
        fundus_image = np.stack([read_image(_) for _ in fundus_image])

    with torch.no_grad():
        # --- Preprocess ---
        x, preprocessing_info = pre_post_process.preprocess(fundus_image, device=device)

        # --- Apply model ---
        y = model(*x)
        if not isinstance(y, tuple):
            y = (y,)

        # --- Postprocess and apply ROI mask ---
        y = pre_post_process.postprocess(*y, preprocessing_info=preprocessing_info)
        vessel_mask = y[0]

    # --- Check or compute ROI mask ---
    if isinstance(roi_mask, str) and roi_mask == "auto":
        if isinstance(fundus_image, torch.Tensor):
            fundus_image = typing.cast(npt.NDArray, fundus_image.cpu().numpy())
        if fundus_image.ndim == 4:
            roi_mask = np.stack([fundus_ROI(_) for _ in fundus_image])
        else:
            roi_mask = fundus_ROI(fundus_image)
    elif roi_mask is not None:
        if isinstance(roi_mask, (str, Path)):
            roi_mask = read_image(roi_mask, binarize=True)
        elif isinstance(roi_mask, Sequence):
            roi_mask = np.stack([read_image(_, binarize=True) for _ in roi_mask])

        n_roi = 1 if roi_mask.ndim == 2 else roi_mask.shape[0]
        n_fundus = 1 if fundus_image.ndim == 3 else fundus_image.shape[0]
        if n_roi != n_fundus:
            raise ValueError(
                f"Number of fundus images ({n_fundus}) and number of ROI masks ({n_roi}) do not match.\n"
                f"Fundus shape: {fundus_image.shape}, ROI shape: {roi_mask.shape}"
            )

    if roi_mask is not None:
        if isinstance(vessel_mask, torch.Tensor):
            if not isinstance(roi_mask, torch.Tensor):
                roi_mask = torch.from_numpy(roi_mask)
            if roi_mask.device != vessel_mask.device:
                roi_mask = roi_mask.to(vessel_mask.device)
        elif not isinstance(roi_mask, np.ndarray):
            roi_mask = roi_mask.numpy(force=True)

        if roi_mask.ndim == 2 and vessel_mask.ndim == 3:  # type: ignore
            roi_mask = roi_mask[None, ...]  # type: ignore
        elif roi_mask.ndim == 3 and vessel_mask.ndim == 2:  # type: ignore
            roi_mask = roi_mask[:, None, ...]  # type: ignore

        vessel_mask *= crop_pad(roi_mask, vessel_mask.shape[-2:])  # type: ignore
    if result_to_numpy and isinstance(vessel_mask, torch.Tensor):
        vessel_mask = vessel_mask.numpy(force=True)
    return vessel_mask


def segmentation_model(model_name: SegmentModel = SegmentModel.RESNET34) -> torch.nn.Module:
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
        case SegmentModel.RESNET34:
            model = smp.Unet("resnet34", classes=2, activation="sigmoid")
            url = "https://huggingface.co/gabriel-lepetitaimon/FundusVessel/resolve/main/Segmentation/resnet34.pt?download=true"
            state_dict = download_state_dict(url, model_name, "vessels", "segmentation", map_location="cpu")
            model.load_state_dict(state_dict)
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\nAvailable models are: {', '.join(_.value for _ in SegmentModel)}."
            )

    return model.eval()


def segmentation_pre_post_processing(
    model_name: SegmentModel = SegmentModel.RESNET34,
) -> PrePostProcessing:
    """
    Returns the pre and post processing functions for the given model.

    Parameters
    ----------
    model_name : SegmentModel, optional
        The model to use for segmentation. Defaults to SegmentModel.resnet34.

    Returns
    -------
    PrePostProcessing
        The pre and post processing functions.
    """
    match model_name:
        case SegmentModel.RESNET34:
            return basic_fundus_pre_postprocessing("resnet34", 1200)
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\nAvailable models are: {', '.join(_.value for _ in SegmentModel)}."
            )


########################################################################################################################
#   *** BASIC PRE and POST PROCESSING ***
#   Used for models:
#   - resnet34
#
########################################################################################################################
def basic_fundus_pre_postprocessing(
    model_name: str, standard_resolution: Optional[int] = 1024, auto_resize=True
) -> PrePostProcessing:
    def preprocess(
        fundus: TensorArray, device: Optional[DeviceLikeType] = None
    ) -> Tuple[Tuple[torch.Tensor], Dict[str, Any]]:
        x = img_to_torch(fundus, device=device)
        final_shape = tuple(x.shape[-2:])
        if fundus.ndim == 4:
            final_shape = (x.shape[0],) + final_shape
        preprocessing_info: Dict[str, Any] = {"final_shape": final_shape}

        if standard_resolution is not None and not (
            standard_resolution * 0.75 < x.shape[-1] < standard_resolution * 1.4
        ):
            if auto_resize:
                f = standard_resolution / x.shape[-1]  # Assume the image is cropped
                x = torch.nn.functional.interpolate(x, scale_factor=(f,) * 2, mode="bilinear")
                preprocessing_info["scale_factor"] = f
            else:
                warnings.warn(
                    f"Image size {x.shape[-2:]} is not optimal for {model_name}.\n"
                    f"Consider resizing the image to a size close to 1024x1024.",
                    stacklevel=2,
                )

        x = torch.flip(x, [1])  # RGB to BGR
        padded_shape = [ensure_superior_multiple(s, 32) for s in x.shape]
        x = crop_pad(x, padded_shape)

        if 1.0 < x.max() <= 255:
            x = x / 255.0

        return (x,), preprocessing_info

    def postprocess(*model_outputs: torch.Tensor, preprocessing_info: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        (y,) = model_outputs
        if y.ndim == 3:
            y.unsqueeze_(1)

        # --- Rescale the output if necessary ---
        if "scale_factor" in preprocessing_info:
            f = preprocessing_info["scale_factor"]
            y = torch.nn.functional.interpolate(y, scale_factor=(1 / f,) * 2, mode="bilinear")
        y = torch.argmax(y, dim=1) if y.shape[1] > 1 else y.unsqueeze(1) > 0.5

        # --- Crop or pad the output to the final shape ---
        final_shape = preprocessing_info.get("final_shape", None)
        if final_shape is not None:
            y = crop_pad(y, final_shape[-2:])
            if len(final_shape) == 2:
                y = y[0]

        return (y,)

    input_info = (TensorSpec("fundus", ("C", "H", "W"), description="The fundus image to segment."),)
    output_info = (TensorSpec("vessels", ("H", "W"), description="The segmentation mask."),)
    return PrePostProcessing(
        preprocess=preprocess,
        postprocess=postprocess,
        input_info=input_info,
        model_input_info=input_info,
        model_output_info=output_info,
        output_info=output_info,
    )
