########################################################################################################################
#   *** VESSELS SEGMENTATION ***
#   This module provides function and pretrained models for vessel segmentation on fundus images.
#
########################################################################################################################
from __future__ import annotations

__all__ = ["segment_vessels_model", "segment_vessels", "SegmentVesselsModel"]

import typing
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import segmentation_models_pytorch as smp
import torch

from fundus_toolkits.models import download_state_dict
from fundus_toolkits.models.cache import cache_model
from fundus_toolkits.models.generic_inference import fundus_inference
from fundus_toolkits.models.pre_postprocessing import PrePostProcessing, basic_fundus_pre_postprocessing
from fundus_toolkits.utils.fundus import fundus_ROI
from fundus_toolkits.utils.image import crop_pad, read_image
from fundus_toolkits.utils.torch import DeviceLikeType
from fundus_toolkits.utils.typing import PathLike


class SegmentVesselsModel(str, Enum):
    """
    The available pretrained models for vessel segmentation.
    """

    RESNET34 = "resnet34"


SegmentVesselsModels: TypeAlias = Literal["resnet34"] | SegmentVesselsModel


@fundus_inference("vessels")
def segment_vessels(
    fundus: torch.Tensor,
    *,
    model: SegmentVesselsModels = SegmentVesselsModel.RESNET34,
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> torch.Tensor:
    """
    Segments vessels on the fundus image using the specified model.

    Parameters
    ----------
     fundus : torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike] | FundusData
        The fundus image(s) to segment. Must be one of:
        - Fundus image(s) as a tensor or array of shape (3, H, W) or (B, 3, H, W) with pixel values in [0, 1].
        - Path(s) to the fundus image file(s).
        - FundusData object(s) containing the fundus image(s).

        In the last cases the FundusData object(s) will be updated in-place with the segmentation result, and should therefore be mutable.

    fundus_mask : Literal["auto"] | npt.NDArray | torch.Tensor | PathLike | Sequence[PathLike] | None, optional
        Mask(s) used in a post-process step to remove any segmentation artefacts outside the fundus area.

        Must be one of:
        - 'auto' (default): if FundusData object(s) are provided, their fundus_mask attribute will be used if available, otherwise the mask will be inferred from the fundus image;
        - Path(s) to the fundus mask file(s);
        - A binary tensor or array of shape (H, W) or (B, H, W);
        - None: no mask will be applied.

    model : SegmentAVModel
        The model to use for segmentation.
    device : DeviceLikeType | Literal["auto"], optional
        The device to use for computation, by default "auto".

    Returns
    -------
    torch.Tensor | npt.NDArray
        The vessels map(s) as a binary tensor or array of shape (C, H, W) or (B, C, H, W).

        If the input fundus was provided as a tensor, the output will be a tensor. In all other cases, the output will be a numpy array. If FundusData object(s) were provided, they will be updated in-place with the segmentation result.
    """  # noqa: E501
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Fetch model by name ---
    net = segment_vessels_model(model).to(device=device)
    pre_post_process = segment_vessels_pre_postprocessing(model)

    # --- Apply preprocess, model and postprocess ---
    x, preprocessing_info = pre_post_process.preprocess(fundus, device=device)
    y = net(*x)
    (y_proba,) = pre_post_process.postprocess(y, preprocessing_info=preprocessing_info)

    # --- Convert probabilities to labels ---
    y_pred = y_proba.argmax(-3) if y_proba.shape[-3] > 1 else (y_proba > 0.5).squeeze(-3)

    return y_pred > 0


# @overload
# def segment_vessels(
#     fundus_image: npt.NDArray | PathLike | Sequence[PathLike],
#     model: SegmentVesselsModels = SegmentVesselsModel.RESNET34,
#     roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
#     device: DeviceLikeType | Literal["auto"] = "auto",
# ) -> npt.NDArray[np.bool_]: ...
# @overload
# def segment_vessels(
#     fundus_image: torch.Tensor,
#     model: SegmentVesselsModels = SegmentVesselsModel.RESNET34,
#     roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
#     device: DeviceLikeType | Literal["auto"] = "auto",
# ) -> torch.Tensor: ...
# def segment_vessels(
#     fundus_image: torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike],
#     model: SegmentVesselsModels = SegmentVesselsModel.RESNET34,
#     roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
#     device: DeviceLikeType | Literal["auto"] = "auto",
# ) -> torch.Tensor | npt.NDArray[np.bool_]:
#     """
#     Segments the vessels in a fundus image.

#     Parameters
#     ----------
#     fundus_image :
#         One or multiple fundus images on which to segment the vessels.

#         - Images can be provided as numpy array or torch tensor, there shape must be ``(C, H, W)`` or ``(B, C, H, W)``.
#           With: ``C`` = channels, ``H`` = height, ``W`` = width, ``B`` = batch size.
#         - They may also be provided as a path to an image file or a list of paths to image files.

#     model : SegmentModels, optional
#         The model to use for segmentation. Defaults to SegmentModel.resnet34.

#     roi_mask : numpy.ndarray | torch.Tensor | str, optional
#         The region of interest mask. If "auto", the ROI mask is computed automatically.

#     device : torch.device, optional
#         The device to use for computation. Defaults to "cuda"..

#     Returns
#     -------
#     torch.Tensor | numpy.ndarray
#         The segmentation mask as a torch tensor or numpy array, depending on the input type.

#         - If the input is a torch tensor, the output will also be a torch tensor.
#         - Otherwise, the output will be a numpy array.
#     """
#     if device == "auto":
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --- Fetch model by name ---
#     net = segment_vessels_model(model).to(device=device)
#     pre_post_process = segment_vessels_pre_postprocessing(model)

#     # --- Read images if provided as path ---
#     result_to_numpy = not isinstance(fundus_image, torch.Tensor)
#     if isinstance(fundus_image, (str, Path)):
#         fundus_image = read_image(fundus_image)
#     elif isinstance(fundus_image, Sequence):
#         if len(fundus_image) == 0:
#             raise ValueError("Empty list of images.")
#         fundus_image = np.stack([read_image(_) for _ in fundus_image])

#     with torch.inference_mode():
#         # --- Preprocess ---
#         x, preprocessing_info = pre_post_process.preprocess(fundus_image, device=device)

#         # --- Apply model ---
#         y = net(*x)
#         if not isinstance(y, tuple):
#             y = (y,)

#         # --- Postprocess ---
#         (y,) = pre_post_process.postprocess(*y, preprocessing_info=preprocessing_info)
#         vessel_mask = y.argmax(-3) if y.shape[-3] > 1 else (y > 0.5).squeeze(-3)

#         # --- Check or compute ROI mask ---
#         if isinstance(roi_mask, str) and roi_mask == "auto":
#             if isinstance(fundus_image, torch.Tensor):
#                 fundus_image = typing.cast(npt.NDArray, fundus_image.cpu().numpy())
#             if fundus_image.ndim == 4:
#                 roi_mask = np.stack([fundus_ROI(_) for _ in fundus_image])
#             else:
#                 roi_mask = fundus_ROI(fundus_image)
#         elif roi_mask is not None:
#             if isinstance(roi_mask, (str, Path)):
#                 roi_mask = read_image(roi_mask, binarize=True)
#             elif isinstance(roi_mask, Sequence):
#                 roi_mask = np.stack([read_image(_, binarize=True) for _ in roi_mask])

#             n_roi = 1 if roi_mask.ndim == 2 else roi_mask.shape[0]
#             n_fundus = 1 if fundus_image.ndim == 3 else fundus_image.shape[0]
#             if n_roi != n_fundus:
#                 raise ValueError(
#                     f"Number of fundus images ({n_fundus}) and number of ROI masks ({n_roi}) do not match.\n"
#                     f"Fundus shape: {fundus_image.shape}, ROI shape: {roi_mask.shape}"
#                 )

#         if roi_mask is not None:
#             if isinstance(vessel_mask, torch.Tensor):
#                 if not isinstance(roi_mask, torch.Tensor):
#                     roi_mask = torch.from_numpy(roi_mask)
#                 if roi_mask.device != vessel_mask.device:
#                     roi_mask = roi_mask.to(vessel_mask.device)
#             elif not isinstance(roi_mask, np.ndarray):
#                 roi_mask = roi_mask.numpy(force=True)

#             if roi_mask.ndim == 2 and vessel_mask.ndim == 3:  # type: ignore
#                 roi_mask = roi_mask[None, ...]  # type: ignore
#             elif roi_mask.ndim == 3 and vessel_mask.ndim == 2:  # type: ignore
#                 roi_mask = roi_mask[:, None, ...]  # type: ignore

#             vessel_mask *= crop_pad(roi_mask, vessel_mask.shape[-2:])  # type: ignore
#         if result_to_numpy and isinstance(vessel_mask, torch.Tensor):
#             vessel_mask = vessel_mask.numpy(force=True)
#     return vessel_mask


@cache_model()
def segment_vessels_model(model_name: SegmentVesselsModel = SegmentVesselsModel.RESNET34) -> torch.nn.Module:
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
        case SegmentVesselsModel.RESNET34:
            model = smp.Unet("resnet34", classes=2, activation="sigmoid")
            url = "https://huggingface.co/gabriel-lepetitaimon/FundusVessel/resolve/main/Segmentation/resnet34.pt?download=true"
            state_dict = download_state_dict(url, model_name, "vessels", "segmentation", map_location="cpu")
            model.load_state_dict(state_dict)
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\nAvailable models are: {', '.join(_.value for _ in SegmentVesselsModel)}."
            )

    return model.eval()


def segment_vessels_pre_postprocessing(
    model_name: SegmentVesselsModels = SegmentVesselsModel.RESNET34,
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
        case SegmentVesselsModel.RESNET34:
            return basic_fundus_pre_postprocessing(
                1200,
                rgb_to_bgr=True,
                model_name="resnet34",
                final_activation="sigmoid",
                pad_to_multiple=32,
                segmented_structure_name="vessels",
            )
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\nAvailable models are: {', '.join(_.value for _ in SegmentVesselsModel)}."
            )
