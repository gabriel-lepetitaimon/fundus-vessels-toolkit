########################################################################################################################
#   *** VESSELS CLASSIFICATION ***
#   This module provides function and pretrained models for vessel classification on fundus images.
#
########################################################################################################################
from __future__ import annotations

__all__ = ["classify_av_model", "classify_av", "ClassifyAVModel"]

import typing
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt
import torch

from steered_cnn.models.steered import SteeredHemelingNet

from fundus_toolkits.models import basic_fundus_pre_postprocessing, cache_model, download_state_dict
from fundus_toolkits.models.generic_inference import fundus_inference
from fundus_toolkits.models.pre_postprocessing import PrePostProcessing, TensorSpec
from fundus_toolkits.utils.fundus import fundus_ROI, gaussian_preprocess_torch
from fundus_toolkits.utils.image import crop_pad, read_image
from fundus_toolkits.utils.torch import DeviceLikeType, TensorArray, img_to_torch
from fundus_toolkits.utils.typing import PathLike


class ClassifyAVModel(str, Enum):
    """
    The available pretrained models for vessel segmentation.
    """

    HEMELING = "hemeling"
    HEMELING_STEERED = "hemeling_steered"


ClassifyAVModels: typing.TypeAlias = Literal["hemeling", "hemeling_steered"] | ClassifyAVModel


@overload
def classify_av(
    fundus_image: npt.NDArray | PathLike | Sequence[PathLike],
    vessels_mask: Optional[torch.Tensor | npt.NDArray[np.bool_]] = None,
    *,
    roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
    model_name: ClassifyAVModel = ClassifyAVModel.HEMELING_STEERED,
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> npt.NDArray[np.uint8]: ...
@overload
def classify_av(
    fundus_image: torch.Tensor,
    vessels_mask: Optional[torch.Tensor | npt.NDArray[np.bool_]] = None,
    *,
    roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
    model_name: ClassifyAVModel = ClassifyAVModel.HEMELING_STEERED,
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> torch.Tensor: ...
def classify_av(
    fundus_image: torch.Tensor | npt.NDArray | PathLike | Sequence[PathLike],
    vessels_mask: Optional[torch.Tensor | npt.NDArray[np.bool_]] = None,
    *,
    roi_mask: Literal["auto"] | npt.NDArray | torch.Tensor | str | Sequence[str] = "auto",
    model_name: ClassifyAVModel = ClassifyAVModel.HEMELING_STEERED,
    device: DeviceLikeType | Literal["auto"] = "auto",
) -> torch.Tensor | npt.NDArray[np.uint8]:
    """
    Classifies the fundus images using a pretrained model.

    Parameters
    ----------
    fundus_image : torch.Tensor
        The fundus images to classify.

    model_name : ClassifyModel, optional
        The model to use for classification. Defaults to ClassifyModel.hemeling_steered.

    vessels_mask : torch.Tensor | npt.NDArray[np.bool_], optional
        The vessels segmentation mask to use for classification. If not provided, the model will compute it from the fundus image.

    device : torch.device, optional
        The device to use for computation. Defaults to "cuda".

    Returns
    -------
    torch.Tensor
        The classification results.
    """  # noqa: E501

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Fetch model by name ---
    model = classify_av_model(model_name).to(device=device)
    pre_post_process = classification_pre_post_processing(model_name)

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
        x, preprocessing_info = pre_post_process.bind_preprocess(
            fundus_image=fundus_image, vessels_mask=vessels_mask, device=device
        )

        # --- Apply model ---
        y = model(*x)
        if not isinstance(y, tuple):
            y = (y,)

        # --- Postprocess and apply ROI mask ---
        y = pre_post_process.postprocess(*y, preprocessing_info=preprocessing_info)
        av_pred = y[0]

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
        if isinstance(av_pred, torch.Tensor):
            if not isinstance(roi_mask, torch.Tensor):
                roi_mask = torch.from_numpy(roi_mask)
            if roi_mask.device != av_pred.device:
                roi_mask = roi_mask.to(av_pred.device)
        elif not isinstance(roi_mask, np.ndarray):
            roi_mask = roi_mask.numpy(force=True)

        if roi_mask.ndim == 2 and av_pred.ndim == 3:  # type: ignore
            roi_mask = roi_mask[None, ...]  # type: ignore
        elif roi_mask.ndim == 3 and av_pred.ndim == 2:  # type: ignore
            roi_mask = roi_mask[:, None, ...]  # type: ignore

        av_pred *= crop_pad(roi_mask, av_pred.shape[-2:])  # type: ignore
    if result_to_numpy and isinstance(av_pred, torch.Tensor):
        av_pred = av_pred.numpy(force=True)
    return av_pred


@cache_model()
def classify_av_model(model_name: ClassifyAVModels = ClassifyAVModel.HEMELING_STEERED) -> torch.nn.Module:
    """
    Loads a pretrained model for vessel segmentation.

    Parameters
    ----------
    model_name : ClassifyModel, optional
        The model to use for segmentation. Defaults to ClassifyModel.resnet34.

    device : torch.device, optional
        The device to use for computation. Defaults to "cuda".

    Returns
    -------
    torch.nn.Module
        The segmentation model.
    """
    match ClassifyAVModel(model_name):
        case ClassifyAVModel.HEMELING_STEERED:
            opts = dict(n_in=6, n_out=1, nfeatures=11, nscale=5, depth=2)
            opts |= dict(batchnorm=True, padding="auto", upsampling="bilinear", downsampling="conv")
            opts |= dict(rho_nonlinearity="normalize", attention_mode=False, attention_base=False)
            opts["base"] = dict(kr=5, max_k=2, cap_k=True, std=0.5, oversample=16, phase=None, size=None)  # type: ignore
            model = SteeredHemelingNet(**opts)  # type: ignore
            url = "https://huggingface.co/gabriel-lepetitaimon/FundusVessel/resolve/main/Classification/steered_hemeling.ckpt?download=true"
            state_dict = download_state_dict(url, model_name, "vessels", "classification", map_location="cpu")
            state_dict = {k[6:]: v for k, v in state_dict["state_dict"].items() if k.startswith("model.")}
            model.load_state_dict(state_dict)
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\nAvailable models are: {', '.join(_.value for _ in ClassifyAVModel)}."
            )

    return model.eval()


def classification_pre_post_processing(
    model_name: ClassifyAVModels = ClassifyAVModel.HEMELING_STEERED,
) -> PrePostProcessing:
    """
    Preprocesses the fundus images for classification.

    Parameters
    ----------
    fundus_images : torch.Tensor
        The fundus images to preprocess.

    model_name : ClassifyModel, optional
        The model to use for preprocessing. Defaults to ClassifyModel.hemeling_steered.

    Returns
    -------
    torch.Tensor
        The preprocessed fundus images.
    """
    match ClassifyAVModel(model_name):
        case ClassifyAVModel.HEMELING_STEERED:
            return steering_pre_post_processing(565, model_name="hemeling_steered", auto_resize=True)
        case ClassifyAVModel.HEMELING:
            return basic_fundus_pre_postprocessing(565, model_name="hemeling", auto_resize=True)
        case _:
            raise ValueError(
                f"Unknown model: {model_name}.\nAvailable models are: {', '.join(_.value for _ in ClassifyAVModel)}."
            )


########################################################################################################################
#   *** PRE and POST PROCESSING FOR STEERABLE MODELS ***
#   Used for models:
#       - SteeredHemelingNet
#
########################################################################################################################
def steering_pre_post_processing(
    standard_resolution: Optional[int] = 512, model_name: str = "this steered model", auto_resize=True
) -> PrePostProcessing:
    """
    Returns the pre and post processing for the classification model.

    Returns
    -------
    PrePostProcessing
        The pre and post processing for the classification model.
    """

    def preprocess(
        fundus: TensorArray,
        vessels: torch.Tensor | npt.NDArray[np.bool_],
        alpha: Optional[torch.Tensor | npt.NDArray[np.float32]] = None,
        device: DeviceLikeType | Literal["auto"] = "auto",
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Preprocess the fundus image ---
        x = img_to_torch(fundus, device=device)

        final_shape = tuple(x.shape[-2:])
        if fundus.ndim == 4:
            final_shape = (x.shape[0],) + final_shape
        preprocessing_info: Dict[str, Any] = {"final_shape": final_shape}

        # --- Check vessel map ---
        assert vessels.ndim == fundus.ndim - 1, (
            "Vessel mask must be a 2D array for single fundus image."
            if fundus.ndim == 4
            else "Vessel mask must be a 3D array for batch of fundus images."
        )
        assert vessels.shape == x.shape[-2:], (
            f"Vessel mask shape {vessels.shape} does not match fundus image shape {x.shape[-2:]}."
        )
        vessels_torch = torch.tensor(vessels, device=device)
        if vessels_torch.dtype != torch.bool:
            vessels_torch = vessels_torch > 0.5
        preprocessing_info["vessels_mask"] = vessels_torch

        # --- Check alpha map ---
        alpha_torch = None
        if alpha is not None:
            assert alpha.shape[-3] == 2, (
                "Alpha field must have 2 channels for the y and x components of the vector field."
            )
            assert alpha.shape[-2:] == x.shape[-2:], (
                f"Alpha field shape {alpha.shape[-2:]} does not match fundus image shape {x.shape[-2:]}."
            )
            assert alpha.ndim == x.ndim, (
                "Alpha field must be a (2, h, w) matrix for a fundus image of shape (h, w)."
                if x.ndim == 4
                else "Alpha field must be a (B, 2, h, w) matrix for a batch of fundus images of shape (B, h, w)."
            )

            alpha_torch = (
                torch.tensor(alpha, device=device) if isinstance(alpha, np.ndarray) else alpha.to(device=device)
            )

        # --- Preprocess the fundus image ---
        if standard_resolution is not None and not (
            standard_resolution * 0.75 < x.shape[-1] < standard_resolution * 1.4
        ):
            if auto_resize:
                f = standard_resolution / x.shape[-1]  # Assume the image is cropped
                x = torch.nn.functional.interpolate(x, scale_factor=(f,) * 2, mode="bilinear")
                if alpha_torch is not None:
                    alpha_torch = torch.nn.functional.interpolate(alpha_torch, size=x.shape[-2:], mode="bilinear")
                preprocessing_info["scale_factor"] = f
            else:
                warnings.warn(
                    f"Image size {x.shape[-2:]} is not optimal for {model_name}.\n"
                    f"Consider resizing the image to a size close to {standard_resolution}x{standard_resolution}.",
                    stacklevel=2,
                )
        if alpha_torch is None:
            from steered_cnn.utils.preprocessing import compute_skeleton_field  # noqa: I001
            from fundus_vessels_toolkit.segment_to_graph.skeletonize import skeletonize

            def compute_alpha(vessels: torch.Tensor) -> torch.Tensor:
                if vessels.shape[-2:] != x.shape[-2:]:
                    mode = "area" if vessels.shape[-1] > x.shape[-1] == 2 else "bilinear"
                    vessels = vessels[None, None].to(torch.float32)  # Add batch and channel dimensions
                    vessels = torch.nn.functional.interpolate(vessels, size=x.shape[-2:], mode=mode) > 0.4
                    vessels = vessels.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
                vessels_np = vessels.numpy(force=True)
                # preprocessing_info["vessels"] = vessels_np
                skel = skeletonize(vessels_np)
                # preprocessing_info["skeleton"] = skel
                return torch.tensor(compute_skeleton_field(skel))

            if vessels_torch.ndim == 2:
                alpha_torch = compute_alpha(vessels_torch).unsqueeze(0)  # Add batch dimension
            elif vessels_torch.ndim == 3:
                alpha_torch = torch.stack([compute_alpha(v) for v in vessels_torch], dim=0)
            else:
                raise ValueError(f"Vessel mask must be a 2D or 3D array, got {vessels_torch.ndim} dimensions instead.")
            alpha_torch = alpha_torch.to(device=device)

        x = x.flip(1)
        x = torch.concatenate((gaussian_preprocess_torch(x), x), dim=1)

        return (x, alpha_torch), preprocessing_info

    def postprocess(*model_outputs: torch.Tensor, preprocessing_info: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
        (y,) = model_outputs
        if y.ndim == 3:
            y.unsqueeze_(1)

        # --- Rescale the output if necessary ---
        if "scale_factor" in preprocessing_info:
            f = preprocessing_info["scale_factor"]
            y = torch.nn.functional.interpolate(y, scale_factor=(1 / f,) * 2, mode="bilinear")
        y = torch.argmax(y, dim=1) if y.shape[1] > 1 else y.squeeze(1) > 0.5

        # --- Crop or pad the output to the final shape ---
        final_shape = preprocessing_info.get("final_shape", None)
        if final_shape is not None:
            y = crop_pad(y, final_shape[-2:])
            if len(final_shape) == 2:
                y = y.squeeze(0)

        # --- Apply the vessel mask if available ---
        if "vessels_mask" in preprocessing_info:
            vessels = preprocessing_info["vessels_mask"]
            y = torch.where(vessels, y + 1, 0)

        return (y,)

    fundus = TensorSpec("fundus_image", ("C", "H", "W"), description="The fundus image.")
    alpha = TensorSpec("alpha_field", ("uv", "H", "W"), dtype=torch.float32, optional=False)
    alpha = alpha.update(
        description="The vector field steering the model. Must be a 2-channel tensor providing the (y,x) components of the vector field."  # noqa: E501
    )
    alpha_opt = alpha.update(
        optional=True,
        description=alpha.description + " If not provided, it will be computed from the vessel segmentation mask.",
    )
    vessels_seg = TensorSpec(
        "vessels_mask",
        ("H", "W"),
        dtype=torch.bool,
        optional=True,
        description="The vessels segmentation mask used to mask the output and optionally compute the alpha field.",
    )
    output_info = (
        TensorSpec("AV", ("H", "W"), description="The vessel classification mask.\n0: Background; 1:Artery; 2:Vein."),
    )
    return PrePostProcessing(
        preprocess=preprocess,
        postprocess=postprocess,
        input_info=(fundus, vessels_seg, alpha_opt),
        model_input_info=(fundus, alpha),
        model_output_info=output_info,
        output_info=output_info,
    )
