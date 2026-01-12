########################################################################################################################
#   *** AV Segmentation ***
#   This module provides function and pretrained models for artery/vein segmentation on fundus images.
#
########################################################################################################################
from __future__ import annotations

__all__ = ["segment_av_model", "segment_av", "SegmentAVModel"]

from enum import Enum
from typing import Any, Dict, Literal, Tuple

import torch

from fundus_toolkits.models.cache import cache_model
from fundus_toolkits.models.generic_inference import fundus_inference
from fundus_toolkits.models.pre_postprocessing import PrePostProcessing
from fundus_toolkits.utils.torch import DeviceLikeType


class SegmentAVModel(str, Enum):
    """
    The available pretrained models for vessel segmentation.
    """

    MULTITASK_IMAGENET = "multitask_imagenet"
    MULTITASK_FUNDUS = "multitask_fundus"
    MULTILABEL_IMAGENET = "multilabel_imagenet"
    MULTILABEL_FUNDUS = "multilabel_fundus"
    MULTILABEL_RANDOM = "multilabel_random"

    @classmethod
    def by_type(
        cls,
        model_type: Literal["multitask", "multilabel"],
        finetuned_from: Literal["imagenet", "fundus", "random"] = "imagenet",
    ) -> SegmentAVModel:
        """
        Returns the available pretrained models for vessel classification.
        Parameters
        ----------
        model_type : Literal["multitask", "multilabel"]
            The type of model to use.
        finetuned_from : Literal["imagenet", "fundus", "random"], optional
            The type of pretraining to use. Defaults to "imagenet".
        Returns
        -------
        ClassifyAVModel
            The available pretrained models for vessel classification.
        """
        match (model_type, finetuned_from):
            case ("multitask", "imagenet"):
                return SegmentAVModel.MULTITASK_IMAGENET
            case ("multitask", "fundus"):
                return SegmentAVModel.MULTITASK_FUNDUS
            case ("multilabel", "imagenet"):
                return SegmentAVModel.MULTILABEL_IMAGENET
            case ("multilabel", "fundus"):
                return SegmentAVModel.MULTILABEL_FUNDUS
            case ("multilabel", "random"):
                return SegmentAVModel.MULTILABEL_RANDOM
            case ("multitask", "random"):
                raise ValueError("No available model for multitask with random initialization.")
            case _:
                raise ValueError(
                    f"Unknown model type: {model_type} or finetuned from: {finetuned_from}.\n"
                    f"Available model types are: 'multitask', 'multilabel'.\n"
                    f"Available finetuned from are: 'imagenet', 'fundus', 'random'."
                )

    @classmethod
    def is_multilabel(cls, model: SegmentAVModels) -> bool:
        """
        Returns whether the model is multitask or multilabel.
        Parameters
        ----------
        model : str
            The model to check.
        Returns
        -------
        bool
            True if the model is multitask, False if it is multilabel.
        """
        match cls(model):
            case cls.MULTITASK_IMAGENET | cls.MULTITASK_FUNDUS:
                return False
            case cls.MULTILABEL_IMAGENET | cls.MULTILABEL_FUNDUS | cls.MULTILABEL_RANDOM:
                return True
            case _:
                raise ValueError(
                    f"Unknown model: {model}.\nAvailable models are: {', '.join(_.value for _ in SegmentAVModel)}."
                )


type SegmentAVModels = (
    Literal["multitask_imagenet", "multitask_fundus", "multilabel_imagenet", "multilabel_fundus", "multilabel_random"]
    | SegmentAVModel
)


@fundus_inference("av")
def segment_av(
    fundus: torch.Tensor,
    *,
    model: SegmentAVModels = SegmentAVModel.MULTILABEL_FUNDUS,
    device: DeviceLikeType | Literal["auto"] = "auto",
    ignore_segmentation: bool = False,
) -> torch.Tensor:
    """
    Segments the fundus image using the specified model.

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
        The segmentation map(s) as a tensor or array of shape (C, H, W) or (B, C, H, W).

        If the input fundus was provided as a tensor, the output will be a tensor. In all other cases, the output will be a numpy array. If FundusData object(s) were provided, they will be updated in-place with the segmentation result.
    """  # noqa: E501
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Fetch model by name ---
    net = segment_av_model(model).to(device=device)
    pre_post_process = segment_av_pre_postprocessing(model)

    # --- Apply preprocess, model and postprocess ---
    x, preprocessing_info = pre_post_process.preprocess(fundus, device=device)
    y = net(*x)
    (y_proba,) = pre_post_process.postprocess(y, preprocessing_info=preprocessing_info)

    # --- Convert probabilities to labels ---
    if SegmentAVModel.is_multilabel(model):
        a_map = y_proba[..., 2, :, :]  # Artery
        v_map = y_proba[..., 1, :, :]  # Vein
        if ignore_segmentation:
            av_pred = torch.ones_like(a_map, dtype=torch.uint8)  # Artery by default
            av_pred[v_map > a_map] = 2  # Vein
            av_pred[abs(v_map - a_map) < 0.05] = 3  # Unknown
        else:
            av_pred = ((a_map > 0.5) * 1 + (v_map > 0.5) * 2).to(torch.uint8)
    else:
        if ignore_segmentation:
            av_pred = torch.ones_like(y_proba[..., 0, :, :], dtype=torch.uint8)  # Artery by default
            av_pred[y_proba[..., 2, :, :] > y_proba[..., 1, :, :]] = 2  # Vein
            av_pred[abs(y_proba[..., 2, :, :] - y_proba[..., 1, :, :]) < 0.05] = 3  # Unknown
        else:
            av_lookup = torch.tensor([0, 2, 1], device=y_proba.device, dtype=torch.uint8)  # Background, Vein, Artery
            av_pred = y_proba.argmax(-3).int()
            av_pred = av_lookup[av_pred.flatten()].reshape(av_pred.shape)

    return av_pred


@cache_model(max_size=2)
def segment_av_model(model: SegmentAVModels = SegmentAVModel.MULTILABEL_FUNDUS) -> torch.nn.Module:
    """
    Loads a pretrained model for vessel segmentation.

    Parameters
    ----------
    model : ClassifyAVModel, optional
        The model to use for AV segmentation. See ``ClassifyAVModel`` for available models.
        Defaults to ClassifyAVModel.MULTILABEL_IMAGENET.

    device : torch.device, optional
        The device to use for computation. Defaults to "cuda".

    Returns
    -------
    torch.nn.Module
        The segmentation model.
    """
    from .models_src.classify_av_models import AVBaseModel

    model = SegmentAVModel(model)
    match model:
        case (
            SegmentAVModel.MULTITASK_IMAGENET
            | SegmentAVModel.MULTITASK_FUNDUS
            | SegmentAVModel.MULTILABEL_IMAGENET
            | SegmentAVModel.MULTILABEL_FUNDUS
            | SegmentAVModel.MULTILABEL_RANDOM
        ):  # Clement AV Models: https://github.com/ClementPla/avseg/tree/main/src/avseg
            task, pretrain = model.value.split("_")
            match task:
                case "multitask":
                    encoder_name = "seresnet50"
                case "multilabel":
                    encoder_name = "seresnext50_32x4d"
                case _:
                    raise ValueError(f"Unknown task: {task}. Available tasks are: 'multitask', 'multilabel'.")
            revision = f"{task}-{encoder_name}-{pretrain}"

        case _:
            raise ValueError(
                f"Unknown model: {model}.\nAvailable models are: {', '.join(_.value for _ in SegmentAVModel)}."
            )

    return AVBaseModel.from_pretrained("ClementP/AVSeg", revision=revision)


def segment_av_pre_postprocessing(model: SegmentAVModels = SegmentAVModel.MULTILABEL_FUNDUS) -> PrePostProcessing:
    """
    Returns the pre and post processing for a given classification model.

    Parameters
    ----------
    model : ClassifyAVModel, optional
        The model to use for AV segmentation. See ``ClassifyAVModel`` for available models.
        Defaults to ClassifyAVModel.MULTILABEL_IMAGENET.

    Returns
    -------
    PrePostProcessing
        The pre and post processing for the classification model.
    """
    model = SegmentAVModel(model)
    match model:
        case SegmentAVModel.MULTITASK_IMAGENET | SegmentAVModel.MULTITASK_FUNDUS:
            return clement_pre_postprocessing(model, multitask=True)
        case SegmentAVModel.MULTILABEL_IMAGENET | SegmentAVModel.MULTILABEL_FUNDUS | SegmentAVModel.MULTILABEL_RANDOM:
            return clement_pre_postprocessing(model, multitask=False)
        case _:
            raise ValueError(
                f"Unknown model: {model}.\nAvailable models are: {', '.join(_.value for _ in SegmentAVModel)}."
            )


########################################################################################################################
#   *** PRE and POST PROCESSING FOR MULTITASK AND MULTILABEL MODELS ***
#   Used for models:
#       - ClassifyAVModel.MULTITASK_IMAGENET
#       - ClassifyAVModel.MULTITASK_FUNDUS
#       - ClassifyAVModel.MULTILABEL_IMAGENET
#       - ClassifyAVModel.MULTILABEL_FUNDUS
#       - ClassifyAVModel.MULTILABEL_RANDOM
#
########################################################################################################################
def clement_pre_postprocessing(model_name: str, multitask: bool = False) -> PrePostProcessing:
    """
    Returns the pre and post processing for the classification model.

    Returns
    -------
    PrePostProcessing
        The pre and post processing for the classification model.
    """
    from fundus_toolkits.models.pre_postprocessing import basic_fundus_pre_postprocessing

    basic_pre_postprocess = basic_fundus_pre_postprocessing(
        1024,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        model_name=model_name,
        final_activation="softmax" if multitask else "sigmoid",
        output_channels=["background", "artery", "vein"],
    )

    if not multitask:
        basic_postprocess = basic_pre_postprocess.postprocess

        def postprocess(*model_outputs: torch.Tensor, preprocessing_info: Dict[str, Any]) -> Tuple[torch.Tensor, ...]:
            (pred,) = basic_postprocess(*model_outputs, preprocessing_info=preprocessing_info)
            background = torch.min(1 - pred, dim=-3, keepdim=True).values
            pred = torch.cat([background, pred], dim=-3)
            return (pred,)

        basic_pre_postprocess = basic_pre_postprocess.update(postprocess=postprocess)

    return basic_pre_postprocess
