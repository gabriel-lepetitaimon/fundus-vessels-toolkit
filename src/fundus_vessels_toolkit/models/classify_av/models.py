# This file is a simplified version of the original module found at:
#   https://github.com/ClementPla/avseg/tree/main/src/avseg/models


from functools import cache
from typing import Dict, List

import huggingface_hub
import segmentation_models_pytorch as smp
from huggingface_hub import CollectionItem, PyTorchModelHubMixin
from pytorch_lightning import LightningModule


class AVBaseModel(LightningModule, PyTorchModelHubMixin):
    """
    Base class for all models in the AVSeg project.
    Inherits from PyTorch Lightning's LightningModule.
    """

    def __init__(
        self,
        arch="unet",
        encoder_name="resnet34",
        task="multiclass",
        classes=3,
    ):
        super().__init__()
        if task == "multilabel":
            classes = 2
        self.arch = arch  # unet
        self.encoder_name = encoder_name
        self.task = task
        self.model = get_segmentation_models(
            arch=self.arch,
            encoder_name=self.encoder_name,
            num_classes=classes,
        )

        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor.
        :return: Model output.
        """
        return self.model(x)


########################################################################################################################
@cache
def available_models() -> List[CollectionItem]:
    """
    Get a list of available models in the collection.
    :return: List of available models.
    """
    COLLECTION = "ClementP/fundus-grading-665e582701ca1c80a0b5797a"
    return list(huggingface_hub.get_collection(COLLECTION).items)


def convert_collection_item_to_encoder(item):
    model_id = item.item_id
    encoder = model_id.split("-")[1]
    return (encoder, item)


@cache
def available_encoders() -> Dict[str, huggingface_hub.CollectionItem]:
    """
    Get a list of available encoders in the collection.
    :return: List of available encoders.
    """
    encoders = [convert_collection_item_to_encoder(item) for item in available_models()]
    return {k: v for k, v in encoders}


def get_segmentation_models(arch, encoder_name, num_classes=3):
    """
    Get a segmentation model based on the encoder name.
    :param encoder_name: Name of the encoder to retrieve.
    :return: Segmentation model.
    """
    item = available_encoders().get(encoder_name)
    assert item is not None, (
        f"Unknown encoder {encoder_name}. Available encoders are: {', '.join(available_encoders().keys())}."
    )

    encoder_smp = f"tu-{encoder_name}"
    model = smp.create_model(
        arch=arch,
        encoder_name=encoder_smp,
        classes=num_classes,
    )

    return model
