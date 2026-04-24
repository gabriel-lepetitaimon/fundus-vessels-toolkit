from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import psutil
import pytorch_lightning as L
import torch
import torch.nn as nn
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT_STR
from pydantic import BaseModel, ConfigDict, Field
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics import MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

import wandb
import yaml
from fundus_vessels_toolkit.models.metrics.tree import (
    MetricCollectionDict,
    ParentAcc,
    ParentCloseAcc,
    ParentSameSubtreeAcc,
    ParentSameSubtreeMeanDist,
    RootSensitivity,
    RootSpecificity,
)
from fundus_vessels_toolkit.models.topology.data import (
    BranchDigraphBatch,
    BranchDigraphData,
)
from fundus_vessels_toolkit.models.topology.dataset import BranchDigraphDataset, BranchDigraphDatasetConfig
from fundus_vessels_toolkit.models.topology.losses import (
    BranchContrastiveLoss,
    BranchContrastiveLossOpt,
    CrossEntropyLoss,
)
from fundus_vessels_toolkit.models.topology.model import BranchDigraphModel, BranchDigraphModelOpt

torch.set_float32_matmul_precision("medium")
torch.backends.fp32_precision = "ieee"  # type: ignore
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"  # type: ignore
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore


type TrainingSets = Literal["FundusAV", "HRF", "LES-AV", "MAPLES-DR", "DRIVE_train", "GAVE-train", "INSPIRE"]


class DigraphGNNTrainerConfig(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    dataset: BranchDigraphDatasetConfig = Field(default_factory=BranchDigraphDatasetConfig)
    model: BranchDigraphModelOpt = Field(default_factory=BranchDigraphModelOpt)
    contrastive_loss: BranchContrastiveLossOpt = Field(default_factory=BranchContrastiveLossOpt)

    training_set: str | list[TrainingSets] | None = Field(default=None)
    """Training set(s) to use. Can be a single dataset name, a list of dataset names, or None to use all datasets."""

    test_version: str | None = Field(default=None)
    """Version of the test set to use. If None, the same version as the training set will be used."""

    epoch: int = 160
    """Maximum number of training epochs."""

    lr: float = 1e-2
    """Learning rate."""

    batch_size: int = 3
    """Batch size for training."""


class HardwareConfig(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    def model_post_init(self, __context):
        cpu_count = psutil.cpu_count(logical=False) or 1
        if self.train_num_workers < 0:
            self.train_num_workers = max(0, cpu_count + self.train_num_workers + 1)
        if self.test_num_workers < 0:
            self.test_num_workers = max(0, cpu_count + self.test_num_workers + 1)

    max_batch_size: int = 3
    """Maximum batch size for training. If the batch_size in the config is larger than this, gradient accumulation will be used."""  # noqa: E501

    train_num_workers: int = -2
    """Number of workers for the training data loader."""

    test_batch_size: int = 6
    """Batch size for validation and testing."""

    test_num_workers: int = -2
    """Number of workers for the validation and testing data loader."""

    compile: bool = True
    """Whether to compile the model with torch.compile()."""

    precision: _PRECISION_INPUT_STR = "bf16-mixed"
    """Precision for training. Can be one of the following: "64-true", "32-true", "16-true", "16-mixed", "bf16-true", "bf16-mixed", "transformer-engine", "transformer-engine-float16"."""  # noqa: E501

    def batch_size_grad_acc(self, batch_size: int) -> tuple[int, int]:
        """Calculate the actual batch size and the number of gradient accumulation steps based on the given batch size and the maximum batch size."""  # noqa: E501
        if batch_size <= self.max_batch_size:
            return (batch_size, 1)
        else:
            grad_acc_steps = math.ceil(batch_size / self.max_batch_size)
            actual_batch_size = int(round(batch_size / grad_acc_steps))
            return (actual_batch_size, grad_acc_steps)


def train(config=None, hdw_cfg=None):
    wandb.init(project="GNN-topo-test", config=config)

    cfg = DigraphGNNTrainerConfig.model_validate(dict(wandb.config))
    cfg_dict = cfg.model_dump()

    if hdw_cfg is None:
        if Path("hardware_cfg.yaml").exists():
            try:
                with open("hardware_cfg.yaml", "r") as f:
                    hdw_cfg = HardwareConfig.model_validate(yaml.safe_load(f))
            except Exception as e:
                print(f"Error loading hardware config: {e}. Using default hardware config.")
                hdw_cfg = HardwareConfig()
        else:
            hdw_cfg = HardwareConfig()
    else:
        hdw_cfg = HardwareConfig.model_validate(hdw_cfg)

    # === DATASET ===
    dataset = BranchDigraphDataset("ALL_DATA_bundle.tar.gz", cfg=cfg.dataset)
    train_set, val_set, test_set = dataset.split_sets(train_ratio=0.7, val_ratio=0.15)

    if cfg.training_set is not None and cfg.training_set:
        train_set = train_set.select_dataset(cfg.training_set)
        val_set = val_set.select_dataset(cfg.training_set)
    if cfg.test_version is not None:
        val_set.cfg.graph_version = cfg.test_version
        test_set.cfg.graph_version = cfg.test_version

    batch_size, grad_acc = hdw_cfg.batch_size_grad_acc(cfg.batch_size)

    train_loader = PyGDataLoader(
        train_set.preload(with_image=False),
        shuffle=True,
        num_workers=hdw_cfg.train_num_workers,
        persistent_workers=True,
        batch_size=batch_size,
    )
    val_loader = PyGDataLoader(
        val_set.preload(with_image=False),
        batch_size=hdw_cfg.test_batch_size,
        num_workers=hdw_cfg.test_num_workers,
    )

    # Setup the logger and trainer
    wandb_logger = WandbLogger(log_model=True)
    model = DigraphGNNTrainer(cfg_dict, compile=hdw_cfg.compile, n_step_per_epoch=len(train_loader))

    checkpoints: list[Callback] = [ModelCheckpoint(monitor="val_agg", mode="max", save_weights_only=True)]

    trainer = L.Trainer(
        max_epochs=cfg.epoch,
        logger=wandb_logger,
        enable_progress_bar=True,
        check_val_every_n_epoch=20,
        accumulate_grad_batches=grad_acc,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",
        # num_sanity_val_steps=0,
        callbacks=checkpoints,
        precision=hdw_cfg.precision,
    )

    trainer.fit(model, train_loader, val_loader)

    test_loaders = {
        k: PyGDataLoader(v, batch_size=hdw_cfg.test_batch_size, num_workers=hdw_cfg.test_num_workers)
        for k, v in test_set.split_by_dataset().items()
    }
    model._test_dataloaders_names = list(test_loaders.keys())
    trainer.test(model, dataloaders=test_loaders, ckpt_path="best")

    # Finish the run
    wandb.finish()


class DigraphGNNTrainer(L.LightningModule):
    def __init__(self, config: DigraphGNNTrainerConfig | dict, compile: bool = False, n_step_per_epoch: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.config = DigraphGNNTrainerConfig.model_validate(config)
        self.n_step_per_epoch = n_step_per_epoch

        self.model = BranchDigraphModel(self.config.model, compile=compile)

        # === LOSSES ===
        self.fp_bce_loss = nn.BCEWithLogitsLoss()
        self.av_bce_loss = nn.BCEWithLogitsLoss()
        self.dir_bce_loss = nn.BCEWithLogitsLoss()
        self.root_bce_loss = nn.BCEWithLogitsLoss()
        self.line_ce_loss = CrossEntropyLoss(invalid_metagroup_penalty=0)
        self.line_contrastive_loss = BranchContrastiveLoss(self.config.contrastive_loss)

        # === METRICS ===
        self.val_metrics = self.metrics_collection(opti_tree=True)
        self.val_preds = {}
        self.test_metrics = self.metrics_collection(opti_tree=True)
        self.test_preds = {}
        self._test_dataloaders_names: list[str] | None = None

    def metrics_collection(self, opti_tree=True) -> MetricCollectionDict:
        collection = {
            "fp": MetricCollection(
                {
                    "-acc": Accuracy("binary"),
                    "-prec": Precision("binary"),
                    "-recall": Recall("binary"),
                },
            ),
            "av": MetricCollection(
                {
                    "-acc": Accuracy("binary"),
                    "-art-recall": Recall("binary"),
                    "-ven-recall": Specificity("binary"),
                },
            ),
            "dir": MetricCollection({"-acc": Accuracy("binary")}),
            "tree": MetricCollection(
                {
                    "-root-spe": RootSpecificity(),
                    "-root-sen": RootSensitivity(),
                    "-parent-acc": ParentAcc(ignore_root=False),
                    "-parent-same-subtree-acc": ParentSameSubtreeAcc(ignore_root=False),
                    "-parent-same-subtree-mean-dist": ParentSameSubtreeMeanDist(),
                    "-parent-1tol-acc": ParentCloseAcc(ignore_root=False),
                }
            ),
        }
        if opti_tree:
            collection["treeOpti"] = MetricCollection(
                {
                    "-root-spe": RootSpecificity(),
                    "-root-sen": RootSensitivity(),
                    "-parent-acc": ParentAcc(ignore_root=False),
                    "-parent-same-subtree-acc": ParentSameSubtreeAcc(ignore_root=False),
                    "-parent-same-subtree-mean-dist": ParentSameSubtreeMeanDist(),
                    "-parent-1tol-acc": ParentCloseAcc(ignore_root=False),
                }
            )
            collection["dirOpti"] = MetricCollection({"-acc": Accuracy("binary")})
            collection["avOpti"] = MetricCollection(
                {
                    "-acc": Accuracy("binary"),
                    "-art-recall": Recall("binary"),
                    "-ven-recall": Specificity("binary"),
                },
            )
        return MetricCollectionDict(collection)

    def update_metrics_collection(
        self, metrics: MetricCollectionDict, batched_out: BranchDigraphModel.Output, prefix="", suffix=""
    ):
        metric_values = dict()

        for out in batched_out.unbatch():
            assert isinstance(out, BranchDigraphModel.Output)
            # === AV metrics ===
            metric_values["fp"] = metrics["fp"](out.fp_p, out.gt_fp_p > 0.5)

            tp_mask = out.gt_fp_p < 0.5
            metric_values["av"] = metrics["av"](out.av_p[tp_mask], out.gt_av_p[tp_mask] > 0.5)

            # === Direction metrics ===
            assert BranchDigraphData.has_gt(out.batch)
            dir_p, dir_gt_p = out.dir_p, out.batch.branch_dir_p
            metric_values["dir"] = metrics["dir"](dir_p[tp_mask], dir_gt_p[tp_mask] > 0.5)

            # === Parent classification metrics ===
            metric_values["tree"] = metrics["tree"](out.max_parent(use_gt=True), out.gt_parent, tp_mask)

            if "treeOpti" in metrics:
                # === Optimal parent classification metrics ===
                opti_parent, opti_dir, opti_av_logit = out.optimal_tree
                metric_values["dirOpti"] = metrics["dirOpti"](opti_dir[tp_mask], dir_gt_p[tp_mask] > 0.5)
                metric_values["treeOpti"] = metrics["treeOpti"](opti_parent, out.gt_parent, tp_mask)

                opti_av_p = opti_av_logit[tp_mask].sigmoid()
                metric_values["avOpti"] = metrics["avOpti"](opti_av_p, out.gt_av_p[tp_mask] > 0.5)

        # Flatten metric values dict
        metric_values = {prefix + k1 + k2 + suffix: v for k1, group in metric_values.items() for k2, v in group.items()}
        return metric_values

    def update_preds(self, preds_dict: dict, batched_out: BranchDigraphModel.Output, optimal=False):
        for out in batched_out.unbatch():
            if "table" not in preds_dict:
                columns = ["name", "parent", "dir"]
                if optimal:
                    columns += ["parentOpti", "dirOpti", "avOpti"]
                preds_dict["table"] = wandb.Table(columns=columns)
            data = [out.name, out.max_parent(use_gt=False).cpu().tolist(), (out.dir_logit > 0).cpu().int().tolist()]
            if optimal:
                opti_parent, opti_dir = out.optimal_tree[:2]
                data += [opti_parent.cpu().tolist(), opti_dir.cpu().int().tolist()]
            preds_dict["table"].add_data(*data)

    def forward(self, data: BranchDigraphBatch) -> BranchDigraphModel.Output:
        return self.model(data)

    def losses(self, out: BranchDigraphModel.Output):
        # AV loss
        fp_loss = self.fp_bce_loss(out.fp_logit, out.gt_fp_p)

        tp_mask = out.gt_fp_p < 0.5
        av_loss = self.av_bce_loss(out.av_logit[tp_mask], out.gt_av_p[tp_mask])

        # Dir losses
        dir_loss = self.dir_bce_loss(out.dir_logit[tp_mask], out.gt_dir_p[tp_mask])

        # Line loss
        # root_loss = self.root_bce_loss(out.root_logit[tp_mask], out.gt_root_p[tp_mask])
        mask = out.lines_mask(filter_dir="gt")
        line_loss = self.line_ce_loss(
            x=out.lines_logit[mask],
            target=out.gt_lines_score[mask],
            group_idx=out.lines[mask].b1,
            mask=tp_mask,
            metagroup_idx=out.batch.branch_subtree_idx,
            other_group_idx=out.lines[mask].b0,
        )

        contrastive_losses = self.line_contrastive_loss(out)
        contrastive_loss = contrastive_losses.pop("loss")

        loss = fp_loss + av_loss + dir_loss + line_loss + contrastive_loss * 0.1
        return (
            {
                "fp_loss": fp_loss,
                "av_loss": av_loss,
                "dir_loss": dir_loss,
                # "root_loss": root_loss,
                "line_loss": line_loss,
            }
            | contrastive_losses
            | {"loss": loss}
        )

    def training_step(self, batch, batch_idx):
        model_out = self(batch)
        losses = self.losses(model_out)
        self.log_dict({k: loss for k, loss in losses.items()}, batch_size=batch.num_graphs, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        model_out: BranchDigraphModel.Output = self(batch)

        losses = self.losses(model_out)
        self.log_dict(
            {"val_" + k: loss for k, loss in losses.items()}, batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        val_metrics = self.update_metrics_collection(self.val_metrics, model_out, prefix="val_")
        self.log_dict(val_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

        self.update_preds(self.val_preds, model_out)

    def on_validation_end(self) -> None:
        self.logger.experiment.log(  # type: ignore
            {
                "val_agg": self.trainer.callback_metrics["val_tree-parent-acc"]
                         * self.trainer.callback_metrics["val_av-acc"]
                         * self.trainer.callback_metrics["val_dir-acc"],
                "running_lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "epoch": self.trainer.current_epoch,
                "val_pred": self.val_preds["table"],
            }
        )
        self.val_preds = {}
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        model_out = self(batch)
        losses = self.losses(model_out)
        if self._test_dataloaders_names is not None:
            suffix = "/" + self._test_dataloaders_names[dataloader_idx] + "_"
        else:
            suffix = ""
        self.log_dict(
            {"test_" + k + suffix: loss for k, loss in losses.items()},
            batch_size=batch.num_graphs,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=self._test_dataloaders_names is not None,
        )
        test_metrics = self.update_metrics_collection(self.test_metrics, model_out, prefix="test_", suffix=suffix)
        self.log_dict(test_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

        self.update_preds(self.test_preds, model_out)

    def on_test_end(self) -> None:
        self.logger.experiment.log({"test_pred": self.test_preds["table"]})  # type: ignore
        self.test_preds = {}
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.lr, epochs=self.config.epoch, steps_per_epoch=self.n_step_per_epoch
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}


if __name__ == "__main__":
    train()


class _BatchSizeGradAccType(TypedDict):
    batch_size: int
    accumulate_grad_batches: NotRequired[int]
