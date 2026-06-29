from __future__ import annotations

import gc
import math
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

import psutil
import pytorch_lightning as L
import torch
import torch.nn as nn
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT_STR
from pydantic import BaseModel, ConfigDict, Field
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics import MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

import wandb
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
from fundus_vessels_toolkit.models.topology.model import BranchDigraphModel, BranchDigraphModelCfg
from fundus_vessels_toolkit.utils.nnet.experiment import ExpCfgBaseModel, ExperimentRunFactory
from fundus_vessels_toolkit.utils.nnet.optuna import ListLiteralHyperParam
from fundus_vessels_toolkit.utils.nnet.pydantic_yaml import model_validate_yaml_file

# torch.set_float32_matmul_precision("medium")
torch.backends.fp32_precision = "ieee"  # type: ignore
torch.backends.cuda.matmul.fp32_precision = "ieee"
torch.backends.cudnn.fp32_precision = "ieee"  # type: ignore
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore


type TrainingSets = Literal["FundusAV", "HRF", "LES-AV", "MAPLES-DR", "DRIVE_train", "GAVE-train", "INSPIRE"]
type GraphVersion = Literal["fvt", "automorph", "vesx", "all", "training"]


class DigraphGNNTrainerConfig(ExpCfgBaseModel):
    dataset: BranchDigraphDatasetConfig = Field(default_factory=BranchDigraphDatasetConfig)
    model: BranchDigraphModelCfg = Field(default_factory=BranchDigraphModelCfg)
    contrastive_loss: BranchContrastiveLossOpt = Field(default_factory=BranchContrastiveLossOpt)
    topo_losses_delay: int = 0
    """Number of epochs to delay the topology losses (line loss and contrastive loss) to allow the model to first learn to classify AV and direction before learning the topology. This can help stabilize the training and improve the final performance."""  # noqa: E501

    training_set: Annotated[list[TrainingSets], ListLiteralHyperParam(TrainingSets)] | None = Field(default=None)
    """Training set(s) to use. Can be a single dataset name, a list of dataset names, or None to use all datasets."""

    test_version: GraphVersion = Field(default="training")
    """Version of the test set to use. If None, the same version as the training set will be used."""

    epoch: int = 160
    """Maximum number of training epochs."""

    lr: float = 1e-2
    """Learning rate."""

    batch_size: int = 12
    """Batch size for training."""


class _GPU_Specs(TypedDict):
    accelerator: NotRequired[Literal["gpu"]]
    devices: NotRequired[int | list[int]]


class HardwareConfig(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    def model_post_init(self, __context):
        cpu_count = psutil.cpu_count(logical=False) or 1
        if self.train_num_workers < 0:
            self.train_num_workers = max(0, cpu_count + self.train_num_workers + 1)
        if self.test_num_workers < 0:
            self.test_num_workers = max(0, cpu_count + self.test_num_workers + 1)

    def gpu_specs(self) -> _GPU_Specs:
        specs: _GPU_Specs = {"accelerator": "gpu"}
        if self.gpu is not None:
            specs["devices"] = self.gpu
        return specs

    @classmethod
    def current(cls) -> HardwareConfig:
        return _hardware_config.get() or HardwareConfig()

    max_batch_size: int = 4
    """Maximum batch size for training. If the batch_size in the config is larger than this, gradient accumulation will be used."""  # noqa: E501

    val_every_n_epoch: int = 10
    """Number of epochs between each validation. If 0, validation will be done only at the end of training."""

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

    progress_bar: bool = False
    """Whether to show the progress bar during training. Can be useful to disable it when running in a non-interactive environment."""  # noqa: E501

    gpu: int | list[int] | None = None
    """GPU device index to use. If None, the default GPU will be used."""

    preload_with_img: bool = True
    """Whether to preload the dataset with images. If False, the dataset will be preloaded without images, which can save memory but slow down the training."""  # noqa: E501

    def batch_size_grad_acc(self, batch_size: int) -> tuple[int, int]:
        """Calculate the actual batch size and the number of gradient accumulation steps based on the given batch size and the maximum batch size."""  # noqa: E501
        if batch_size <= self.max_batch_size:
            return (batch_size, 1)
        else:
            actual_batch_size = 1
            for i in reversed(range(2, self.max_batch_size + 1)):
                if batch_size % i == 0:
                    actual_batch_size = i
                    break
            grad_acc_steps = batch_size // actual_batch_size
            return (actual_batch_size, grad_acc_steps)


_hardware_config: ContextVar[HardwareConfig | None] = ContextVar("_hardware_config", default=None)


def load_hardware_config(cfg: Path | str | dict | None) -> HardwareConfig:
    if cfg is None:
        default_path = Path("hardware_cfg.yaml")
        if default_path.exists():
            return model_validate_yaml_file(default_path, HardwareConfig)

    if isinstance(cfg, (Path, str)):
        cfg = Path(cfg)
        if cfg.exists():
            return model_validate_yaml_file(cfg, HardwareConfig)
        else:
            return HardwareConfig()

    hdw_cfg = HardwareConfig.model_validate(cfg)
    _hardware_config.set(hdw_cfg)
    return hdw_cfg


def train(experiment: ExperimentRunFactory[DigraphGNNTrainerConfig], hdw_cfg=None):
    hdw_cfg = load_hardware_config(hdw_cfg)

    with experiment as exp_run:
        cfg = exp_run.cfg

        # === DATASET ===
        dataset = BranchDigraphDataset("ALL_DATA_bundle.tar.gz", cfg=cfg.dataset)
        train_set, val_set, test_set = dataset.split_sets(train_ratio=0.7, val_ratio=0.15)

        if cfg.training_set is not None and cfg.training_set:
            train_set = train_set.select_dataset(cfg.training_set)
            val_set = val_set.select_dataset(cfg.training_set)
        if cfg.test_version not in ("training", "all"):
            val_set.cfg.graph_version = cfg.test_version

        batch_size, grad_acc = hdw_cfg.batch_size_grad_acc(cfg.batch_size)

        train_loader = PyGDataLoader(
            train_set.preload(with_image=hdw_cfg.preload_with_img),
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
        gc.freeze()

        # Setup the logger and trainer
        n_step_per_epoch = math.ceil(len(train_loader) / grad_acc)

        model = DigraphGNNTrainer(cfg.model_dump(), compile=hdw_cfg.compile, n_step_per_epoch=n_step_per_epoch)

        checkpoint = ModelCheckpoint(monitor="val_agg", mode="max", save_weights_only=True)

        trainer = L.Trainer(
            max_epochs=cfg.epoch,
            logger=exp_run.logger,
            enable_progress_bar=exp_run.header.progress_bar,
            check_val_every_n_epoch=hdw_cfg.val_every_n_epoch,
            accumulate_grad_batches=grad_acc,
            # gradient_clip_val=0.5,
            # gradient_clip_algorithm="value",
            # num_sanity_val_steps=0,
            callbacks=[checkpoint],
            precision=hdw_cfg.precision,
            **hdw_cfg.gpu_specs(),
        )

        trainer.fit(model, train_loader, val_loader)

        if checkpoint.best_model_score is not None:
            exp_run.tell(checkpoint.best_model_score.item())

        test_args = dict(batch_size=hdw_cfg.test_batch_size, num_workers=hdw_cfg.test_num_workers)
        if cfg.test_version == "all":
            test_loaders = {
                f"{k}-{v}": PyGDataLoader(d.use_version(v), **test_args)  # type: ignore
                for k, d in test_set.split_by_dataset().items()
                for v in test_set.list_versions()
            }
        elif cfg.test_version == "training" and isinstance(train_set.cfg.graph_version, dict):
            test_loaders = {
                f"{k}-{v}": PyGDataLoader(d.use_version(v), **test_args)  # type: ignore
                for k, d in test_set.split_by_dataset().items()
                for v in train_set.cfg.graph_version.keys()
            }
        else:
            test_version = cfg.test_version if cfg.test_version != "training" else cfg.dataset.graph_version
            test_set.cfg.graph_version = test_version
            test_loaders = {k: PyGDataLoader(d, **test_args) for k, d in test_set.split_by_dataset().items()}  # type: ignore

        model._test_dataloaders_names = list(test_loaders.keys())
        trainer.test(model, dataloaders=test_loaders, ckpt_path="best")


class DigraphGNNTrainer(L.LightningModule):
    def __init__(self, config: DigraphGNNTrainerConfig | dict, compile: bool = False, n_step_per_epoch: int = 64):
        super().__init__()
        self.save_hyperparameters(logger=False)
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

        if self.config.topo_losses_delay > 0:
            delay_coef = (self.current_epoch - self.config.topo_losses_delay) / self.config.topo_losses_delay
            delay_coef = max(min(1.0, delay_coef), 0)
        else:
            delay_coef = 1.0
        loss = fp_loss + av_loss + dir_loss + (line_loss + contrastive_loss * 0.1) * delay_coef
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

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log("val_agg", metrics["tree"]["-parent-acc"] * metrics["av"]["-acc"] * metrics["dir"]["-acc"])
        self.val_metrics.reset()

    def on_validation_end(self) -> None:
        self.logger.experiment.log(  # type: ignore
            {
                "running_lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "epoch": self.trainer.current_epoch,
                "val_pred": self.val_preds["table"],
            }
        )
        self.val_preds = {}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        model_out = self(batch)
        losses = self.losses(model_out)
        if self._test_dataloaders_names is not None:
            suffix = "/" + self._test_dataloaders_names[dataloader_idx]
        else:
            suffix = ""
        self.log_dict(
            {"test_" + k + suffix: loss for k, loss in losses.items()},
            batch_size=batch.num_graphs,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=suffix == "",
        )
        test_metrics = self.update_metrics_collection(self.test_metrics, model_out, prefix="test_", suffix=suffix)
        self.log_dict(
            test_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True, add_dataloader_idx=suffix == ""
        )

        self.update_preds(self.test_preds, model_out)

    def on_test_end(self) -> None:
        self.logger.experiment.log({"test_pred": self.test_preds["table"]})  # type: ignore
        self.test_preds = {}
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr / 25)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.lr,
            epochs=self.config.epoch,
            steps_per_epoch=self.n_step_per_epoch,
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "train_loss", "interval": "step", "frequency": 1}]


class _BatchSizeGradAccType(TypedDict):
    batch_size: int
    accumulate_grad_batches: NotRequired[int]
