from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics import MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

import wandb
from fundus_vessels_toolkit.models.branch_digraph_gnn.data import (
    BranchDigraphBatch,
    BranchDigraphData,
    BranchDigraphDataset,
)
from fundus_vessels_toolkit.models.branch_digraph_gnn.losses import BranchContrastiveLoss, CrossEntropyLoss
from fundus_vessels_toolkit.models.branch_digraph_gnn.model import BranchDigraphModel, BranchDigraphModelOpt
from fundus_vessels_toolkit.models.metrics.tree import (
    MetricCollectionDict,
    ParentAcc,
    ParentCloseAcc,
    ParentSameSubtreeAcc,
    ParentSameSubtreeMeanDist,
    RootSensitivity,
    RootSpecificity,
)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.conv.fp32_precision = "tf32"  # type: ignore
torch.backends.cuda.matmul.fp32_precision = "tf32"


def train():
    # Initialize a new W&B run (wandb.agent handles the config)
    wandb.init(project="GNN-topo-test")

    # Access the hyperparms assigned to this specific run
    config = DigraphGNNTrainerConfig.model_validate(dict(wandb.config))
    config_dict = config.model_dump()

    # === DATASET ===
    PATH = [
        Path("/run/media/gaby/GREY SSD/PostDoc/DATA/Fundus/" + folder)
        for folder in ["GAVE-train", "MAPLES-DR", "Fundus-AV", "LES-AV", "INSPIRE"]
    ]
    RAW = [path / "1-images" for path in PATH]
    AV = [path / "2-av-pred_CLEMENT" for path in PATH]
    TOPO = [path / "3-topo" for path in PATH]
    dataset = BranchDigraphDataset.load_from_dirs(
        RAW,
        TOPO,
        av_dir=AV,
        resize_to=1024,
        root=str(Path(__file__).parent / "tmp/DATA2"),
        overwrite=False,
        # ignore_recent=datetime(2026, 2, 19),
        ignore_recent=datetime(2026, 3, 8),
    )
    train_set, val_set, test_set = dataset.split_loaders(train_ratio=0.7, val_ratio=0.15)
    train_loader = PyGDataLoader(train_set, batch_size=3, shuffle=True, num_workers=5)
    val_loader = PyGDataLoader(val_set, batch_size=6, num_workers=2)

    #
    # Setup the logger and trainer
    wandb_logger = WandbLogger(log_model=True)  # logs model checkpoints
    model = DigraphGNNTrainer(config_dict, n_step_per_epoch=len(train_loader))

    checkpoints: list[Callback] = [ModelCheckpoint(monitor="val_tree-parent-acc", mode="max")]

    trainer = L.Trainer(
        max_epochs=config.epoch,
        logger=wandb_logger,
        enable_progress_bar=True,  # Optional: cleaner console output during sweeps
        check_val_every_n_epoch=20,
        accumulate_grad_batches=2,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",wandb
        # num_sanity_val_steps=0,
        callbacks=checkpoints,
        precision="bf16-mixed",
    )

    trainer.fit(model, train_loader, val_loader)

    test_loader = PyGDataLoader(test_set, batch_size=6, num_workers=2)
    trainer.test(model, dataloaders=[test_loader], ckpt_path="best")

    # Finish the run
    wandb.finish()


class DigraphGNNTrainerConfig(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    model: BranchDigraphModelOpt = Field(default_factory=BranchDigraphModelOpt)

    epoch: int = 160
    """Maximum number of training epochs."""

    lr: float = 1e-2
    """Learning rate."""


class DigraphGNNTrainer(L.LightningModule):
    def __init__(self, config: DigraphGNNTrainerConfig | dict, n_step_per_epoch: int = 64):
        super().__init__()
        self.save_hyperparameters()
        self.config = DigraphGNNTrainerConfig.model_validate(config)
        self.n_step_per_epoch = n_step_per_epoch

        self.model = BranchDigraphModel(self.config.model)

        # === LOSSES ===
        self.fp_bce_loss = nn.BCEWithLogitsLoss()
        self.av_bce_loss = nn.BCEWithLogitsLoss()
        self.dir_bce_loss = nn.BCEWithLogitsLoss()
        self.root_bce_loss = nn.BCEWithLogitsLoss()
        self.line_ce_loss = CrossEntropyLoss(invalid_metagroup_penalty=0)
        self.line_contrastive_loss = BranchContrastiveLoss()

        # === METRICS ===
        self.val_metrics = self.metrics_collection(opti_tree=True)
        self.val_preds = {}
        self.test_metrics = self.metrics_collection(opti_tree=True)
        self.test_preds = {}

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
        self, metrics: MetricCollectionDict, batched_out: BranchDigraphModel.Output, prefix=""
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
        metric_values = {prefix + k1 + k2: v for k1, group in metric_values.items() for k2, v in group.items()}
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

        loss = fp_loss + av_loss + dir_loss + line_loss + contrastive_losses["triplet_loss"] * 0.1
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
        self.logger.experiment.log({"val_pred": self.val_preds["table"]})  # type: ignore
        self.val_preds = {}
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        model_out = self(batch)
        losses = self.losses(model_out)
        self.log_dict(
            {"test_" + k: loss for k, loss in losses.items()}, batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        test_metrics = self.update_metrics_collection(self.test_metrics, model_out, prefix="test_")
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
            optimizer, max_lr=self.config.lr, epochs=self.config.epoch, steps_per_epoch=54
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}


if __name__ == "__main__":
    train()
