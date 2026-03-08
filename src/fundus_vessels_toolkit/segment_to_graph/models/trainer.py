from __future__ import annotations

import pytorch_lightning as L
import torch
import torch.nn as nn
from torchmetrics import MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

import wandb

from .dataset import VBranchDigraphBatch
from .losses import BranchContrastiveLoss, CrossEntropyLoss
from .metrics import (
    MetricCollectionDict,
    ParentAcc,
    ParentCloseAcc,
    ParentSameSubtreeAcc,
    ParentSameSubtreeMeanDist,
    RootSensitivity,
    RootSpecificity,
)
from .model import BranchDigraphModel, BranchFeaturesEfficientNetV2S, Gatv2GCN, TransformerGCN


# Define your LightningModule
class DigraphGNNTrainer(L.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        # Access hyperparameters from wandb.config
        self.config = config if config is not None else {}
        self.model = BranchDigraphModel(
            BranchFeaturesEfficientNetV2S(), TransformerGCN(n_in=784, n_out=512, edge_attr_dim=7)
        )

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
        # register metrics to be properly reset at each epoch end and moved to the right device

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

    def forward(self, data: VBranchDigraphBatch) -> BranchDigraphModel.Output:
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

        loss = fp_loss + av_loss + dir_loss + line_loss + sum(contrastive_losses.values()) * 0.2
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
        self.log_dict({k: l for k, l in losses.items()}, batch_size=batch.num_graphs, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        model_out: BranchDigraphModel.Output = self(batch)

        losses = self.losses(model_out)
        self.log_dict(
            {"val_" + k: l for k, l in losses.items()}, batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        val_metrics = self.update_metrics_collection(self.val_metrics, model_out, prefix="val_")
        self.log_dict(val_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

        self.update_preds(self.val_preds, model_out)

    def on_validation_end(self) -> None:
        self.logger.experiment.log({"val_pred": self.val_preds["table"]})
        self.val_preds = {}
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        model_out = self(batch)
        test_metrics = self.update_metrics_collection(self.test_metrics, model_out, prefix="test_")
        self.log_dict(test_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

        self.update_preds(self.test_preds, model_out)

    def on_test_end(self) -> None:
        self.logger.experiment.log({"test_pred": self.test_preds["table"]})
        self.test_preds = {}
        self.test_metrics.reset()

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.get("lr"), epochs=self.config.get("epoch"), steps_per_epoch=54
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}
