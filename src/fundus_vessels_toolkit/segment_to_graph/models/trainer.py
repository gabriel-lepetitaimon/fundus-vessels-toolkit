from __future__ import annotations

from typing import Iterable

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric, MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

from ...utils.math import lerp
from ...utils.torch import GroupByCrossEntropyLoss
from .digraph_model import BranchDigraphModel, BranchFeaturesEfficientNetV2S, Gatv2GCN, optimal_parent_gt


class MetricCollectionDict(nn.ModuleDict):
    _modules: dict[str, Metric]  # type: ignore[assignment]

    def reset(self):
        for metric in self.values():
            metric.reset()

    def items(self) -> Iterable[tuple[str, Metric]]:  # type: ignore
        return super().items()  # type: ignore

    def values(self) -> Iterable[Metric]:  # type: ignore
        return super().values()  # type: ignore

    def __getitem__(self, key: str) -> Metric:  # type: ignore
        return super().__getitem__(key)  # type: ignore


# Define your LightningModule
class DigraphGNNTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Access hyperparameters from wandb.config
        self.config = config
        self.model = BranchDigraphModel(BranchFeaturesEfficientNetV2S(), Gatv2GCN(n_in=784, n_out=512))

        # === LOSSES ===
        self.fp_bce_loss = nn.BCEWithLogitsLoss()
        self.av_bce_loss = nn.BCEWithLogitsLoss()
        self.dir_bce_loss = nn.BCEWithLogitsLoss()
        self.line_ce_loss = GroupByCrossEntropyLoss()

        # === METRICS ===
        self.val_metrics = self.metrics_collection()
        self.test_metrics = self.metrics_collection()
        # register metrics to be properly reset at each epoch end and moved to the right device

    def metrics_collection(self) -> MetricCollectionDict:
        return MetricCollectionDict(
            {
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
                "parent": MetricCollection({"-acc": ParentAcc()}),
            }
        )

    def update_metrics_collection(
        self, metrics: MetricCollectionDict, batched_out: BranchDigraphModel.Output, prefix=""
    ):
        metric_values = dict()

        for out in batched_out.unbatch():
            # === AV metrics ===
            fp_gt_p, av_gt_p = split_fp_av_p(out.batch.branch_av_p)
            metric_values["fp"] = metrics["fp"](out.fp_p, fp_gt_p > 0.5)

            tp_mask = fp_gt_p < 0.5
            metric_values["av"] = metrics["av"](out.av_p[tp_mask], av_gt_p[tp_mask] > 0.5)

            # === Direction metrics ===
            dir_p, dir_gt_p = out.dir_p, out.batch.branch_dir
            metric_values["dir"] = metrics["dir"](dir_p, dir_gt_p > 0.5)

            # === Parent classification metrics ===
            metric_values["parent"] = metrics["parent"](out.optimal_parent(), out.optimal_parent_gt())  # type: ignore

        # Flatten metric values dict
        metric_values = {prefix + k1 + k2: v for k1, group in metric_values.items() for k2, v in group.items()}
        return metric_values

    def forward(self, data):
        return self.model(data)

    def losses(self, batch, out: BranchDigraphModel.Output):
        # AV loss
        fp_p_gt, av_p_gt = split_fp_av_p(batch.branch_av_p)
        fp_loss = self.fp_bce_loss(out.fp_logit, fp_p_gt)

        tp_mask = fp_p_gt < 0.5
        av_loss = self.av_bce_loss(out.av_logit[tp_mask], av_p_gt[tp_mask])

        # Dir losses
        dir_loss = self.dir_bce_loss(out.dir_logit[tp_mask], batch.branch_dir[tp_mask])

        # Line loss
        line_loss = self.line_ce_loss(out.lines_logit, out.lines_gt_score(), out.b1())

        f_curi = max(min(1.0, (self.current_epoch - 20) / 50), 0)
        loss = fp_loss + av_loss * lerp(1.0, 0.2, f_curi) + line_loss * lerp(0.0, 1.0, f_curi)  # + dir_loss
        return {"fp_loss": fp_loss, "av_loss": av_loss, "dir_loss": dir_loss, "line_loss": line_loss, "loss": loss}

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        losses = self.losses(batch, outs)
        self.log_dict(
            {"train_" + k: l for k, l in losses.items()},
            batch_size=batch.num_graphs,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        outs = self(batch)

        losses = self.losses(batch, outs)
        self.log_dict(
            {"val_" + k: l for k, l in losses.items()}, batch_size=batch.num_graphs, on_step=False, on_epoch=True
        )
        val_metrics = self.update_metrics_collection(self.val_metrics, outs, prefix="val_")
        self.log_dict(val_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

    def on_validation_end(self) -> None:
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        outs = self(batch)
        test_metrics = self.update_metrics_collection(self.test_metrics, outs, prefix="test_")
        self.log_dict(test_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

    def on_test_end(self) -> None:
        self.test_metrics.reset()

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-3))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.config.get("lr", 1e-2), epochs=self.config.get("epoch", 200), steps_per_epoch=40
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}


class ParentAcc(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self) -> Tensor:
        return self.correct.float() / self.total  # type: ignore


def split_fp_av_p(av_p: Tensor) -> tuple[Tensor, Tensor]:
    """Split pairs of artery and vein probabilities into false positive and artery/vein probabilities."""
    av_sum = av_p.sum(dim=-1)
    fp = 1 - av_sum
    art = av_p[..., 0]
    art[av_sum != 0] /= av_sum[av_sum != 0]
    return fp, art
