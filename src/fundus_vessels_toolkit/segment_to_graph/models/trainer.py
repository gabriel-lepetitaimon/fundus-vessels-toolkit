from typing import Iterable

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric, MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

from .digraph_model import (
    BranchDigraphModel,
    BranchFeaturesEfficientNetV2S,
    Gatv2GCN,
    Lines,
    gt_parent_from_batch,
    pred_max_parent,
)


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
        self.av_nll_loss = nn.NLLLoss()
        self.dir_bce_loss = nn.BCEWithLogitsLoss()
        self.line_ce_loss = nn.CrossEntropyLoss()

        # === METRICS ===
        self.val_metrics = self.metrics_collection()
        self.test_metrics = self.metrics_collection()
        # register metrics to be properly reset at each epoch end and moved to the right device

    def metrics_collection(self) -> MetricCollectionDict:
        return MetricCollectionDict(
            {
                "missing": MetricCollection(
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

    def update_metrics_collection(self, metrics: MetricCollectionDict, batch, out, prefix=""):
        parent_gt = gt_parent_from_batch(batch)

        edge_p, valid_lines_mask_pred, valid_roots_pred = out[2:]
        valid_lines_pred = Lines(batch.edge_index, batch.edge_first_tip).select(valid_lines_mask_pred)
        parent_pred = pred_max_parent(edge_p, valid_lines_pred, valid_roots_pred)

        metric_values = dict()

        for idx in range(batch.batch_size):
            branch_mask = batch.batch == idx
            av_p, dir_p = out[:2]

            # === AV metrics ===
            av_p, branch_av_gt = av_p[branch_mask], batch.branch_av_p[branch_mask]
            av_pred = av_p.argmax(dim=-1)
            missing_gt = branch_av_gt.sum(dim=-1) < 0.5
            metric_values["missing"] = metrics["missing"](av_pred == 0, missing_gt)
            metric_values["av"] = metrics["av"](
                av_p[~missing_gt, 1:].argmax(dim=-1), branch_av_gt[~missing_gt].argmax(dim=-1)
            )

            # === Direction metrics ===
            dir_p = dir_p[branch_mask]
            metric_values["dir"] = metrics["dir"](dir_p > 0, batch.branch_dir[branch_mask] > 0.5)

            # === Parent classification metrics ===
            metric_values["parent"] = metrics["parent"](parent_gt[branch_mask], parent_pred[branch_mask])

        # Flatten metric values dict
        metric_values = {prefix + k1 + k2: v for k1, group in metric_values.items() for k2, v in group.items()}
        return metric_values

    def forward(self, data):
        return self.model(data)

    def losses(self, batch, outs):
        av_p, dir_p, edge_p, valid_lines, valid_roots = outs
        # AV loss
        target_av_p = torch.cat([1 - batch.branch_av_p.sum(dim=-1, keepdim=True), batch.branch_av_p], dim=-1)
        target_av = torch.argmax(target_av_p, dim=-1)
        av_loss = self.av_nll_loss(F.log_softmax(av_p, dim=1), target_av)

        # Dir and line losses
        dir_loss = self.dir_bce_loss(dir_p, batch.branch_dir)
        edge_p_gt = torch.cat([batch.edge_p[valid_lines], batch.branch_root_p[valid_roots]], dim=0)
        line_loss = self.line_ce_loss(edge_p, edge_p_gt)

        loss = av_loss + dir_loss
        return {"av_loss": av_loss, "dir_loss": dir_loss, "line_loss": line_loss, "loss": loss}

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
        val_metrics = self.update_metrics_collection(self.val_metrics, batch, outs, prefix="val_")
        self.log_dict(val_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

    def on_validation_end(self) -> None:
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        outs = self(batch)
        test_metrics = self.update_metrics_collection(self.test_metrics, batch, outs, prefix="test_")
        self.log_dict(test_metrics, batch_size=batch.num_graphs, on_step=False, on_epoch=True)

    def on_test_end(self) -> None:
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-4))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
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
        return self.correct.float() / self.total
