from __future__ import annotations

from typing import Iterable

import pytorch_lightning as L
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from torch import Tensor
from torchmetrics import Metric, MetricCollection, Specificity
from torchmetrics.classification import Accuracy, Precision, Recall

import wandb

from ...utils.torch import GroupByCrossEntropyLoss
from .dataset import VBranchDigraphBatch
from .digraph_model import BranchDigraphModel, BranchFeaturesEfficientNetV2S, Gatv2GCN


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
        self.model = BranchDigraphModel(BranchFeaturesEfficientNetV2S(), Gatv2GCN(n_in=784, n_out=512, edge_attr_dim=7))

        # === LOSSES ===
        self.fp_bce_loss = nn.BCEWithLogitsLoss()
        self.av_bce_loss = nn.BCEWithLogitsLoss()
        self.dir_bce_loss = nn.BCEWithLogitsLoss()
        self.line_ce_loss = GroupByCrossEntropyLoss()

        # === METRICS ===
        self.val_metrics = self.metrics_collection(opti_tree=False)
        self.val_preds = {}
        self.test_metrics = self.metrics_collection()
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
            "opti_dir": MetricCollection({"-acc": Accuracy("binary")}),
            "tree": MetricCollection(
                {
                    "-root-acc": RootAcc(),
                    "-parent-acc": ParentAcc(ignore_root=False),
                    "-parent-same-subtree-acc": ParentSameSubtreeAcc(ignore_root=False),
                    "-parent-same-subtree-mean-dist": ParentSameSubtreeMeanDist(),
                    "-parent-1tol-acc": ParentCloseAcc(ignore_root=False),
                }
            ),
        }
        if opti_tree:
            collection["opti_tree"] = MetricCollection(
                {
                    "-root-acc": RootAcc(),
                    "-parent-acc": ParentAcc(ignore_root=False),
                    "-parent-same-subtree-acc": ParentSameSubtreeAcc(ignore_root=False),
                    "-parent-same-subtree-mean-dist": ParentSameSubtreeMeanDist(),
                    "-parent-1tol-acc": ParentCloseAcc(ignore_root=False),
                }
            )
        return MetricCollectionDict(collection)

    def update_metrics_collection(
        self, metrics: MetricCollectionDict, batched_out: BranchDigraphModel.Output, prefix=""
    ):
        metric_values = dict()
        opti_tree = "opti_tree" in metrics

        if opti_tree:

            def optimize_tree(out: BranchDigraphModel.Output):
                opti_parent, opti_dir = out.optimal_tree
                return out, opti_parent, opti_dir

            parallel = Parallel(n_jobs=batched_out.batch_size)
            outs = [parallel(delayed(optimize_tree)(out) for out in batched_out.unbatch())]
        else:
            outs = batched_out.unbatch()
        for out in outs:
            if opti_tree:
                assert isinstance(out, tuple)
                out, opti_parent, opti_dir = out
            assert isinstance(out, BranchDigraphModel.Output)
            # === AV metrics ===
            metric_values["fp"] = metrics["fp"](out.fp_p, out.gt_fp_p > 0.5)

            tp_mask = out.gt_fp_p < 0.5
            metric_values["av"] = metrics["av"](out.av_p[tp_mask], out.gt_av_p[tp_mask] > 0.5)

            # === Direction metrics ===
            dir_p, dir_gt_p = out.dir_p, out.batch.branch_dir_p
            metric_values["dir"] = metrics["dir"](dir_p[tp_mask], dir_gt_p[tp_mask] > 0.5)

            # === Parent classification metrics ===
            metric_values["tree"] = metrics["tree"](out.max_parent(use_gt=True), out.gt_parent(), tp_mask)

            if opti_tree:
                metric_values["opti_dir"] = metrics["opti_dir"](opti_dir[tp_mask], dir_gt_p[tp_mask] > 0.5)
                metric_values["opti_tree"] = metrics["opti_tree"](opti_parent, out.gt_parent(), tp_mask)

        # Flatten metric values dict
        metric_values = {prefix + k1 + k2: v for k1, group in metric_values.items() for k2, v in group.items()}
        return metric_values

    def update_preds(self, preds_dict: dict, batched_out: BranchDigraphModel.Output):
        for out in batched_out.unbatch():
            if "table" not in preds_dict:
                preds_dict["table"] = wandb.Table(columns=["name", "parent", "dir", "opti_parent", "opti_dir", "av"])
            preds_dict["table"].add_data(
                out.name,
                out.max_parent(use_gt=False).cpu().tolist(),
                (out.dir_logit > 0).cpu().int().tolist(),
                out.optimal_tree[0].cpu().tolist(),
                out.optimal_tree[1].cpu().int().tolist(),
                out.fp_av_class.cpu().tolist(),
            )

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
        mask = out.lines_mask(filter_dir="gt")
        line_loss = self.line_ce_loss(out.lines_score[mask], out.gt_lines_score[mask], out.lines[mask].b1, tp_mask)

        f_curi_dir = max(min(1.0, (self.current_epoch - 20) / 10), 0)
        f_curi_line = max(min(1.0, (self.current_epoch - 40) / 10), 0)
        loss = fp_loss + av_loss + line_loss * f_curi_line + dir_loss * f_curi_dir
        return {"fp_loss": fp_loss, "av_loss": av_loss, "dir_loss": dir_loss, "line_loss": line_loss, "loss": loss}

    def training_step(self, batch, batch_idx):
        model_out = self(batch)
        losses = self.losses(model_out)
        self.log_dict({"train_" + k: l for k, l in losses.items()}, batch_size=batch.num_graphs, prog_bar=True)
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


class TreeMetric(Metric):
    total: Tensor
    root_tp: Tensor
    root_fp: Tensor
    root_fn: Tensor
    true: Tensor
    wrong_subtree: Tensor
    false_same_subtree: Tensor
    dist_1_or_less: Tensor
    same_subtree_mean_dist: Tensor

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("root_tp", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("root_fp", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("root_fn", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("true", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("wrong_subtree", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("false_same_subtree", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("dist_1_or_less", default=torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("same_subtree_mean_dist", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="mean")

    def update(self, pred: Tensor, target: Tensor, mask: Tensor) -> None:
        from ...utils.cpp_extensions.fvt_cpp import tree_distance

        if pred.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.total += len(pred)

        assert target.max() < len(target), "target contains invalid parent indices"
        tree_dist_gt = tree_distance(target.detach().cpu()).to(device=target.device)[0]
        pred, target = pred[mask], target[mask]

        root_preds = pred == -1
        root_target = target == -1

        self.root_tp += torch.sum(root_preds & root_target)
        self.root_fp += torch.sum(root_preds & ~root_target)
        self.root_fn += torch.sum(~root_preds & root_target)

        pred, target = pred[~root_target], target[~root_target]  # Exclude root nodes from parent metrics
        true_mask = pred == target
        true_parent_n = torch.sum(true_mask)
        self.true += true_parent_n

        pred, target = pred[~true_mask], target[~true_mask]
        same_subtree_mask = ~tree_dist_gt[pred, target].isnan()
        same_subtree_n = same_subtree_mask.sum()
        self.wrong_subtree += len(pred) - same_subtree_n
        self.false_same_subtree += same_subtree_n

        pred, target = pred[same_subtree_mask], target[same_subtree_mask]
        dist = tree_dist_gt[pred, target]
        self.same_subtree_mean_dist += dist.sum() / (same_subtree_n + true_parent_n + 1e-8)
        self.dist_1_or_less += torch.sum(dist <= 1) + true_parent_n

    @property
    def root_tn(self):
        return self.true + self.wrong_subtree + self.false_same_subtree

    def compute(self) -> Tensor:
        return (self.root_tp + self.true) / self.total


class RootAcc(TreeMetric):
    def compute(self) -> Tensor:
        return self.root_tp / (self.root_tp + self.root_fp + self.root_fn)


class RootSpecificity(TreeMetric):
    def compute(self) -> Tensor:
        return self.root_tp / (self.root_tp + self.root_fp)


class RootSensitivity(TreeMetric):
    def compute(self) -> Tensor:
        return self.root_tp / (self.root_tp + self.root_fn)


class ParentAcc(TreeMetric):
    def __init__(self, ignore_root=True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_root = ignore_root

    def compute(self) -> Tensor:
        return self.true / self.root_tn if self.ignore_root else (self.root_tp + self.true) / self.total


class ParentSameSubtreeAcc(TreeMetric):
    def __init__(self, ignore_root=True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_root = ignore_root

    def compute(self) -> Tensor:
        if self.ignore_root:
            return (self.false_same_subtree + self.true) / self.root_tn
        return (self.root_tp + self.true + self.false_same_subtree) / self.total


class ParentSameSubtreeMeanDist(TreeMetric):
    def compute(self) -> Tensor:
        return self.same_subtree_mean_dist / self.root_tn


class ParentCloseAcc(TreeMetric):
    def __init__(self, ignore_root=True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_root = ignore_root

    def compute(self) -> Tensor:
        if self.ignore_root:
            return self.dist_1_or_less / self.root_tn
        return (self.root_tp + self.dist_1_or_less) / self.total
