from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric


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


class BaseTreeMetric(Metric):
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


class RootAcc(BaseTreeMetric):
    def compute(self) -> Tensor:
        return self.root_tp / (self.root_tp + self.root_fp + self.root_fn)


class RootSpecificity(BaseTreeMetric):
    def compute(self) -> Tensor:
        return self.root_tp / (self.root_tp + self.root_fp)


class RootSensitivity(BaseTreeMetric):
    def compute(self) -> Tensor:
        return self.root_tp / (self.root_tp + self.root_fn)


class ParentAcc(BaseTreeMetric):
    def __init__(self, ignore_root=True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_root = ignore_root

    def compute(self) -> Tensor:
        return self.true / self.root_tn if self.ignore_root else (self.root_tp + self.true) / self.total


class ParentSameSubtreeAcc(BaseTreeMetric):
    def __init__(self, ignore_root=True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_root = ignore_root

    def compute(self) -> Tensor:
        if self.ignore_root:
            return (self.false_same_subtree + self.true) / self.root_tn
        return (self.root_tp + self.true + self.false_same_subtree) / self.total


class ParentSameSubtreeMeanDist(BaseTreeMetric):
    def compute(self) -> Tensor:
        return self.same_subtree_mean_dist / (self.true + self.false_same_subtree + 1e-8)


class ParentCloseAcc(BaseTreeMetric):
    def __init__(self, ignore_root=True, **kwargs):
        super().__init__(**kwargs)
        self.ignore_root = ignore_root

    def compute(self) -> Tensor:
        if self.ignore_root:
            return self.dist_1_or_less / self.root_tn
        return (self.root_tp + self.dist_1_or_less) / self.total
