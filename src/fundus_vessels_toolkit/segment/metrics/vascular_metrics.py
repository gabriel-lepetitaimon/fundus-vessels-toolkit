from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from torchmetrics import Metric as TorchMetric

from ...segment_to_graph import RetinalVesselSegToGraph
from ...vascular_graph.matching import naive_edit_distance


class ClDice(TorchMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, max_struct_width=5):
        """
        Compute the Center Line Dice score between the skeletonized prediction and target.

        Args:
            max_struct_width: Estimation of the maximum width of the structure element (e.g. largest vessel diameter).
                              Any branch smaller than this will be considered as a spurs (i.e. an artefact of the
                               skeletonization) and will be removed from the skeleton.
        """
        super().__init__()
        self.seg2graph = RetinalVesselSegToGraph()
        self.seg2graph.max_spurs_length = max_struct_width
        self.eps = 1e-6

        self.add_state("correct_prediction", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_prediction", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_target", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0), dist_reduce_fx="sum")

        self.precision = None
        self.recall = None
        self.cl_dice = None

    @property
    def max_struct_width(self):
        return self.seg2graph.max_spurs_length

    def _input_format(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ):
        return _check_input_format(pred, target, skel_pred, skel_target, self.seg2graph)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ):
        pred, target, skel_pred, skel_target = self._input_format(pred, target, skel_pred, skel_target)
        self.correct_prediction += torch.tensor(np.sum((target != 0) & (skel_pred != 0)))
        self.total_prediction += torch.tensor(np.sum(skel_pred != 0))
        self.correct_target += torch.tensor(np.sum((pred != 0) & (skel_target != 0)))
        self.total_target += torch.tensor(np.sum(skel_target != 0))

    def compute(self):
        eps = self.eps
        self.recall = self.correct_target / (self.total_target + eps)
        self.precision = self.correct_prediction / (self.total_prediction + eps)
        self.cl_dice = 2 * self.precision * self.recall / (self.precision + self.recall + eps)
        return self.cl_dice


class MeanClDice(TorchMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, max_struct_width=5):
        """
        Compute the Center Line Dice score between the skeletonized prediction and target.

        Args:
            max_struct_width: Estimation of the maximum width of the structure element (e.g. largest vessel diameter).
                              Any branch smaller than this will be considered as a spurs (i.e. an artefact of the
                               skeletonization) and will be removed from the skeleton.
        """
        super().__init__()
        self.seg2graph = RetinalVesselSegToGraph()
        self.seg2graph.max_spurs_length = max_struct_width
        self.eps = 1e-6

        self.add_state("sum_precision", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_f1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        self.cl_dice = None

    @property
    def max_struct_width(self):
        return self.seg2graph.max_spurs_length

    def _input_format(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ):
        return _check_input_format(pred, target, skel_pred, skel_target, self.seg2graph)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ):
        pred, target, skel_pred, skel_target = self._input_format(pred, target, skel_pred, skel_target)
        eps = self.eps

        recall = np.sum((skel_pred != 0) & (target != 0), axis=(1, 2)) / (np.sum(skel_pred != 0, axis=(1, 2)) + eps)
        precision = np.sum((skel_target != 0) & (pred != 0), axis=(1, 2)) / (
            np.sum(skel_target != 0, axis=(1, 2)) + eps
        )
        self.sum_recall += torch.tensor(recall.sum())
        self.sum_precision += torch.tensor(precision.sum())
        self.sum_f1 += torch.tensor((2 * precision * recall / (precision + recall + eps)).sum())
        self.n_samples += torch.tensor(len(pred))

    def compute(self):
        self.recall = self.sum_recall / self.n_samples
        self.precision = self.sum_precision / self.n_samples
        self.cl_dice = self.sum_f1 / self.n_samples
        return self.cl_dice


class F1Topo(TorchMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, max_struct_width: int = 20):
        """
        Compute the Center Line Dice score between the skeletonized prediction and target.

        Args:
            max_struct_width: Estimation of the maximum width of the structure element (e.g. largest vessel diameter).
                              Any branch smaller than this will be considered as a spurs (i.e. an artefact of the
                               skeletonization) and will be removed from the skeleton.
        """
        super().__init__()
        self.seg2graph = RetinalVesselSegToGraph(max_struct_width)
        self.eps = 1e-6

        self.add_state("correct_prediction", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_prediction", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_target", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0), dist_reduce_fx="sum")

        self.precision = None
        self.recall = None
        self.f1_topo = None

    @property
    def max_struct_width(self):
        return self.seg2graph.max_vessel_diameter

    def _input_format(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _check_input_format(pred, target, skel_pred, skel_target, self.seg2graph)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ):
        pred, target, skel_pred, skel_target = self._input_format(pred, target, skel_pred, skel_target)

        v_preds = [self.seg2graph.skel2vgraph(_) for _ in skel_pred]
        v_targets = [self.seg2graph.skel2vgraph(_) for _ in skel_target]

        pred_diffs, target_diffs = zip(
            *[
                naive_edit_distance(
                    graph1=v_pred,
                    graph2=v_target,
                    max_matching_distance=self.max_struct_width * 7,
                    min_distance=self.max_struct_width / 2,
                    density_matching_sigma=self.max_struct_width,
                )
                for v_pred, v_target in zip(v_preds, v_targets, strict=True)
            ],
            strict=True,
        )
        pred_n = [p.branch_count for p in v_preds]
        target_n = [t.branch_count for t in v_targets]

        self.correct_prediction += torch.tensor(sum(n - d for n, d in zip(pred_n, pred_diffs, strict=True)))
        self.total_prediction += torch.tensor(sum(pred_n))

        self.correct_target += torch.tensor(sum(n - d for n, d in zip(target_n, target_diffs, strict=True)))
        self.total_target += torch.tensor(sum(target_n))

    def compute(self):
        eps = self.eps
        self.recall = self.correct_target / (self.total_target + eps)
        self.precision = self.correct_prediction / (self.total_prediction + eps)
        self.f1_topo = 2 * self.precision * self.recall / (self.precision + self.recall + eps)
        return self.f1_topo


class MeanF1Topo(TorchMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, max_struct_width: int = 20):
        """
        Compute the Center Line Dice score between the skeletonized prediction and target.

        Args:
            max_struct_width: Estimation of the maximum width of the structure element (e.g. largest vessel diameter).
                              Any branch smaller than this will be considered as a spurs (i.e. an artefact of the
                               skeletonization) and will be removed from the skeleton.
        """
        super().__init__()
        self.seg2graph = RetinalVesselSegToGraph(max_struct_width)
        self.eps = 1e-6

        self.add_state("sum_precision", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_f1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

        self.precision = None
        self.recall = None
        self.f1_topo = None

    @property
    def max_struct_width(self):
        return self.seg2graph.max_vessel_diameter

    def _input_format(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _check_input_format(pred, target, skel_pred, skel_target, self.seg2graph)

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        skel_pred: torch.Tensor = None,
        skel_target: torch.Tensor = None,
    ):
        pred, target, skel_pred, skel_target = self._input_format(pred, target, skel_pred, skel_target)
        eps = self.eps

        v_preds = [self.seg2graph.skel2vgraph(_) for _ in skel_pred]
        v_targets = [self.seg2graph.skel2vgraph(_) for _ in skel_target]
        pred_diffs, target_diffs = zip(
            *[naive_edit_distance(p, t, self.max_struct_width) for p, t in zip(v_preds, v_targets, strict=True)],
            strict=True,
        )
        pred_n = [p.branch_count for p in v_preds]
        target_n = [t.branch_count for t in v_targets]

        recall = sum((n - d) / (n + eps) for n, d in zip(pred_n, pred_diffs, strict=True))
        precision = sum((n - d) / (n + eps) for n, d in zip(target_n, target_diffs, strict=True))
        self.sum_recall += torch.tensor(recall.sum())
        self.sum_precision += torch.tensor(precision.sum())
        self.sum_f1 += torch.tensor((2 * precision * recall / (precision + recall + eps)).sum())
        self.n_samples += torch.tensor(len(pred))

    def compute(self):
        self.recall = self.sum_recall / self.n_samples
        self.precision = self.sum_precision / self.n_samples
        self.f1_topo = self.sum_f1 / self.n_samples
        return self.f1_topo


def _check_input_format(
    pred: torch.Tensor,
    target: torch.Tensor,
    skel_pred: torch.Tensor = None,
    skel_target: torch.Tensor = None,
    seg2graph: RetinalVesselSegToGraph = None,
):
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    assert pred.ndim == 3, f"preds must be of size (B, H, W), got preds.ndim={pred.ndim}."
    if target.ndim == 4:
        target = target.squeeze(1)
    assert target.ndim == 3, f"preds must be of size (B, H, W), got preds.ndim={pred.ndim}."
    assert (
        pred.shape == target.shape
    ), f"preds and target must have the same shape, got {pred.shape} and {target.shape}."

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    assert pred.dtype == bool, f"preds must be of type bool, got preds.dtype={pred.dtype}."
    assert target.dtype == bool, f"target must be of type bool, got target.dtype={target.dtype}."

    if skel_pred is not None:
        assert skel_pred.ndim == 3, f"skel_preds must be of size (B, H, W), got skel_preds.shape={skel_pred.shape}."
        assert (
            skel_pred.shape == pred.shape
        ), f"skel_preds and preds must have the same shape, got {skel_pred.shape} and {pred.shape}."
        if isinstance(skel_pred, torch.Tensor):
            skel_pred = skel_pred.detach().cpu().numpy()
    else:
        if seg2graph is None:
            seg2graph = RetinalVesselSegToGraph()
        skel_pred = np.stack([seg2graph.skeletonize(_) for _ in pred], axis=0)

    if skel_target is not None:
        assert (
            skel_target.ndim == 3
        ), f"skel_target must be of size (B, H, W), got skel_target.shape={skel_target.shape}."
        assert (
            skel_target.shape == target.shape
        ), f"skel_target and target must have the same shape, got {skel_target.shape} and {target.shape}."
        if isinstance(skel_target, torch.Tensor):
            skel_target = skel_target.detach().cpu().numpy()
    else:
        if seg2graph is None:
            seg2graph = RetinalVesselSegToGraph()
        skel_target = np.stack([seg2graph.skeletonize(_) for _ in target], axis=0)

    return pred, target, skel_pred, skel_target


try:
    from nntemplate import Cfg
    from nntemplate.task.metrics import MetricCfg, register_metric
except ImportError:
    pass
else:

    @register_metric("clDice")
    class CfgClDice(MetricCfg):
        max_struct_width = Cfg.int(
            5,
            min=0,
            help="Estimation of the maximum width of the structure element "
            "(e.g. largest vessel diameter). Any branch smaller than this will be"
            "considered as a spurs (i.e. an artefact of the skeletonization) and will"
            "be removed from the skeleton.",
        )
        mean = Cfg.bool(
            False,
            help="If True, the Dice score is computed as the mean of the Dice score of each image in the batch.",
        )

        def prepare_data(self, pred, target, mask=None):
            if mask is not None:
                pred = pred * mask
                target = target * mask
            return pred, target

        def create(self):
            if self.mean:
                return MeanClDice(max_struct_width=self.max_struct_width)
            else:
                return ClDice(max_struct_width=self.max_struct_width)

        def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
            module.log(
                f"{name}.clPrecision",
                metric.precision,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(
                f"{name}.clRecall",
                metric.recall,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(f"{name}.clDice", metric, add_dataloader_idx=False, enable_graph=False)

    @register_metric("f1Topo")
    class CfgF1Topo(MetricCfg):
        max_struct_width = Cfg.int(
            5,
            min=0,
            help="Estimation of the maximum width of the structure element "
            "(e.g. largest vessel diameter). Any branch smaller than this will be"
            "considered as a spurs (i.e. an artefact of the skeletonization) and will"
            "be removed from the skeleton.",
        )
        mean = Cfg.bool(
            False,
            help="If True, the F1-Topo score is computed as the mean of the F1-Topo score of each image in the batch.",
        )

        def prepare_data(self, pred, target, mask=None):
            if mask is not None:
                pred = pred * mask
                target = target * mask
            return pred, target

        def create(self):
            if self.mean:
                return MeanF1Topo(max_struct_width=self.max_struct_width)
            else:
                return F1Topo(max_struct_width=self.max_struct_width)

        def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
            module.log(
                f"{name}.topoPrecision",
                metric.precision,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(
                f"{name}.topoRecall",
                metric.recall,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(f"{name}.topoF1", metric, add_dataloader_idx=False, enable_graph=False)

    @register_metric("sumClDiceF1Topo")
    class CfgSumClDiceF1Topo(MetricCfg):
        max_struct_width = Cfg.int(
            12,
            min=0,
            help="Estimation of the maximum width of the structure element "
            "(e.g. largest vessel diameter). Any branch smaller than this will be"
            "considered as a spurs (i.e. an artefact of the skeletonization) and will"
            "be removed from the skeleton.",
        )
        mean = Cfg.bool(
            False,
            help="If True, the scores are computed as the mean of the scores of each image in the batch.",
        )
        clDiceCoef = Cfg.float(0.5, min=0, max=1, help="Weight of the ClDice score in the sum.")

        def __init__(self, data=None, parent=None):
            super(CfgSumClDiceF1Topo, self).__init__(data=None, parent=parent)
            self.seg2graph = RetinalVesselSegToGraph(self.max_struct_width)

        def prepare_data(self, pred, target, mask=None):
            if pred.ndim == 4:
                pred = pred[:, 1] > 0.5
            if mask is not None:
                pred = pred * mask
                target = target * mask
            return _check_input_format(pred, target, seg2graph=self.seg2graph)

        def create(self):
            a = self.clDiceCoef
            b = 1 - a

            if self.mean:
                metric = MeanClDice() * a + MeanF1Topo(max_struct_width=self.max_struct_width) * b
                metric.cl_dice = metric.metric_a.metric_a
                metric.f1_topo = metric.metric_b.metric_a
                return metric
            else:
                metric = ClDice() * a + F1Topo(max_struct_width=self.max_struct_width) * b
                metric.cl_dice = metric.metric_a.metric_a
                metric.f1_topo = metric.metric_b.metric_a
                return metric

        def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
            clDice = metric.metric_a.metric_a
            f1Topo = metric.metric_b.metric_a
            module.log(
                f"{name}.clPrecision",
                clDice.precision,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(
                f"{name}.clRecall",
                clDice.recall,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(f"{name}.clDice", clDice, add_dataloader_idx=False, enable_graph=False)
            module.log(
                f"{name}.topoPrecision",
                f1Topo.precision,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(
                f"{name}.topoRecall",
                f1Topo.recall,
                add_dataloader_idx=False,
                enable_graph=False,
            )
            module.log(f"{name}.topoF1", f1Topo, add_dataloader_idx=False, enable_graph=False)

            module.log(f"{name}.sumClDiceF1Topo", metric, add_dataloader_idx=False, enable_graph=False)
