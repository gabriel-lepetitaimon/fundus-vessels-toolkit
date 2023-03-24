from nntemplate.task.metrics import MetricCfg, register_metric
from nntemplate.misc.clip_pad import select_pixels_by_mask
import numpy as np
import torch
from torchmetrics import Metric as TorchMetric
import torchmetrics as tm
import pytorch_lightning as pl

from .seg2graph import skeletonize


class F1Skeleton(TorchMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("correct_precision", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_precision", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_recall", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_recall", default=torch.tensor(0), dist_reduce_fx="sum")

        self.precision = None
        self.recall = None

    def _input_format(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 4:
            preds = preds.squeeze(1)
        assert preds.ndim == 3, f"preds must be of size (B, H, W), got preds.ndim={preds.ndim}."
        if target.ndim == 4:
            target = target.squeeze(1)
        assert target.ndim == 3, f"preds must be of size (B, H, W), got preds.ndim={preds.ndim}."
        assert preds.shape == target.shape, f"preds and target must have the same shape, got {preds.shape} and {target.shape}."
        return preds, target

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        preds = preds.detach().cpu().numpy()
        preds_skel = skeletonize(preds)
        target = target.detach().cpu().numpy()
        target_skel = skeletonize(target)

        self.correct_precision += torch.Tensor(np.sum(preds_skel == target))
        self.total_precision += torch.Tensor(np.sum(preds_skel != 0))

        self.correct_recall += torch.Tensor(np.sum(target_skel == preds))
        self.total_recall += torch.Tensor(np.sum(target_skel != 0))

    def compute(self):
        self.recall = self.correct_recall / self.total_recall
        self.precision = self.correct_precision / self.total_precision
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@register_metric('f1-skeleton')
class CfgF1Skeleton(MetricCfg):

    def prepare_data(self, pred, target, mask=None):
        if mask is not None:
            pred, target = select_pixels_by_mask(pred, target, mask=mask)
            return pred, target
        else:
            return pred, target

    def create(self):
        return F1Skeleton()

    def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
        module.log(f'{name}.precision', metric.precision, add_dataloader_idx=False, enable_graph=False)
        module.log(f'{name}.recall', metric.recall, add_dataloader_idx=False, enable_graph=False)
        module.log(f'{name}.f1', metric, add_dataloader_idx=False, enable_graph=False)
