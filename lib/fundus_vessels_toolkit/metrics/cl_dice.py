import numpy as np
import torch
from torchmetrics import Metric as TorchMetric
import torchmetrics as tm
import pytorch_lightning as pl

from ..seg2graph import skeletonize


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
        self.max_struct_width = max_struct_width

        self.add_state("correct_prediction", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_prediction", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_target", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0), dist_reduce_fx="sum")

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
        preds_skel = skeletonize(preds, max_spurs_length=self.max_struct_width)
        target = target.detach().cpu().numpy()
        target_skel = skeletonize(target, max_spurs_length=self.max_struct_width)

        self.correct_prediction += torch.Tensor(np.sum(preds_skel == target))
        self.total_prediction += torch.Tensor(np.sum(preds_skel != 0))

        self.correct_target += torch.Tensor(np.sum(target_skel == preds))
        self.total_target += torch.Tensor(np.sum(target_skel != 0))

    def compute(self):
        self.recall = self.correct_target / self.total_target
        self.precision = self.correct_prediction / self.total_prediction
        return 2 * self.precision * self.recall / (self.precision + self.recall)



try:
    from nntemplate import Cfg
    from nntemplate.task.metrics import MetricCfg, register_metric
    from nntemplate.torch_utils.clip_pad import select_pixels_by_mask
except ImportError:
    pass
else:
    @register_metric('clDice')
    class CfgClDice(MetricCfg):
        max_struct_width = Cfg.int(5, min=0, help='Estimation of the maximum width of the structure element '
                                                  '(e.g. largest vessel diameter). Any branch smaller than this will be'
                                                  'considered as a spurs (i.e. an artefact of the skeletonization) and will'
                                                  'be removed from the skeleton.')

        def prepare_data(self, pred, target, mask=None):
            if mask is not None:
                pred, target = select_pixels_by_mask(pred, target, mask=mask)
                return pred, target
            else:
                return pred, target

        def create(self):
            return ClDice(max_struct_width=self.max_struct_width)

        def log(self, module: pl.LightningModule, name: str, metric: tm.Metric):
            module.log(f'{name}.clPrecision', metric.precision, add_dataloader_idx=False, enable_graph=False)
            module.log(f'{name}.clRecall', metric.recall, add_dataloader_idx=False, enable_graph=False)
            module.log(f'{name}.clDice', metric, add_dataloader_idx=False, enable_graph=False)
