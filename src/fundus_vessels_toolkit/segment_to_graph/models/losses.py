from typing import Literal, Optional

import torch
from pytorch_metric_learning import distances as pml_distances
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners as pml_miners
from torch import Tensor

from fundus_vessels_toolkit.segment_to_graph.models.model import BranchDigraphModel
from fundus_vessels_toolkit.utils.torch import with_weight


class CrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        invalid_metagroup_penalty: float = 0.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.invalid_metagroup_penalty = invalid_metagroup_penalty

    def forward(
        self,
        x: Tensor,
        target: Tensor,
        group_idx: Tensor,
        mask: Optional[Tensor] = None,
        metagroup_idx: Optional[Tensor] = None,
        other_group_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the cross-entropy loss which enforce the selection of one sample per group, given sample-wise logits and group indices.

        Parameters
        ----------
        x : Tensor
            A tensor of shape (N,) containing the predicted logits for each sample.

        target : Tensor
            A tensor of shape (N,) containing whether each sample is the elected one in its group. The sum of target values for each group should be 1.

        group_idx : Tensor
            A tensor of shape (N,) containing the group affiliation for each sample. The values in `group_idx` should be non-negative integers, and samples with the same group index belong to the same group.

        mask : Optional[Tensor], optional
            A boolean tensor of shape (N,) indicating which group should be included in the loss computation. If None, all samples are included.

        metagroup_idx : Optional[Tensor], optional
            A tensor of shape (G,) containing the metagroup affiliation for each group.

        other_group_idx : Optional[Tensor], optional
            A tensor of shape (N,) containing the group affiliation for each sample according to another grouping scheme, used to apply a penalty to samples belonging to metagroup that are not consistent between the two grouping schemes.

        Returns
        -------
        Tensor
            A scalar tensor containing the mean cross-entropy loss over all groups.
        """  # noqa: E501
        group_idx = group_idx.long()
        num_group = len(mask) if mask is not None else int(group_idx.max().item()) + 1
        group_size = torch.bincount(group_idx, minlength=num_group)
        assert torch.all(group_size > 0), "All groups must have at least one sample."

        # Normalize logits by group max for numerical stability
        group_max = torch.zeros(num_group, device=x.device).scatter_reduce_(0, group_idx, x, "amax", include_self=False)
        x_normed = x - group_max[group_idx]

        # Add penalty for invalid metagroups if provided
        if self.invalid_metagroup_penalty > 0 and metagroup_idx is not None and other_group_idx is not None:
            penalized_samples = (other_group_idx >= 0) & metagroup_idx[group_idx] != other_group_idx[group_idx]
            x_normed[penalized_samples] += self.invalid_metagroup_penalty

        # Compute the log-sum-exp
        group_sum_exp = torch.zeros(num_group, device=x.device).scatter_add_(0, group_idx, x_normed.exp())
        group_log_sum_exp = (group_sum_exp + 1e-16).log()

        # Compute the log-softmax
        log_softmax = x_normed - group_log_sum_exp[group_idx]

        # Compute the loss
        if self.label_smoothing > 0:
            log_softmax = log_softmax * (1 - self.label_smoothing) + self.label_smoothing / group_size[group_idx]
        loss = -log_softmax * target

        # Sum over samples in the same group
        group_loss = torch.zeros(num_group, device=x.device).scatter_add_(0, group_idx, loss)

        # Apply mask if provided
        if mask is not None:
            group_loss = group_loss[mask]

        # Reduce the loss according to the specified reduction method
        if self.reduction == "mean":
            group_loss = group_loss.sum() / num_group
        elif self.reduction == "sum":
            group_loss = group_loss.sum()
        return group_loss


class VBranchDigraphMiner(pml_miners.BaseMiner):
    def __init__(self, sample_ratio: float = 1.0, triplet: bool = False, same_tail_node: bool = True):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.triplet = triplet
        self.same_tail_node = same_tail_node

    def mine(
        self, embedding, subtree_idx, tail_node_idx
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        device = embedding.device

        # → Shuffle samples to ensure random sampling of pairs
        # sample_idx = torch.arange(embedding.size(0), device=device)
        sample_idx = torch.randperm(embedding.size(0), device=device)
        subtree_idx = subtree_idx[sample_idx]
        tail_node_idx = tail_node_idx[sample_idx]

        # → List all positive and negative pairs
        pos_pairs = subtree_idx[:, None] == subtree_idx[None, :]
        neg_pairs = ~pos_pairs
        if self.same_tail_node:
            same_tail_mask = tail_node_idx[:, None] == tail_node_idx[None, :]
            pos_pairs &= same_tail_mask
            neg_pairs &= same_tail_mask

        if self.triplet:
            pos_pairs.triu_(diagonal=1)
            neg_pairs.fill_diagonal_(False)
            anchors = pos_pairs.any(dim=1) & neg_pairs.any(dim=1)
            a_idx = anchors.argwhere().squeeze()

            # Sample one positive and one negative pair for each anchor
            def rng_one_sample_per_row(mask):
                pairs_ = torch.zeros_like(mask, dtype=torch.int)
                pairs_[mask] = torch.randperm(mask.sum(), dtype=torch.int, device=device) + 1
                return pairs_.argmax(dim=1)

            p_idx = rng_one_sample_per_row(pos_pairs[a_idx])
            n_idx = rng_one_sample_per_row(neg_pairs[a_idx])
            return sample_idx[a_idx], sample_idx[p_idx], sample_idx[n_idx]
        else:
            pos_pairs = sample_idx[pos_pairs.triu(diagonal=1).argwhere()].T
            neg_pairs = sample_idx[neg_pairs.triu(diagonal=1).argwhere()].T
            return pos_pairs[0], pos_pairs[1], neg_pairs[0], neg_pairs[1]

    def forward(
        self, out: BranchDigraphModel.Output
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        self.reset_stats()
        with torch.no_grad():
            mining_output = self.mine(out.b1_embedding, out.gt_subtree_idx, out.gt_tail_nodes(use_parent_head=True))
        self.output_assertion(mining_output)
        return mining_output


class BranchContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dist = pml_distances.CosineSimilarity()
        self.contrastive_loss = with_weight(pml_losses.CircleLoss(distance=dist), 0.1)
        self.triplet_loss = with_weight(pml_losses.TripletMarginLoss(distance=dist), 1)
        self.pairs_miner = VBranchDigraphMiner(triplet=False, same_tail_node=True)
        self.triplet_miner = VBranchDigraphMiner(triplet=True, same_tail_node=True)

    def __call__(self, out: BranchDigraphModel.Output) -> dict[str, Tensor]:
        return super().__call__(out)

    def forward(self, out: BranchDigraphModel.Output) -> dict[str, Tensor]:
        pairs = self.pairs_miner(out)
        triplets = self.triplet_miner(out)
        contrastive_loss = self.contrastive_loss(out.b1_embedding, indices_tuple=pairs)
        triplet_loss = self.triplet_loss(out.b1_embedding, indices_tuple=triplets)
        return {"contr_loss": contrastive_loss, "triplet_loss": triplet_loss}
