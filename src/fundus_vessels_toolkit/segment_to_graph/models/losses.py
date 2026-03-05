import math
from typing import Literal, Optional

import torch

from fundus_vessels_toolkit.utils.torch import unique_first


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
        x: torch.Tensor,
        target: torch.Tensor,
        group_idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        metagroup_idx: Optional[torch.Tensor] = None,
        other_group_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss which enforce the selection of one sample per group, given sample-wise logits and group indices.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N,) containing the predicted logits for each sample.

        target : torch.Tensor
            A tensor of shape (N,) containing whether each sample is the elected one in its group. The sum of target values for each group should be 1.

        group_idx : torch.Tensor
            A tensor of shape (N,) containing the group affiliation for each sample. The values in `group_idx` should be non-negative integers, and samples with the same group index belong to the same group.

        mask : Optional[torch.Tensor], optional
            A boolean tensor of shape (N,) indicating which group should be included in the loss computation. If None, all samples are included.

        metagroup_idx : Optional[torch.Tensor], optional
            A tensor of shape (G,) containing the metagroup affiliation for each group.

        other_group_idx : Optional[torch.Tensor], optional
            A tensor of shape (N,) containing the group affiliation for each sample according to another grouping scheme, used to apply a penalty to samples belonging to metagroup that are not consistent between the two grouping schemes.

        Returns
        -------
        torch.Tensor
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


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        sample_ratio: float = 1.0,
        use_same_idx: bool = False,
    ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.use_same_idx = use_same_idx

    def forward(
        self,
        x: torch.Tensor,
        contrast_idx: torch.Tensor,
        group_idx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss which enforce the selection of one sample per group, given sample-wise logits and group indices.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N,F) containing features vectors for each sample.

        target : torch.Tensor
            A tensor of shape (N,) containing whether each sample is the elected one in its group. The sum of target values for each group should be 1.

        contrast_idx : torch.Tensor
            A tensor of shape (N,) containing affiliation for each sample. Contrastive pairs will be sampled to contains elements with different affiliation according to `main_idx`.

        group_idx: Optional[torch.Tensor] = None
            A tensor of shape (N,) containing  a second affiliation for each sample. If `use_same_idx` is True, contrastive pairs will be sampled to contains elements with same affiliation according to `group_idx` in addition to different affiliation according to `main_idx`.

        Returns
        -------
        torch.Tensor
            A scalar tensor containing the mean cross-entropy loss over all groups.
        """  # noqa: E501
        S = x.shape[0]
        device = x.device

        # === Sample contrastive pairs ===
        if group_idx is not None and self.use_same_idx:
            samples_idx = torch.arange(S, device=device)
            group_idx_, group_counts, group_inv = group_idx.unique(return_counts=True, return_inverse=True)
            G = group_idx_[-1] + 1

            def select_samples(by_sample=None, *, by_group=None):
                nonlocal samples_idx, group_inv, S
                if by_group is not None:
                    if by_group.dtype == torch.bool:
                        mask = group_idx_[by_group]
                    else:
                        mask = torch.zeros(G, device=device, dtype=torch.bool)
                        mask[by_group] = True
                        mask = group_idx_[mask]
                elif by_sample is not None:
                    mask = by_sample
                else:
                    raise ValueError("Either by_samples or by_groups must be provided.")
                samples_idx = samples_idx[mask]
                S = len(samples_idx)
                group_inv = group_inv[mask]
                
            # → Select pairable groups (with more than 1 sample)
            pairable_group_mask = group_counts > 1
            pairable_group = group_idx_[pairable_group_mask]
            select_samples(by_group=pairable_group_mask)
            
            N_group = len(pairable_group)
            N_pair = math.ceil(N_group * self.sample_ratio)
            if N_pair == 0:
                return torch.tensor(0.0, device=device)

            # → Randomly select a subset of pairable groups
            selected_group_idx = pairable_group[torch.randperm(N_group, device=device)[:N_pair]]
            selected_group_mask = torch.zeros(G, device=device, dtype=torch.bool)
            selected_group_mask[selected_group_idx] = True
            select_samples(by_group=selected_group_mask)

            # → Randomly select one sample amongst each valid group
            random_order = torch.randperm(S, device=device)
            inv_sample_order = torch.empty_like(random_order)
            inv_sample_order[random_order] = torch.arange(S, device=device)
            selected_group, s0 = unique_first(group_inv[random_order])
            s0 = inv_sample_order[s0]  # Map back to original sample index

            s0_contrast_idx = contrast_idx[s0]
            group_inv[] = 0 # Set samples of the same group as p0 and same contrast_idx to 0
        else:
            N_pair = x.shape[0] * self.sample_ratio
