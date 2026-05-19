from typing import Literal

import numpy as np
import torch
from skimage.morphology import skeletonize as skimage_skeletonize

from ...utils.cpp_optimized import split_by
from ...utils.safe_import import import_cv2


def vascular_graph_edit_distance(branch_to_node1, node1_yx, branch_to_node2, node2_yx):
    ny1, nx1 = node1_yx
    ny2, nx2 = node2_yx
    ny1 = ny1[:, None]
    nx1 = nx1[:, None]
    ny2 = ny2[None, :]
    nx2 = nx2[None, :]
    node_dist = np.sqrt((ny1 - ny2) ** 2 + (nx1 - nx2) ** 2)
    del ny1, nx1, ny2, nx2

    node_extended_match = node_dist < 10
    node_dist = 1 / (node_dist + 1e8)
    node_dist[~node_extended_match] = 0
    n1_match = np.where(np.sum(node_extended_match, axis=1))[0]
    n2_match = np.argmax(node_dist[n1_match], axis=0)
    del node_dist, node_extended_match

    lookup_n1_idx = np.concatenate(
        [n1_match, np.isin(np.arange(len(node1_yx[0])), n1_match, invert=True, assume_unique=True)]
    )
    lookup_n2_idx = np.concatenate(
        [n2_match, np.isin(np.arange(len(node2_yx[0])), n2_match, invert=True, assume_unique=True)]
    )

    node_to_branch1 = branch_to_node1.T[lookup_n1_idx]
    node_to_branch2 = branch_to_node2.T[lookup_n2_idx]

    return 0


def valid_path_ratio(gt_mask, pred_mask, skeletonize=False):
    cv2 = import_cv2()
    n_pred, pred_cc = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)
    n_gt, gt_cc = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    if skeletonize:
        pred_ref = skimage_skeletonize(pred_cc > 0) * pred_cc
        gt_ref = skimage_skeletonize(gt_cc > 0) * gt_cc
    else:
        pred_ref = pred_cc
        gt_ref = gt_cc

    pred_cc, gt_cc = pred_cc.flatten(), gt_cc.flatten()
    pred_ref, gt_ref = pred_ref.flatten(), gt_ref.flatten()

    gt_by_preds = split_by(gt_cc, pred_ref - 1, n_pred - 1)
    pred_by_gts = split_by(pred_cc, gt_ref - 1, n_gt - 1)

    pred_path_total = 0
    pred_path_valid = 0
    for gt_by_pred in gt_by_preds:
        n = len(gt_by_pred)
        if n == 0:
            continue
        vals, counts = np.unique(gt_by_pred, return_counts=True)
        if vals[0] == 0:
            counts = counts[1:]
        pred_path_total += n * (n - 1) // 2
        pred_path_valid += np.sum(counts * (counts - 1) // 2)

    gt_path_total = 0
    gt_path_valid = 0
    for pred_by_gt in pred_by_gts:
        n = len(pred_by_gt)
        if n == 0:
            continue
        vals, counts = np.unique(pred_by_gt, return_counts=True)
        if vals[0] == 0:
            counts = counts[1:]
        gt_path_total += n * (n - 1) // 2
        gt_path_valid += np.sum(counts * (counts - 1) // 2)

    return pred_path_valid / pred_path_total, gt_path_valid / gt_path_total


def valid_path_ratio_cpp(gt_mask, pred_mask, skeletonize=False):
    from ...utils.cpp_extensions.fvt_cpp import valid_path_ratio

    cv2 = import_cv2()
    n_pred, pred_cc = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)
    n_gt, gt_cc = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    if skeletonize:
        pred_ref = skimage_skeletonize(pred_cc > 0) * pred_cc
        gt_ref = skimage_skeletonize(gt_cc > 0) * gt_cc
    else:
        pred_ref = pred_cc
        gt_ref = gt_cc

    pred_cc, gt_cc = pred_cc.flatten(), gt_cc.flatten()
    pred_ref, gt_ref = pred_ref.flatten(), gt_ref.flatten()

    pred_cc = torch.from_numpy(pred_cc).flatten().to(torch.int32)
    gt_cc = torch.from_numpy(gt_cc).flatten().to(torch.int32)
    pred_ref = torch.from_numpy(pred_ref).flatten().to(torch.int32)
    gt_ref = torch.from_numpy(gt_ref).flatten().to(torch.int32)
    (gt_ratio, gt_n_path), _ = valid_path_ratio(gt_ref, pred_cc, n_gt, n_pred)
    (pred_ratio, pred_n_path), _ = valid_path_ratio(pred_ref, gt_cc, n_pred, n_gt)
    return (pred_ratio, pred_n_path), (gt_ratio, gt_n_path)


def sample_valid_path_ratio(gt_mask, pred_mask, n_samples=1000, cuda: Literal["if-available"] | bool = "if-available"):
    from skimage.segmentation import expand_labels

    from ...utils.cpp_extensions.fvt_cpp import shortest_skeleton_path_length

    gt_mask = gt_mask > 0
    pred_mask = pred_mask > 0
    gt_skel = skimage_skeletonize(gt_mask)
    pred_skel = skimage_skeletonize(pred_mask)

    with torch.no_grad():
        gt_skel = torch.from_numpy(gt_skel).int()
        pred_skel = torch.from_numpy(pred_skel).int()
        out_gt = shortest_skeleton_path_length(gt_skel)
        out_pred = shortest_skeleton_path_length(pred_skel)

        gt_skel = torch.from_numpy(expand_labels(gt_skel.numpy(), 20) & (gt_mask))
        pred_skel = torch.from_numpy(expand_labels(pred_skel.numpy(), 20) & (pred_mask))

        if cuda == "if-available":
            cuda = torch.cuda.is_available()
        elif cuda is True:
            assert torch.cuda.is_available(), "CUDA is not available"

        if cuda:
            out_gt = [_.cuda() for _ in out_gt]
            out_pred = [_.cuda() for _ in out_pred]
            gt_skel = gt_skel.cuda()
            pred_skel = pred_skel.cuda()

        common_mask_ids = mask.argwhere()
        N_px = len(common_mask_ids)
        if n_samples < N_px**2:
            idx1 = torch.randperm(N_px * (n_samples // N_px + 1))[:n_samples] % N_px
            idx2 = torch.randperm(N_px * (n_samples // N_px + 1))[:n_samples] % N_px
        else:
            idx1 = torch.arange(N_px).repeat_interleave(N_px)
            idx2 = torch.arange(N_px).repeat(N_px)

        p1 = common_mask_ids[idx1]
        p2 = common_mask_ids[idx2]

        def sample_shortest_path(p1, p2, skel_pid, p_branch, p_pos, branch_len, shortest_path):
            p1, p2 = skel_pid[*p1.T], skel_pid[*p2.T]
            b1, b2 = p_branch[p1], p_branch[p2]
            p1_pos, p2_pos = p_pos[p1], p_pos[p2]
            shortest_paths = shortest_path[b1, b2].clone()
            shortest_paths[..., 0, :] += p1_pos
            shortest_paths[..., 1, :] += branch_len[b1] - p1_pos
            shortest_paths[..., :, 0] += p2_pos
            shortest_paths[..., :, 1] += branch_len[b2] - p2_pos
            shortest_paths.reshape(-1, 4)
            return shortest_paths.min(dim=1).values

        gt_shortest_path = sample_shortest_path(p1, p2, gt_skel, gt_p_branch, gt_p_pos, gt_branch_len, gt_shortest_path)
        pred_shortest_path = sample_shortest_path(
            p1, p2, pred_skel, pred_p_branch, pred_p_pos, pred_branch_len, pred_shortest_path
        )
        return abs(pred_shortest_path - gt_shortest_path) <= gt_shortest_path * 1.1 + 5
