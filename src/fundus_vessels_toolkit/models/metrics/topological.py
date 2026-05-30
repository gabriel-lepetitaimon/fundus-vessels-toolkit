from doctest import debug
import itertools
from typing import Literal

import numpy as np
import torch

from ...segment_to_graph.skeletonize import skeletonize
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


def valid_path_ratio(gt_mask, pred_mask, skeleton=False):
    cv2 = import_cv2()
    n_pred, pred_cc = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)
    n_gt, gt_cc = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    if skeleton:
        pred_ref = skeletonize(pred_mask) * pred_cc
        gt_ref = skeletonize(gt_mask) * gt_cc
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

    return (pred_path_valid / pred_path_total, pred_path_total), (gt_path_valid / gt_path_total, gt_path_total)


def valid_path_ratio_cpp(gt_mask, pred_mask, skeleton=False):
    from ...utils.cpp_extensions.fvt_cpp import valid_path_ratio

    cv2 = import_cv2()
    n_pred, pred_cc = cv2.connectedComponents(pred_mask.astype(np.uint8), connectivity=8)
    n_gt, gt_cc = cv2.connectedComponents(gt_mask.astype(np.uint8), connectivity=8)
    if skeleton:
        pred_ref = skeletonize(pred_cc > 0) * pred_cc
        gt_ref = skeletonize(gt_cc > 0) * gt_cc
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


def sample_valid_path_ratio(
    mask1,
    mask2,
    n_samples: int = int(1e5),
    cuda: Literal["if-available"] | bool = "if-available",
    batch_size: int = int(1e4),
):
    from skimage.segmentation import expand_labels

    from ...utils.cpp_extensions.fvt_cpp import shortest_skeleton_path_length

    cv2 = import_cv2()

    mask1 = mask1 > 0
    mask2 = mask2 > 0

    # n_cc1, cc1 = cv2.connectedComponents(mask1.astype(np.uint8), connectivity=8)
    # n_cc2, cc2 = cv2.connectedComponents(mask2.astype(np.uint8), connectivity=8)

    skel_mask1 = skeletonize(mask1)
    skel_mask2 = skeletonize(mask2)

    with torch.no_grad():
        skel_mask1 = torch.from_numpy(skel_mask1).int()
        skel_mask2 = torch.from_numpy(skel_mask2).int()
        out1 = shortest_skeleton_path_length(skel_mask1)
        out2 = shortest_skeleton_path_length(skel_mask2)
        # cc1 = torch.from_numpy(cc1).int()
        # cc2 = torch.from_numpy(cc2).int()
        # out: p_yx, p_branch, p_pos, branch_len, shortest_path, branch_subgraph

        skel1 = torch.from_numpy(expand_labels(skel_mask1.numpy(), 40) * mask1)
        skel2 = torch.from_numpy(expand_labels(skel_mask2.numpy(), 40) * mask2)

        if cuda == "if-available":
            cuda = torch.cuda.is_available()
        elif cuda is True:
            assert torch.cuda.is_available(), "CUDA is not available"

        if cuda:
            out1 = tuple(o.cuda() for o in out1)
            out2 = tuple(o.cuda() for o in out2)
            skel1 = skel1.cuda()
            skel2 = skel2.cuda()
            # cc1 = cc1.cuda()
            # cc2 = cc2.cuda()
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        def sample_pairs(p_subgraph, N):
            S = p_subgraph[-1] + 1
            lookup = p_subgraph.argsort()
            subgraph_counts = p_subgraph.bincount(minlength=S)
            pairs = []
            # if N >= (subgraph_counts * (subgraph_counts - 1) / 2).sum():
            # If N is larger than the total number of pairs, return all possible pairs
            offset = 0
            for c in subgraph_counts:
                if c == 0:
                    continue
                pairs.append(torch.combinations(torch.arange(offset, offset + c, device=device), r=2))
                offset += c
            pairs = lookup[torch.cat(pairs, dim=0)]
            # else:
            # Otherwise, sample pairs randomly from the subgraphs until we have enough pairs
            #    for _ in range(N // len(p_subgraph) + 1):
            #        offset = 0
            #        for c in subgraph_counts:
            #            if c == 0:
            #                continue
            #            p1 = torch.arange(c, device=device)
            #            p2 = torch.randperm(c, device=device)
            #            pairs.append(torch.stack([p1, p2], dim=-1) + offset)
            #            offset += c
            #    pairs = lookup[torch.cat(pairs, dim=0)]
            if N < len(pairs):
                pairs = pairs[torch.randperm(len(pairs))[:N]]
            return pairs

        p_yx1, p_branch1, branch_subgraph1 = out1[0], out1[1], out1[-1]
        p_yx2, p_branch2, branch_subgraph2 = out2[0], out2[1], out2[-1]
        pairs1 = sample_pairs(p_subgraph=branch_subgraph1[p_branch1], N=n_samples // 2)
        pairs2 = sample_pairs(p_subgraph=branch_subgraph2[p_branch2], N=n_samples // 2)

        def compute_shortest_path(
            pairs, p_branch, p_pos, branch_len, shortest_path, branch_subgraph, assume_valid=False
        ):
            N_pairs = pairs.shape[0]
            if not assume_valid:
                valid_pairs = (pairs >= 0).all(dim=-1)
                pairs = pairs[valid_pairs, :]

            p1, p2 = pairs.unbind(dim=1)
            b1, b2 = p_branch[p1], p_branch[p2]
            s1, s2 = branch_subgraph[b1], branch_subgraph[b2]

            if not assume_valid:
                same_graph = s1 == s2
                valid_pairs[valid_pairs.clone()] = same_graph
                p1, p2, b1, b2 = p1[same_graph], p2[same_graph], b1[same_graph], b2[same_graph]

            p1_pos, p2_pos = p_pos[p1], p_pos[p2]
            paths = shortest_path[b1, b2].clone()
            paths[..., 0, :] += p1_pos[:, None]
            paths[..., 1, :] += branch_len[b1, None] - p1_pos[:, None]
            paths[..., :, 0] += p2_pos[:, None]
            paths[..., :, 1] += branch_len[b2, None] - p2_pos[:, None]
            shortest_path = paths.reshape(-1, 4).min(dim=1).values

            same_branch = b1 == b2
            shortest_path[same_branch] = (p1_pos[same_branch] - p2_pos[same_branch]).abs()

            if not assume_valid:
                out = torch.full((N_pairs,), float("inf"), device=device)
                out[valid_pairs] = shortest_path
                return out
            else:
                return shortest_path

        sum_valid = 0
        sum_same_path = 0
        for pairs in torch.split(pairs1, batch_size):
            yx1 = p_yx1[pairs]
            shortest_path1 = compute_shortest_path(skel1[*yx1.unbind(-1)] - 1, *out1[1:], assume_valid=True)
            shortest_path2 = compute_shortest_path(skel2[*yx1.unbind(-1)] - 1, *out2[1:], assume_valid=False)
            valid = shortest_path2 != float("inf")
            sum_valid += valid.sum()
            path_diff = (shortest_path2[valid] - shortest_path1[valid]).abs()
            sum_same_path += (path_diff <= shortest_path1[valid] * 1.1 + 5).sum()
        invalid_ratio1 = sum_valid / len(pairs1)
        same_ratio1 = sum_same_path / sum_valid

        sum_valid = 0
        sum_same_path = 0
        for pairs in torch.split(pairs2, batch_size):
            yx2 = p_yx2[pairs]
            shortest_path1 = compute_shortest_path(skel1[*yx2.unbind(-1)] - 1, *out1[1:], assume_valid=False)
            shortest_path2 = compute_shortest_path(skel2[*yx2.unbind(-1)] - 1, *out2[1:], assume_valid=True)
            valid2 = shortest_path1 != float("inf")
            sum_valid += valid2.sum()
            path_diff = (shortest_path1[valid2] - shortest_path2[valid2]).abs()
            sum_same_path += (path_diff <= shortest_path2[valid2] * 1.1 + 5).sum()
        invalid_ratio2 = sum_valid / len(pairs2)
        same_ratio2 = sum_same_path / sum_valid
        return (invalid_ratio1.item(), same_ratio1.item()), (invalid_ratio2.item(), same_ratio2.item())
