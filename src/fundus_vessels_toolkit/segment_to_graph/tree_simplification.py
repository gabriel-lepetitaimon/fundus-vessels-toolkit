import warnings
from typing import Literal, Optional, overload

import numpy as np

from fundus_toolkits import AVLabel
from fundus_vessels_toolkit.vascular_data_objects.vgraph import NodeIndices, NodeIndicesLike

from ..vascular_data_objects import VTree
from .graph_simplification import simplify_passing_nodes


def clean_vtree(vtree: VTree, *, passing_node_min_angle: float = 0, av_attr: str = "av") -> VTree:
    # === Remove terminal branches with unknown type ===
    if av_attr in vtree.branch_attr:
        while to_delete := [b.id for b in vtree.branches() if not b.has_successors and b.attr[av_attr] == AVLabel.UNK]:
            vtree.delete_branch(to_delete, inplace=True)

    # === Remove passing nodes ===
    simplify_passing_nodes(vtree, min_angle=passing_node_min_angle, inplace=True)

    return vtree


@overload
def disconnect_crossing(
    tree: VTree,
    nodes: Optional[NodeIndicesLike] = None,
    *,
    redefined_subtree_by: Optional[Literal["TANGENT"] | str] = None,
    fuse_passing_nodes=True,
    return_new_nodes: Literal[False] = False,
    inplace=False,
) -> VTree: ...
@overload
def disconnect_crossing(
    tree: VTree,
    nodes: Optional[NodeIndicesLike] = None,
    *,
    redefined_subtree_by: Optional[Literal["TANGENT"] | str] = None,
    fuse_passing_nodes=True,
    return_new_nodes: Literal[True],
    inplace=False,
) -> tuple[VTree, NodeIndices]: ...
def disconnect_crossing(
    tree: VTree,
    nodes: Optional[NodeIndicesLike] = None,
    *,
    redefined_subtree_by: Optional[Literal["TANGENT"] | str] = None,
    fuse_passing_nodes=True,
    return_new_nodes=False,
    inplace=False,
) -> VTree | tuple[VTree, NodeIndices]:
    """
    Disconnect crossing nodes by splitting them into multiple nodes.

    Parameters
    ----------
    tree : VTree
        The tree to simplify.
    nodes : Optional[NodeIndicesLike], optional
        The nodes to disconnect. If None, all crossing nodes will be disconnected, by default None.
    fuse_passing_nodes : bool, optional
        Whether to fuse the resulting passing nodes after splitting, by default True.
    return_new_nodes : bool, optional
        Whether to return the new nodes created by the splitting, by default False.
    inplace : bool, optional
        Whether to modify the tree in place or return a new tree, by default False.

    Returns
    -------
    VTree
        The simplified tree.

    NodeIndices, optional
        The new nodes created by the splitting, only returned if `return_new_nodes` is True

    """  # noqa: E501
    if not inplace:
        tree = tree.copy()

    if nodes is None:
        nodes_ = tree.crossing_nodes_ids()
    else:
        nodes_ = np.unique(tree.as_node_ids(nodes))
        invalid_nodes = np.intersect1d(nodes_, tree.endpoint_nodes(), assume_unique=True)
        if len(invalid_nodes) > 0:
            raise ValueError(f"The following nodes are not crossing nodes: {invalid_nodes}")

    if len(nodes_) == 0:
        return tree

    new_nodes_ids = set()

    for node in tree.nodes(nodes_, dynamic_iterator=True):
        parent_branches = list(node.incoming_branches())
        child_branches = list(node.outgoing_branches())
        branches = parent_branches + child_branches
        if len(branches) == 2:
            # Consider both nodes as terminal and split them
            tree.split_node(node.id, [[b.id] for b in branches], inplace=True)
            continue

        # Get tangents at the node
        tangents = {b: t for b, t in zip(node.adjacent_branch_ids, node.tips_tangent(), strict=True)}

        # Force incoming branches to be in different clusters
        clusters = [[b.id] for b in parent_branches]

        if len(clusters) < 2 and len(branches) >= 3:
            # If there is only one incoming branch but more than 3 branches total (only for user defined nodes),
            assert len(clusters) == 1, "There at least one incoming branch"
            # set the closest branch to the incoming branch as a separate cluster
            t0 = tangents[clusters[0][0]]
            child_t = np.array([tangents[b.id] for b in child_branches])
            closest_child = np.argmax(np.sum(t0[None, :] * child_t, axis=1))
            clusters.append([child_branches.pop(closest_child).id])

        if redefined_subtree_by == "TANGENT":
            clusters_t = -np.array([tangents[cluster[0]] for cluster in clusters])

            # Clusters child branches based on tangents
            for child_branch in child_branches:
                t_child = tangents[child_branch.id]
                closest_cluster = np.argmax(np.sum(clusters_t * t_child[None, :], axis=1))
                clusters[closest_cluster].append(child_branch.id)
        elif redefined_subtree_by is not None:
            if redefined_subtree_by not in tree.branch_attr:
                raise ValueError(f"Branch attribute '{redefined_subtree_by}' not found in the tree.")
            cluster_attr = tree.branch_attr.loc[[cluster[0] for cluster in clusters], redefined_subtree_by]
            for child_branch in child_branches:
                attr = tree.branch_attr.loc[child_branch.id, redefined_subtree_by]
                for i, c_attr in enumerate(cluster_attr):
                    if attr == c_attr:
                        clusters[i].append(child_branch.id)
                        break
                else:
                    raise ValueError(
                        f"Branch attribute '{redefined_subtree_by}' value '{attr}' not found in any of the clusters."
                    )
        else:
            cluster_by_in = {cluster[0]: c for c, cluster in enumerate(clusters)}
            for child_branch in child_branches:
                parent_id = tree.branch_tree[child_branch.id]
                if parent_id == -1:
                    clusters.append([child_branch.id])
                else:
                    clusters[cluster_by_in[parent_id]].append(child_branch.id)

        # Split nodes
        _, splitted_nodes_ids = tree.split_node(node.id, clusters, inplace=True, return_node_ids=True)
        new_nodes_ids.update(splitted_nodes_ids)

    new_nodes = list(tree.nodes(list(new_nodes_ids)))

    if fuse_passing_nodes:
        simplify_passing_nodes(tree, inplace=True, only_fusable=list(new_nodes_ids))

    new_nodes_ids = np.array([n.id for n in new_nodes if n.is_valid()])

    return tree if not return_new_nodes else (tree, new_nodes_ids)
