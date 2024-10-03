from typing import Optional

import numpy as np
import numpy.typing as npt

from ..vascular_data_objects import AVLabel, VBranchGeoData, VTree


def clean_vtree(vtree: VTree, *, av_attr: str = "av") -> VTree:
    # === Remove terminal branches with unknown type ===
    if av_attr in vtree.branches_attr:
        while to_delete := [b.id for b in vtree.branches() if not b.has_successors and b.attr[av_attr] == AVLabel.UNK]:
            vtree.delete_branches(to_delete, inplace=True)

    # === Remove passing nodes ===
    simplify_passing_nodes(vtree, inplace=True)

    return vtree


def simplify_passing_nodes(
    vtree: VTree,
    *,
    not_fusable: Optional[npt.ArrayLike] = None,
    min_angle: float = 0,
    with_same_label=None,
    inplace=False,
) -> VTree:
    """
    Merge nodes of the vessel graph that are connected to only 2 branches.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        not_fusable:
            A list of nodes that should not be merged.

        min_angle:
            Under this minimum angle (in degrees) between the two branches connected to a passing node, the node is considered as a junction and is not removed.

            (Require the terminaison tangents field to be field in `VGeometricData`)

        with_same_label:
            If not None, the nodes are merged only if they have the same label.
            If a string, use ``vessel_graph.branches_attr[with_same_label]`` as the labels.

    Returns
    -------
        The modified graph with the nodes merged.

    """  # noqa: E501
    incident_branches = None

    # === Get all nodes with degree 2 ===
    indegrees = vtree.node_indegree()
    outdegrees = vtree.node_outdegree()
    passing_nodes = np.argwhere((indegrees == 1) & (outdegrees == 1)).flatten()

    if not_fusable is not None:
        # === Filter out nodes that should not be merged ===
        passing_nodes = np.setdiff1d(passing_nodes, not_fusable)

    if min_angle > 0:
        # === Filter out nodes with a too small angle between their two incident branches ===
        geo_data = vtree.geometric_data()
        d = geo_data.tip_data_around_node([VBranchGeoData.Fields.TIPS_TANGENT], passing_nodes)
        incident_branches = np.stack(d["branches"])
        t = np.stack(d[VBranchGeoData.Fields.TIPS_TANGENT])
        cos = np.sum(t[..., 0] * t[..., 1], axis=1)
        fuseable_nodes = cos >= np.cos(np.deg2rad(min_angle))
        passing_nodes = passing_nodes[fuseable_nodes]
        incident_branches = incident_branches[fuseable_nodes]

    if with_same_label is not None:
        # === Filter nodes which don't have the same label ===
        if isinstance(with_same_label, str):
            # Attempt to get the labels from the branches attributes
            with_same_label = vtree.branches_attr[with_same_label]
        with_same_label = np.asarray(with_same_label)

        if incident_branches is None:
            # Get the incident branches if not already computed
            incident_branches = np.stack(vtree.incident_branches_individual(passing_nodes))

        same_label = with_same_label[incident_branches[:, 0]] == with_same_label[incident_branches[:, 1]]
        passing_nodes = passing_nodes[same_label]
        incident_branches = incident_branches[same_label]

    if len(passing_nodes):
        return vtree.fuse_nodes(
            passing_nodes, inplace=inplace, quiet_invalid_node=True, incident_branches=incident_branches
        )
    return vtree
