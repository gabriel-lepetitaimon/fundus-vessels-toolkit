import warnings

from ..vascular_data_objects import AVLabel, VTree
from .graph_simplification import simplify_passing_nodes


def clean_vtree(vtree: VTree, *, av_attr: str = "av") -> VTree:
    # === Remove terminal branches with unknown type ===
    if av_attr in vtree.branches_attr:
        while to_delete := [b.id for b in vtree.branches() if not b.has_successors and b.attr[av_attr] == AVLabel.UNK]:
            vtree.delete_branches(to_delete, inplace=True)

    # === Remove passing nodes ===
    simplify_passing_nodes(vtree, inplace=True)

    return vtree


def disconnect_crossing_nodes(tree: VTree, inplace=False) -> VTree:
    """
    Merge crossing nodes of a vessel tree.

    Crossing nodes are nodes with two or more incoming branches with successors.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

    Returns
    -------
        The modified graph with the crossing nodes fused.

    """  # noqa: E501
    if not inplace:
        tree = tree.copy()

    crossing_nodes = tree.crossing_nodes_ids()
    if len(crossing_nodes) == 0:
        return tree

    warnings.warn("simplify_crossing_nodes is not implemented yet.", stacklevel=2)
