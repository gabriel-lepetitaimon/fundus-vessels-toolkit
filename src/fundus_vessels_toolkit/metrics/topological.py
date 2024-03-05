import numpy as np
from ..seg2graph.graph_extraction import branches_by_nodes_to_node_graph


def vascular_graph_edit_distance(branch_to_node1, node1_yx, branch_to_node2, node2_yx):
    ny1, nx1 = node1_yx
    ny2, nx2 = node2_yx
    ny1 = ny1[:, None]
    nx1 = nx1[:, None]
    ny2 = ny2[None, :]
    nx2 = nx2[None, :]
    node_dist = np.sqrt((ny1 - ny2)**2 + (nx1 - nx2)**2)
    del ny1, nx1, ny2, nx2

    node_extended_match = node_dist < 10
    node_dist = 1/(node_dist+1e8)
    node_dist[~node_extended_match] = 0
    n1_match = np.where(np.sum(node_extended_match, axis=1))[0]
    n2_match = np.argmax(node_dist[n1_match], axis=0)
    del node_dist, node_extended_match

    lookup_n1_idx = np.concatenate([n1_match, np.isin(np.arange(len(node1_yx[0])), n1_match, invert=True, assume_unique=True)])
    lookup_n2_idx = np.concatenate([n2_match, np.isin(np.arange(len(node2_yx[0])), n2_match, invert=True, assume_unique=True)])

    node_to_branch1 = branch_to_node1.T[lookup_n1_idx]
    node_to_branch2 = branch_to_node2.T[lookup_n2_idx]

    g1 = branches_by_nodes_to_node_graph(node_to_branch1.T)
    g2 = branches_by_nodes_to_node_graph(node_to_branch2.T)

    return 0
