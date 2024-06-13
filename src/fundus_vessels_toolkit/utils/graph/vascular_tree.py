import numpy as np
import pandas as pd


def extract_bifurcations_parameters(branches_calibre, branches_tangent, branches_list, directed=True) -> pd.DataFrame:
    """
    Extracts the parameters of the bifurcations from the branches.
    """
    adj_list = [[] for _ in range(branches_list.max() + 1)]
    # Create the adjacency list, storing for each node the list of branches and wether they are incident to the node
    for branchID, (node0, node1) in enumerate(branches_list):
        adj_list[node0].append((branchID, False))  # The branch is outgoing from the node
        adj_list[node1].append((branchID, True))  # The branch is incident to the node

    bifurcations = []
    for nodeID, node_adjacency in enumerate(adj_list):
        if len(node_adjacency) == 3:
            if any(len(branches_tangent[_[0]]) == 0 for _ in node_adjacency):
                continue
            if directed:
                b0 = [b for b, incident in node_adjacency if incident][0]
                b1, b2 = [b for b, incident in node_adjacency if b != b0]
                dir0 = (np.arctan2(*branches_tangent[b0][-1]) + np.pi) % (2 * np.pi)
                dir1 = np.arctan2(*branches_tangent[b1][0])
                dir2 = np.arctan2(*branches_tangent[b2][0])
                c0 = np.mean(branches_calibre[b0][-10:])
                c1 = np.mean(branches_calibre[b1][:10])
                c2 = np.mean(branches_calibre[b2][:10])
            else:
                (b0, b0_incident), (b1, b1_incident), (b2, b2_incident) = node_adjacency
                if b0_incident:
                    dir0 = (np.arctan2(*branches_tangent[b0][-1]) + np.pi) % (2 * np.pi)
                    c0 = np.mean(branches_calibre[b0][-10:])
                else:
                    dir0 = np.arctan2(*branches_tangent[b0][0])
                    c0 = np.mean(branches_calibre[b0][:10])
                if b1_incident:
                    dir1 = (np.arctan2(*branches_tangent[b1][-1]) + np.pi) % (2 * np.pi)
                    c1 = np.mean(branches_calibre[b1][-10:])
                else:
                    dir1 = np.arctan2(*branches_tangent[b1][0])
                    c1 = np.mean(branches_calibre[b1][:10])
                if b2_incident:
                    dir2 = (np.arctan2(*branches_tangent[b2][-1]) + np.pi) % (2 * np.pi)
                    c2 = np.mean(branches_calibre[b2][-10:])
                else:
                    dir2 = np.arctan2(*branches_tangent[b2][0])
                    c2 = np.mean(branches_calibre[b2][:10])

                # Use the largest branch as the main branch
                if c1 > c0 and c1 > c2:
                    b0, b1 = b1, b0
                    dir0, dir1 = dir1, dir0
                    c0, c1 = c1, c0
                elif c2 > c0 and c2 > c1:
                    b0, b2 = b2, b0
                    dir0, dir2 = dir2, dir0
                    c0, c2 = c2, c0

            # Sort the branches by their direction
            dir1 = (dir1 - dir0) % (2 * np.pi)
            dir2 = (dir2 - dir0) % (2 * np.pi)
            if dir1 > dir2:
                b1, b2 = b2, b1
                dir1, dir2 = dir2, dir1
                c1, c2 = c2, c1

            # Compute the angles between the incident branch and the outgoing branches
            theta1 = np.pi - dir1
            theta2 = dir2 - np.pi

            # Ensure branch1 is the main branch (the one with the smallest angle with the incident branch)
            if theta1 > theta2:
                b1, b2 = b2, b1
                theta1, theta2 = theta2, theta1
                c1, c2 = c2, c1

            bifurcations.append(
                dict(
                    nodeID=int(nodeID),
                    branch0=int(b0),
                    branch1=int(b1),
                    branch2=int(b2),
                    theta1=theta1 * 180 / np.pi,
                    theta2=theta2 * 180 / np.pi,
                    c0=c0,
                    c1=c1,
                    c2=c2,
                )
            )

    return pd.DataFrame(bifurcations).set_index("nodeID")
