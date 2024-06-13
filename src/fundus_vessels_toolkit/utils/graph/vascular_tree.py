import numpy as np
import pandas as pd


def extract_bifurcations_parameters(branches_calibre, branches_tangent, branches_list, directed=True) -> pd.DataFrame:
    """
    Extracts the parameters of the bifurcations from the branches.
    """
    adj_list = [set() for _ in range(branches_list.max() + 1)]
    for branchID, (node0, node1) in enumerate(branches_list):
        adj_list[node0].add(branchID)
        adj_list[node1].add(-branchID)

    bifurcations = []
    for nodeID, node_adjacency in enumerate(adj_list):
        if len(node_adjacency) == 3:
            if any(len(branches_tangent[np.abs(_)]) == 0 for _ in node_adjacency):
                continue
            if directed:
                branch0 = [-b for b in node_adjacency if b < 0][0]
                branch1, branch2 = [b for b in node_adjacency if b != branch0]
                dir0 = (np.arctan2(*branches_tangent[branch0][-1]) + np.pi) % (2 * np.pi)
                dir1 = np.arctan2(*branches_tangent[branch1][0])
                dir2 = np.arctan2(*branches_tangent[branch2][0])
                c0 = np.mean(branches_calibre[branch0][-10:])
                c1 = np.mean(branches_calibre[branch1][:10])
                c2 = np.mean(branches_calibre[branch2][:10])
            else:
                branch0, branch1, branch2 = node_adjacency
                if branch0 < 0:
                    branch0 = -branch0
                    dir0 = (np.arctan2(*branches_tangent[branch0][-1]) + np.pi) % (2 * np.pi)
                    c0 = np.mean(branches_calibre[branch0][-10:])
                else:
                    dir0 = np.arctan2(*branches_tangent[branch0][0])
                    c0 = np.mean(branches_calibre[branch0][:10])
                if branch1 < 0:
                    branch1 = -branch1
                    dir1 = (np.arctan2(*branches_tangent[branch1][-1]) + np.pi) % (2 * np.pi)
                    c1 = np.mean(branches_calibre[branch1][-10:])
                else:
                    dir1 = np.arctan2(*branches_tangent[branch1][0])
                    c1 = np.mean(branches_calibre[branch1][:10])
                if branch2 < 0:
                    branch2 = -branch2
                    dir2 = (np.arctan2(*branches_tangent[branch2][-1]) + np.pi) % (2 * np.pi)
                    c2 = np.mean(branches_calibre[branch2][-10:])
                else:
                    dir2 = np.arctan2(*branches_tangent[branch2][0])
                    c2 = np.mean(branches_calibre[branch2][:10])

                # Use the largest branch as the main branch
                if c1 > c0 and c1 > c2:
                    branch0, branch1 = branch1, branch0
                    dir0, dir1 = dir1, dir0
                    c0, c1 = c1, c0
                elif c2 > c0 and c2 > c1:
                    branch0, branch2 = branch2, branch0
                    dir0, dir2 = dir2, dir0
                    c0, c2 = c2, c0

            # Sort the branches by their direction
            dir1 = (dir1 - dir0) % (2 * np.pi)
            dir2 = (dir2 - dir0) % (2 * np.pi)
            if dir1 > dir2:
                branch1, branch2 = branch2, branch1
                dir1, dir2 = dir2, dir1
                c1, c2 = c2, c1

            # Compute the angles between the incident branch and the outgoing branches
            theta1 = np.pi - dir1
            theta2 = dir2 - np.pi

            # Ensure branch1 is the main branch (the one with the smallest angle with the incident branch)
            if theta1 > theta2:
                branch1, branch2 = branch2, branch1
                theta1, theta2 = theta2, theta1
                c1, c2 = c2, c1

            bifurcations.append(
                dict(
                    nodeID=int(nodeID),
                    branch0=int(branch0),
                    branch1=int(branch1),
                    branch2=int(branch2),
                    theta1=theta1 * 180 / np.pi,
                    theta2=theta2 * 180 / np.pi,
                    c0=c0,
                    c1=c1,
                    c2=c2,
                )
            )

    return pd.DataFrame(bifurcations).set_index("nodeID")
