#distutils: language=c++

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.list cimport list as cpplist
# from libc.stdlib cimport malloc, free


DTYPE = int
ctypedef np.int_t DTYPE_t


cdef struct NodeID:
    int id
    bool isPrimary

cdef struct EdgeNode:
    int edge
    int node

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def shortest_secondary_path(np.ndarray[DTYPE_t, ndim=2] adjacency_list,
                            np.ndarray[DTYPE_t, ndim=1] primary_nodes, np.ndarray[DTYPE_t, ndim=1] secondary_nodes):
    assert adjacency_list.ndim == 2 and adjacency_list.shape[1] == 2

    # Instanciate length variables
    cdef int n_primary = primary_nodes.shape[0]
    cdef int n_secondary = secondary_nodes.shape[0]
    cdef int n_nodes = n_primary + n_secondary
    cdef int n_edges = adjacency_list.shape[0]

    # Allocate distance output matrix
    cdef np.ndarray[DTYPE_t, ndim=2] dist = np.empty((n_primary, n_nodes), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3] backtrack_edge_node = np.empty((n_primary, n_nodes, 2), dtype=DTYPE)

    # Allocate to_visit array
    cdef vector[cpplist[int]] to_visit = vector[cpplist[int]](2)

    # Declare temporary variables
    cdef int n1, n2, d, p
    cdef NodeID nid1, nid2
    cdef int now, next

    # Allocate node_lookup
    cdef vector[NodeID] node_lookup = vector[NodeID](n_nodes)

    # Allocate bidirectional adjacency list
    cdef vector[cpplist[EdgeNode]] adj = vector[cpplist[EdgeNode]](n_nodes)

    # --------------------
    # Initialize lookup table
    ###  print("Initialize lookup table")
    for primary_id in range(n_primary):
        p = primary_nodes[primary_id]
        node_lookup[p] = NodeID(primary_id, True)
    for secondary_id in range(n_secondary):
        p = secondary_nodes[secondary_id]
        node_lookup[p] = NodeID(secondary_id, False)

    # Initiliaze distance matrix
    ###  print("Initialize distance matrix")
    dist.fill(-1)
    for primary_id in range(n_primary):
        p = primary_nodes[primary_id]
        dist[primary_id, p] = 0

    # Initialize backtrack node matrix
    ###  print("Initialize backtrack node matrix")
    backtrack_edge_node.fill(-1)

    # Initialize adjacency list
    ###  print("Populate adjacency list")
    for edge_id in range(n_edges):
        n1 = adjacency_list[edge_id, 0]
        n2 = adjacency_list[edge_id, 1]
        nid1 = node_lookup[n1]
        nid2 = node_lookup[n2]

        adj[n1].push_back(EdgeNode(edge_id, n2))
        adj[n2].push_back(EdgeNode(edge_id, n1))

        if nid1.isPrimary and nid2.isPrimary:
            dist[nid1.id, n2] = 1
            dist[nid2.id, n1] = 1
            # Update backtrack edge and nodes
            backtrack_edge_node[nid1.id, n2, 0] = edge_id
            backtrack_edge_node[nid1.id, n2, 1] = n1

            backtrack_edge_node[nid2.id, n1, 0] = n2
            backtrack_edge_node[nid2.id, n1, 1] = n2

    # Compute distance from primary nodes to all other nodes
    ###  print("Compute distance from primary nodes to all other nodes")
    for p_primary_id in range(n_primary):
        now = 0
        next = 1-now

        p = primary_nodes[p_primary_id]
        to_visit[now].clear()
        for connected in adj[p]:
            to_visit[now].push_back(connected.node)
            backtrack_edge_node[p_primary_id, connected.node, 0] = connected.edge
            backtrack_edge_node[p_primary_id, connected.node, 1] = p
        d = 1

        ###  print(f'Compute distance from {p} to all other nodes')
        ###  print(f'adj: {adj[p]}')
        ###  print(f'dist: {dist[node_lookup[p][1]]}')

        while not to_visit[now].empty():
            # Clear next to_visit
            to_visit[next].clear()

            ###  print(f'  [d={d}]: Nodes to visit {to_visit[now]}')

            # Iterate over nodes to_visit
            for n in to_visit[now]:
                # Update distance
                dist[p_primary_id, n] = d

                # If n is a primary node, don't follow edges
                if node_lookup[n].isPrimary:
                    ###  print(f'   {n} -> ... skipped')
                    continue

                # Otherwise Follow edges of node n
                for connected in adj[n]:
                    # Skip if already visited or if m is p
                    if dist[p_primary_id, connected.node] >= 0:
                        ###  print(f'   {n} -> {m} skipped')
                        continue
                    # Update backtrack node
                    backtrack_edge_node[p_primary_id, connected.node, 0] = connected.edge   # edge
                    backtrack_edge_node[p_primary_id, connected.node, 1] = n                # node
                    ###  print(f'   {n} -> {m}')
                    # Add m to next to_visit
                    to_visit[next].push_back(connected.node)

            # Switch to_visit and next to_visit
            now = next
            next = 1-now

            # Increment distance
            d += 1

    return dist, backtrack_edge_node