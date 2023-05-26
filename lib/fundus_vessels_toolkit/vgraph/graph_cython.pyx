#distutils: language=c++

cimport cython
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free


DTYPE = int
ctypedef np.int_t DTYPE_t

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
    cdef np.ndarray[DTYPE_t, ndim=2] backtrack_node = np.empty((n_primary, n_nodes), dtype=DTYPE)

    # Allocate to_visit array
    cdef vector[int] to_visit1 = vector[int]()
    cdef vector[int] to_visit2 = vector[int]()
    cdef vector[int] *to_visit
    cdef vector[int] *next_to_visit

    # Declare temporary variables
    cdef int n1, n2, d, p
    cdef int[2] p1, p2
    cdef bool t

    # Allocate node_lookup
    cdef int[2] *node_lookup
    node_lookup = <int[2] *> malloc(sizeof(int[2]) * n_nodes)

    # Allocate bidirectional adjacency list
    cdef vector[int] *adj
    adj = <vector[int] *> malloc(sizeof(vector[int]) * n_nodes)

    # --------------------
    # Initialize lookup table
    for primary_id in range(n_primary):
        p = primary_nodes[primary_id]
        node_lookup[p][0] = 0
        node_lookup[p][1] = primary_id
    for secondary_id in range(n_secondary):
        p = secondary_nodes[secondary_id]
        node_lookup[p][0] = 1
        node_lookup[p][1] = secondary_id

    # Initiliaze distance matrix
    dist.fill(-1)
    for primary_id in range(n_primary):
        p = primary_nodes[primary_id]
        dist[primary_id, p] = 0

    # Initialize backtrack node matrix
    backtrack_node.fill(-1)

    # Initialize adjacency list
    for i in range(n_nodes):
        adj[i] = vector[int]()
    for i in range(n_edges):
        n1 = adjacency_list[i, 0]
        n2 = adjacency_list[i, 1]
        p1 = node_lookup[n1]
        p2 = node_lookup[n2]
        if p1[0] == 0 and p2[0] == 0:
            dist[p1[1], n2] = 1
            dist[p2[1], n1] = 1
        else:
            adj[n1].push_back(n2)
            adj[n2].push_back(n1)

    # Compute distance from primary nodes to all other nodes
    for p_primary_id in range(n_primary):
        p = primary_nodes[p_primary_id]
        to_visit1.clear()
        for m in adj[p]:
            to_visit1.push_back(m)
            backtrack_node[p_primary_id, m] = p
        t = True
        d = 1

        ### print(f'Compute distance from {p} to all other nodes')
        ### print(f'adj: {adj[p]}')
        ### print(f'dist: {dist[node_lookup[p][1]]}')

        while not (to_visit1 if t else to_visit2).empty():
            # Clear next to_visit
            if t: to_visit2.clear()
            else: to_visit1.clear()

            ### print(f'  [d={d}]: Nodes to visit {to_visit1 if t else to_visit2}')

            # Iterate over nodes to_visit
            for n in (to_visit1 if t else to_visit2):
                # Update distance
                dist[p_primary_id, n] = d

                # If n is a primary node, don't follow edges
                if node_lookup[n][0] == 0:
                    ### print(f'   {n} -> ... skipped')
                    continue

                # Otherwise Follow edges of node n
                for m in adj[n]:
                    # Skip if already visited or if m is p
                    if dist[p_primary_id, m] >= 0:
                        ### print(f'   {n} -> {m} skipped')
                        continue
                    # Update backtrack node
                    backtrack_node[p_primary_id, m] = n
                    ### print(f'   {n} -> {m}')
                    # Add m to next to_visit
                    if t: to_visit2.push_back(m)
                    else: to_visit1.push_back(m)

            # Switch to_visit and next to_visit
            t = not t
            # Increment distance
            d += 1

    # Free memory
    free(adj)
    free(node_lookup)

    return dist, backtrack_node