#distutils: language=c++
#cython: language_level=3, wraparound=False, boundscheck=False, cdivision=True, nonecheck=False

cimport cython
from cython.parallel import prange
from cython.operator import dereference as deref, preincrement as inc
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool
from libcpp.list cimport list as cpplist
from libcpp.stack cimport stack as cppstack
from libcpp.vector cimport vector

ctypedef np.npy_int INT
ctypedef np.npy_uint8 UINT8
ctypedef np.npy_bool BOOL

cdef struct Neighbor:
    int y
    int x
    int id


cdef struct Point:
    int y
    int x


def label_skeleton(np.ndarray[UINT8, ndim=2, cast=True] skeleton):
    """
    Label connected components algorithm for skeleton images.
    The algorithm follow the same principle as the standard connected component labeling algorithm,
    but with two additional rules:
        - Branch labels are not propagated through junctions
        - All neighbors pixels of a  must be labeled with different labels.

    Parameters
    ----------
    skeleton : np.ndarray[UINT8, ndim=2, cast=True]
        Skeleton image to label. Must be a uint8 image with 0 for background and positive values for skeleton pixels.
        The algorithm assumes that endpoints are encoded 1, branches 2 and junctions 3+. 
            (Encoding the connectivity of the pixel to its neighbors.)
    
    Returns
    -------
    np.ndarray[UINT32, ndim=2]
        Branch labels map. Each branch is labeled with a unique positive integer. Background pixels are labeled 0.
    
    np.ndarray[INT32, ndim=2]
    """
    cdef:
        int H = skeleton.shape[0]
        int W = skeleton.shape[1]

        np.ndarray[np.npy_int32, ndim=2] branch_label = np.zeros((H, W), dtype=np.int32)
        np.ndarray[np.npy_int32, ndim=2] branch_adj_list
        np.ndarray[np.npy_uint32, ndim=2] np_nodes_yx
        cpplist[Point] nodes_yx

        int i_node = 0 
        int i_branch = 0
        Point yx
        int x, y, l, skel, b
        int n_skel, n_branch_label, min_n_branch_label, node_id, branch_count
        Neighbor n
        cpplist[Neighbor] neighbors, neighbor_nodes
        cpplist[cpplist[int]] branch_equivalences
        vector[cpplist[int]] node_adj_list
        cpplist[int] eq
        vector[int] lookup

    node_adj_list.reserve(512)

    # First pass: label junctions and endpoints
    for y in range(H):
        for x in range(W):
            skel = skeleton[y, x]
            if skel:
                if skel != 1:
                    # If current pixel is a node (junction or endpoint), ensure it's labeled
                    if branch_label[y, x] == 0:
                        i_node += 1
                        branch_label[y, x] = -i_node
                        nodes_yx.push_back(Point(y, x))
                        node_adj_list.push_back(cpplist[int]())
                    continue

                # Otherwise if its a branch: clear temporary variables...
                min_n_branch_label = 0
                neighbor_nodes.clear()

                # ... then search for neighbor nodes ...
                neighbors = get_neighbors(y, x, H, W)
                for n in neighbors:
                    n_skel = skeleton[n.y, n.x]
                    if n_skel > 1:
                        # If a node is found add it to the list of neighbors nodes.
                        neighbor_nodes.push_back(n)
                        # and ensure it's properly labeled (to remember the connection to the current branch)
                        if branch_label[n.y, n.x] >= 0:
                            i_node += 1
                            branch_label[n.y, n.x] = -i_node
                            nodes_yx.push_back(Point(n.y, n.x))
                            node_adj_list.push_back(cpplist[int]())

                # Remove node close neigbors from the list of neighbors
                for n in neighbor_nodes:
                    drop_close_neighbors_sorted(neighbors, n.id)
                
                # ... then search for neighbor branches ...
                for n in neighbors:
                    n_skel = skeleton[n.y, n.x]
                    if n_skel == 1:
                        # If a branch is found...
                        n_branch_label = branch_label[n.y, n.x]
                        if n_branch_label > 0:
                            # ... and is already labeled ...
                            if min_n_branch_label == 0:
                                # ... and is the first neighbor to be found, set it as the minimum label, ...
                                min_n_branch_label = n_branch_label
                            elif min_n_branch_label != n_branch_label:
                                # ... otherwise remember the equivalence for the second pass. 
                                # (See connected component labeling algorithm for more details)
                                eq = cpplist[int]()
                                eq.push_back(min_n_branch_label)
                                eq.push_back(n_branch_label)
                                branch_equivalences.push_back(eq)
                                if min_n_branch_label > n_branch_label:
                                    min_n_branch_label =  n_branch_label
                            

                # Then when all neighbors have been processed, label the current branch.
                if min_n_branch_label == 0:
                    i_branch += 1
                    min_n_branch_label = i_branch
                branch_label[y, x] = min_n_branch_label

                # Finally remember link between branch and node labels
                if not neighbor_nodes.empty():
                    for n in neighbor_nodes:
                        node_id = -branch_label[n.y, n.x] - 1
                        node_adj_list[node_id].push_back(min_n_branch_label)

                    

    # Second pass: solve branch equivalences
    lookup = vector[int](i_branch + 1)
    branch_count = equivalence2lookup(branch_equivalences, lookup)
    for y in range(H):
        for x in range(W):
            l = branch_label[y, x]
            if l > 0:
                branch_label[y, x] = lookup[l]

    # Convert node adjacencies list to branch adjacencies list
    branch_adj_list = np.empty((branch_count-1, 2), dtype=np.int32)
    branch_adj_list.fill(-1)
    i_node = 0
    for eq in node_adj_list:
        for b in eq:
            b = lookup[b]-1
            if branch_adj_list[b, 0] == -1:
                branch_adj_list[b, 0] = i_node
            else:
                branch_adj_list[b, 1] = i_node
        i_node += 1

    # Cast nodes_yx to numpy array
    i_node = 0
    np_nodes_yx = np.empty((len(nodes_yx), 2), dtype=np.uint32)
    for yx in nodes_yx:
        np_nodes_yx[i_node, 0] = yx.y
        np_nodes_yx[i_node, 1] = yx.x
        i_node += 1

    return branch_label, branch_adj_list, np_nodes_yx
        

cdef int equivalence2lookup(cpplist[cpplist[int]]& equivalences, vector[int]& lookup):
    cdef: 
        int N = lookup.size() 
        cppstack[int] stack
        int node, current_node, neighbor, nb_label
        vector[cpplist[int]] adj_list = equivalence2adj_list(equivalences, N)

    for node in range(N):
        lookup[node] = 0

    nb_label = 0
    for current_node in range(N):
        if lookup[current_node] != 0:
            # If the node is already labeled, skip it
            continue

        # Initialize the stack with the current node
        stack.push(current_node)

        # Then perform depth first search
        while not stack.empty():
            # Pop the top node from the stack
            current_node = stack.top()
            stack.pop()

            # Label the node and push its unlabeled neighbors to the stack
            lookup[current_node] = nb_label
            for neighbor in adj_list[current_node]:
                if lookup[neighbor] == 0:
                    stack.push(neighbor)
        nb_label += 1
    return nb_label


def solve_clusters(list[tuple[int]] equivalences) -> list[tuple[int]]:
    """
    Find clusters of connected nodes from a graph described as a list of equivalences (connected nodes). 

    Parameters
    ----------
        equivalences (list[tuple[int]]): List of equivalences (connected nodes).
    
    Returns
    -------
        list[tuple[int]]: List of clusters of connected nodes.
    """
    cdef: 
        cpplist[cpplist[int]] cy_equivalences
        vector[cpplist[int]] adj_list
        list[tuple[int]] clusters = []
        cpplist[int] cluster
        vector[bool] visited
        cppstack[int] stack
        int i, j, k, N

    N = -1
    for eq in equivalences:
        cy_eq = cpplist[int]()
        for i in eq:
            if i > N:
                N = i
            cy_eq.push_back(i)
        cy_equivalences.push_back(cy_eq)
    N += 1

    adj_list = equivalence2adj_list(cy_equivalences, N)

    visited = vector[bool](N)
    for i in range(N):
        visited[i] = False

    # Traverse graph (DFS) grouping connected nodes into cluster
    for i in range(N):
        cluster = cpplist[int]()
        if not visited[i]:
            stack.push(i)
            while not stack.empty():
                j = stack.top()
                stack.pop()
                cluster.push_back(j)
                visited[j] = True
                for k in adj_list[j]:
                    if not visited[k]:
                        stack.push(k)
        clusters.append(tuple(cluster))
    return clusters


cdef vector[cpplist[int]] equivalence2adj_list(cpplist[cpplist[int]]& equivalences, int max_i):
    cdef: 
        vector[cpplist[int]] adj_list = vector[cpplist[int]](max_i+1)
        cpplist[int].iterator it1, it2

    # Build adjacency list
    for eq in equivalences:
        it1 = eq.begin()
        inc(it1)
        while it1 != eq.end():
            it2 = eq.begin()
            while it2 != it1:
                adj_list[deref(it1)].push_back(deref(it2))
                adj_list[deref(it2)].push_back(deref(it1))
                inc(it2)
            inc(it1)
    
    return adj_list


cdef cpplist[Neighbor] get_neighbors(int y, int x, int h, int w):
    cdef cpplist[Neighbor] neighbors_list
    if y > 0:
        if x > 0:
            neighbors_list.push_back(Neighbor(y-1, x-1, 0))
        neighbors_list.push_back(Neighbor(y-1, x, 1))
        if x < w-1:
            neighbors_list.push_back(Neighbor(y-1, x+1, 2))
    if x < w-1:
        neighbors_list.push_back(Neighbor(y, x+1, 3))
    if y < h-1:
        if x < w-1:
            neighbors_list.push_back(Neighbor(y+1, x+1, 4))
        neighbors_list.push_back(Neighbor(y+1, x, 5))
        if x > 0:
            neighbors_list.push_back(Neighbor(y+1, x-1, 6))
    if x > 0:
        neighbors_list.push_back(Neighbor(y, x-1, 7))

    return neighbors_list


cdef void drop_close_neighbors_sorted(cpplist[Neighbor]& neigbors, int neighbor_id):
    drop_neighbors_sorted(neigbors, CLOSE_NEIGHBORS[neighbor_id])


cdef void drop_neighbors_sorted(cpplist[Neighbor]& neighbors, cpplist[int] select_ids):
    cdef Neighbor n
    cdef cpplist[Neighbor].iterator it_neighbor = neighbors.begin()
    cdef cpplist[int].iterator it_selected = select_ids.begin()  
    
    while it_neighbor != neighbors.end():
        n = deref(it_neighbor)
        while n.id > deref(it_selected):
            inc(it_selected)
            if it_selected == select_ids.end():
                return

        if n.id == deref(it_selected):
            it_neighbor = neighbors.erase(it_neighbor)
            inc(it_selected)
            if it_selected == select_ids.end():
                return
        else:
            inc(it_neighbor)


cdef vector[cpplist[int]] CLOSE_NEIGHBORS = vector[cpplist[int]](8)
CLOSE_NEIGHBORS[0].push_back(0)
CLOSE_NEIGHBORS[0].push_back(1)
CLOSE_NEIGHBORS[0].push_back(7)

CLOSE_NEIGHBORS[1].push_back(0)
CLOSE_NEIGHBORS[1].push_back(1)
CLOSE_NEIGHBORS[1].push_back(2)
CLOSE_NEIGHBORS[1].push_back(3)
CLOSE_NEIGHBORS[1].push_back(7)

CLOSE_NEIGHBORS[2].push_back(1)
CLOSE_NEIGHBORS[2].push_back(2)
CLOSE_NEIGHBORS[2].push_back(3)

CLOSE_NEIGHBORS[3].push_back(1)
CLOSE_NEIGHBORS[3].push_back(2)
CLOSE_NEIGHBORS[3].push_back(3)
CLOSE_NEIGHBORS[3].push_back(4)
CLOSE_NEIGHBORS[3].push_back(5)

CLOSE_NEIGHBORS[4].push_back(3)
CLOSE_NEIGHBORS[4].push_back(4)
CLOSE_NEIGHBORS[4].push_back(5)

CLOSE_NEIGHBORS[5].push_back(3)
CLOSE_NEIGHBORS[5].push_back(4)
CLOSE_NEIGHBORS[5].push_back(5)
CLOSE_NEIGHBORS[5].push_back(6)
CLOSE_NEIGHBORS[5].push_back(7)

CLOSE_NEIGHBORS[6].push_back(5)
CLOSE_NEIGHBORS[6].push_back(6)
CLOSE_NEIGHBORS[6].push_back(7)

CLOSE_NEIGHBORS[7].push_back(0)
CLOSE_NEIGHBORS[7].push_back(1)
CLOSE_NEIGHBORS[7].push_back(5)
CLOSE_NEIGHBORS[7].push_back(6)
CLOSE_NEIGHBORS[7].push_back(7)
