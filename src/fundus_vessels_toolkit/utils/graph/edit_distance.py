from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Generator, List, NamedTuple, Optional, Same, Set, Tuple, NamedTupleMeta

import numpy as np
import numpy.typing as npt


@dataclass
class GEDSearchContext:
    #: Upper bound on the cost of the optimal complete path
    UB: float

    #: Priority of nodes
    #: Stores the index of nodes from the two graphs in the order they should be processed
    #: If the value ``v`` is positive it correspond to the ``v-1``-th node of the first graph
    #: If the value ``v`` is negative it correspond to the ``-v-1``-th node of the second graph
    nodes_priority: npt.ArrayLike


########################################################################################################################
class BaseMappingValue(NamedTuple):
    pass


class NodeAssign(BaseMappingValue):
    node_id: int

    def __str__(self) -> str:
        return f"Node[{self.node_id}]"


class EdgeSplit(BaseMappingValue):
    edge_id: int
    split_pos: float

    def __str__(self) -> str:
        return f"Edge[{self.edge_id}]"


class UnassignedNode(BaseMappingValue):
    def __str__(self) -> str:
        return "Empty"


class NodeMapping:
    def __init__(self, m12: List[BaseMappingValue | None], m21: List[BaseMappingValue | None]):
        self.m12 = m12
        self.m21 = m21

    def copy(self) -> Same:
        return NodeMapping(self.m12.copy(), self.m21.copy())

    def pending_nodes(self) -> Tuple[Set[int], Set[int]]:
        return {i for i, m in enumerate(self.m12) if m is None}, {i for i, m in enumerate(self.m21) if m is None}

    def __setitem__(self, key: Tuple[bool, int], value: BaseMappingValue | None) -> None:
        assert isinstance(key, tuple) and len(key) == 2, "Invalid key"
        from_graph1, node_id = key
        if from_graph1:
            self.m12[node_id] = value
            if isinstance(value, NodeAssign) and self.m21[value.node_id] is None:
                self.m21[value.node_id] = NodeAssign(node_id)
        else:
            self.m21[node_id] = value
            if isinstance(value, NodeAssign) and self.m12[value.node_id] is None:
                self.m12[value.node_id] = NodeAssign(node_id)

    def __getitem__(self, key: Tuple[bool, int]) -> BaseMappingValue | None:
        assert isinstance(key, tuple) and len(key) == 2, "Invalid key"
        return self.m12[key[1]] if key[0] else self.m21[key[1]]


########################################################################################################################
class EditPath(NamedTuple):
    node_map: NodeMapping
    cost: float
    parent: Optional[Same] = None

    def is_complete(self) -> bool:
        return all(len(_) == 0 for _ in self.node_map.pending_nodes())


########################################################################################################################
def children(path: EditPath, ctx: GEDSearchContext) -> List[Tuple[EditPath, float]]:
    # === FETCH PENDING NODES WITH HIGHEST PRIORITY ===
    pending_nodes1, pending_nodes2 = path.node_map.pending_nodes()
    N1, N2 = len(pending_nodes1), len(pending_nodes2)
    highest_pending_nodes1 = np.argmax(np.isin(ctx.nodes_priority, pending_nodes1 + 1))
    highest_pending_nodes2 = np.argmax(np.isin(ctx.nodes_priority, -pending_nodes2 - 1))
    if highest_pending_nodes1 < highest_pending_nodes2:
        node_from_G1 = True
        node = abs(ctx.nodes_priority[highest_pending_nodes1]) - 1
        opposite_pending_nodes = pending_nodes2
    else:
        node_from_G1 = False
        node = abs(ctx.nodes_priority[highest_pending_nodes2]) - 1
        opposite_pending_nodes = pending_nodes1

    # === GENERATE CHILDREN ===
    children = []

    # 1. Generate all possible substitutions with a pending node from the opposite graph
    for opposite_node in opposite_pending_nodes:
        new_node_map = path.node_map.copy()
        new_node_map[node_from_G1, node] = NodeAssign(opposite_node)
        cost = ...
        children.append(EditPath(new_node_map, path.cost + cost, path))

    # 2. Generate all possible merge with an already assigned node from the opposite graph
    for opposite_node in ...:
        new_node_map = path.node_map.copy()
        new_node_map[node_from_G1, node] = NodeAssign(opposite_node)
        cost = ...
        children.append(EditPath(new_node_map, path.cost + cost, path))

    # 3. Insert the current node to the opposite graph
    new_node_map = path.node_map.copy()
    new_node_map[node_from_G1, node] = UnassignedNode()
    cost = ...
    children.append(EditPath(new_node_map, path.cost + cost, path))

    # 4. Generate all possible splits with pending branches from the opposite graph
    for opposite_branch in ...:
        new_node_map = path.node_map.copy()
        new_node_map[node_from_G1, node] = EdgeSplit(opposite_node)
        cost = ...
        children.append(EditPath(new_node_map, path.cost + cost, path))

    # === COMPUTE ESTIMATED REMAINING COST AND TRIM CHILDREN ===
    children_with_cost = []
    for child in children:
        total_estimated_cost = child.cost + ...
        if child.total_estimated_cost < ctx.UB:
            children_with_cost.append((child, total_estimated_cost))

    # === SORT CHILDREN BY DECREASING COST ===
    return sorted(children_with_cost, key=lambda x: x[1], reverse=True)


########################################################################################################################
