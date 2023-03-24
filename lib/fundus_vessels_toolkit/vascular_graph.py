from torch_geometric.data import Data
import torch
import kornia as K

from .seg2graph import torch_medial_axis


class VascularGraph(Data):
    def __init__(self, x, edge_index, edge_attr, y=None, pos=None, vessels_label=None, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, **kwargs)

    @staticmethod
    def from_vessels_map(vessel_map: torch.Tensor, remove_small_vessels=True):
        # Skeletonize the vessels map
        skel_map = torch_medial_axis(vessel_map).float()

        batched = vessel_map.ndim == 3
        if not batched:
            skel_map = skel_map.unsqueeze(0)
            vessel_map = vessel_map.unsqueeze(0)

        # Detect bifurcation points


        vessels_label = K.contrib.connected_components(skel_map.unsqueeze(1)).squeeze(1)

        if remove_small_vessels is True:
            remove_small_vessels = 15
            def remove_small_branches():
                for i in range(vessels_label.max().item()):
                    if (vessels_label == i).sum() < remove_small_vessels:
                        vessels_label[vessels_label == i] = 0


_bifurcations_signatures = torch.tensor([
    # 3 vessels
    [[1, 1, 1],
     [0, 1, 0],
     [0, 0, 0]],

])