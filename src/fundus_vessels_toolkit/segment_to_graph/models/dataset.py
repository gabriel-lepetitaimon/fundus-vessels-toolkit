from pathlib import Path
from typing import Self

import torch
import torch_geometric as pyg
import tqdm
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset

from fundus_toolkits import FundusData
from fundus_toolkits.utils.image import read_image
from fundus_vessels_toolkit.vascular_data_objects.vbranch_geodata import VBranchGeoData
from fundus_vessels_toolkit.vascular_data_objects.vtree import VTree

from ..vbranch_digraph import TreeTopology, VBranchDigraph, VGraph


class VBranchDigraphData(PygData):
    def __init__(
        self,
        edge_index: torch.Tensor,
        edge_first_tip: torch.Tensor,
        fundus_img: torch.Tensor,
        branch_curves: list[torch.Tensor],
        edge_p: torch.Tensor,
        branch_av_p: torch.Tensor,
        branch_dir: torch.Tensor,
    ):
        """Store branch digraph data in PyG format.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edges list as a 2xE tensor, storing the indices of the source and target branches.
        edge_first_tip : torch.Tensor
            A 2xE boolean tensor storing whether the edges connect the first or second tip of the source and target branches.
        fundus_img : torch.Tensor
            The fundus image tensor as a 3xHxW tensor.
        branch_curves : list[torch.Tensor]
            List of branch curves, each as a Nx2 tensor of (x, y) coordinates.
        edge_p : torch.Tensor
            Probability of each edge being correct as a tensor of shape (E,).
        branch_av_p : torch.Tensor
            Probability of each branch being an artery or a vein (or neither) as a tensor of shape (B, 2).
        branch_dir : torch.Tensor
            Direction of each branch as a tensor of shape (B, 2).
        """  # noqa: E501
        if edge_index is not None:
            assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
                f"edge_index must be of shape (2, E) but got {edge_index.shape}"
            )
            E = edge_index.shape[1]
            assert edge_first_tip.shape == (E, 2), (
                f"edge_first_tip must be of shape (E, 2) but got {edge_first_tip.shape}"
            )
            assert edge_first_tip.dtype == torch.bool, "edge_first_tip must be a boolean tensor"
            assert fundus_img.ndim == 3 and fundus_img.shape[0] == 3, (
                f"fundus_img must be of shape (3, H, W) but got {fundus_img.shape}"
            )
            B = len(branch_curves)
            assert branch_av_p.shape == (B, 2), f"branch_av_p must be of shape (B, 2) but got {branch_av_p.shape}"
            assert branch_dir.shape == (B,), f"branch_dir must be of shape (B,) but got {branch_dir.shape}"
            pos = torch.stack([curve[curve.shape[0] // 2] for curve in branch_curves], dim=0)
        else:
            B = 0
            pos = None
        super().__init__(
            edge_index=edge_index,
            edge_first_tip=edge_first_tip,
            fundus_img=fundus_img,
            branch_curves=branch_curves,
            edge_p=edge_p,
            branch_av_p=branch_av_p,
            branch_dir=branch_dir,
            pos=pos,  # Use mid-point as node
        )
        self.num_nodes = B

    def is_node_attr(self, key: str) -> bool:
        return super().is_node_attr(key) or key in {"branch_curves", "branch_av_p", "branch_dir"}

    def is_edge_attr(self, key: str) -> bool:
        return super().is_edge_attr(key) or key in {"edge_first_tip", "edge_p"}

    @classmethod
    def from_branch_digraph(cls, branch_digraph: VBranchDigraph, fundus_path: Path) -> Self:
        # === Load fundus image ===
        fundus_image = torch.tensor(read_image(fundus_path, cast_to_float=True))

        # === Extract branch digraph data ===
        line_list = branch_digraph.line_list
        edge_index = torch.stack([torch.from_numpy(line_list[:, 0]), torch.from_numpy(line_list[:, 2])], dim=0)
        edge_first_tip = torch.stack([torch.from_numpy(line_list[:, 1]), torch.from_numpy(line_list[:, 3])], dim=1) == 0

        geodata = branch_digraph.graph.geometric_data()
        branch_curves = [torch.from_numpy(curve).float() for curve in geodata.branch_curve(fill_with_nodes=True)]
        edge_p = torch.from_numpy(branch_digraph.line_p).float()
        branch_av_p = torch.from_numpy(branch_digraph.branch_av_p).float()
        branch_dir = torch.from_numpy(branch_digraph.branch_dir_p).float()

        return cls(
            edge_index=edge_index,
            edge_first_tip=edge_first_tip,
            fundus_img=fundus_image,
            branch_curves=branch_curves,
            edge_p=edge_p,
            branch_av_p=branch_av_p,
            branch_dir=branch_dir,
        )


class VBranchDigraphDataset(PygDataset):
    def __init__(
        self,
        fundus_paths: list[Path],
        graphs: list[VGraph],
        target_topologies: list[tuple[TreeTopology, TreeTopology]],
    ):
        super().__init__()
        assert len(fundus_paths) == len(graphs) == len(target_topologies), "All input lists must have the same length"
        self.fundus_paths = fundus_paths
        self.target_topologies = target_topologies
        self.graphs = graphs

    def len(self):
        return len(self.fundus_paths)

    def get(self, idx) -> VBranchDigraphData:
        fundus_path = self.fundus_paths[idx]

        # === Augment graph ===
        graph = self.graphs[idx]
        # TODO: Add data augmentation here

        # === Compute BranchDigraph ===
        branch_digraph = VBranchDigraph.from_graph(graph)
        branch_digraph.compute_p_from_gt(*self.target_topologies[idx], check=False)

        return VBranchDigraphData.from_branch_digraph(branch_digraph, fundus_path)

    @classmethod
    def load_from_dirs(
        cls,
        fundus_dir: Path,
        target_topology_dir: Path,
        graph_dir: Path | None = None,
        av_dir: Path | None = None,
        *,
        fundus_ext=".png",
        av_ext=".png",
        verbose: bool = True,
    ) -> Self:
        fundus_paths = fundus_dir.glob(f"*{fundus_ext}")
        if graph_dir is not None:
            graphs = graph_dir.glob("*.npz")
        elif av_dir is not None:
            graphs = av_dir.glob(f"*{av_ext}")
        else:
            raise ValueError("Either graph_dir or av_dir must be provided")
        target_topo_art = target_topology_dir.glob("*_art.npz")
        target_topo_vei = target_topology_dir.glob("*_vei.npz")

        filenames = sorted(
            {p.stem for p in fundus_paths}
            & {g.stem for g in graphs}
            & {t.stem[:-4] for t in target_topo_art}
            & {t.stem[:-4] for t in target_topo_vei}
        )
        if verbose:
            print(f"Loading {len(filenames)} branch digraphs...")

        fundus_paths = [fundus_dir / f"{name}{fundus_ext}" for name in filenames]

        if graph_dir is not None:
            graphs = [VGraph.load(graph_dir / f"{name}.npz") for name in filenames]
        else:
            from ...pipelines.avseg_to_tree import GNNAVSegToTree

            assert av_dir is not None
            av2tree = GNNAVSegToTree()

            graphs = [
                av2tree.to_vgraph(FundusData(av=av_dir / f"{file}{av_ext}"))
                for file in tqdm.tqdm(filenames, desc="Parsing input graphs...", disable=not verbose)
            ]

        for graph in graphs:
            graph.geometric_data().clear_attribute(
                all_except={VBranchGeoData.Fields.TANGENTS, VBranchGeoData.Fields.TIPS_TANGENT}
            )

        def load_topology(filename: str) -> TreeTopology:
            tree = VTree.load(target_topology_dir / filename)
            return TreeTopology.from_tree(tree, sparse=True, discard_tree=True)

        target_topologies = [
            (load_topology(f"{name}_art.npz"), load_topology(f"{name}_vei.npz"))
            for name in tqdm.tqdm(filenames, desc="Rasterizing target topologies...", disable=not verbose)
        ]

        return cls(fundus_paths, graphs, target_topologies)
