from __future__ import annotations

import copy
import hashlib
import math
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Self, Sequence, TypeGuard

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import tqdm
from fundus_data_toolkit.functional import open_image
from joblib import Parallel, delayed
from jppype import Mosaic
from numpy.random import MT19937, RandomState, SeedSequence
from torch import Tensor
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset

from fundus_odmac_toolkit.models.segmentation import segment
from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.utils.color import color_jitter
from fundus_toolkits.utils.geometric import Point, Rect
from fundus_toolkits.utils.image import read_image
from fundus_vessels_toolkit.segment_to_graph.av_tree_parsing import naive_infer_arborescence
from fundus_vessels_toolkit.utils.tree import tree_connected_components

from ...pipelines.avseg_to_tree import GNNAVSegToTree
from ...utils import if_none
from ...utils.data_io import most_common_image_ext
from ...utils.fundus_projections import ResizeTranslateProjection
from ...utils.jppype import AV_COLORS, draw_graph, draw_tree, draw_trees
from ...utils.numpy import np_group_by
from ...utils.typing import Bool1DArray, Int1DArray
from ...vascular_data_objects import VBranchGeoData, VTree
from ..graph_simplification import merge_nodes_by_distance
from ..tree_topology import TopologicalLabel
from ..vbranch_digraph import BaseEdgeAttrExtractor, EdgeAttrExtractor, TreeTopology, VBranchDigraph, VGraph
from .data_augmentation import deteriorate_graph, geometric_augment


class VBranchDigraphData(PygData):
    img: Tensor
    """Fundus image as a tensor of shape (3, H, W)."""

    od_yx: Tensor
    """(y, x) coordinates of the optic disc center as a tensor of shape (2,)."""

    mac_yx: Tensor
    """(y, x) coordinates of the macula center as a tensor of shape (2,)."""

    vnode_count: int
    """Number of nodes in the vascular graph."""

    branch_nodes: Tensor
    """Indices of the two tip nodes of each branch in the graph, as a tensor of shape (B, 2)."""

    branch_curves: Tensor
    """List of the branch curves, each as a Nx2 tensor of (x, y) coordinates."""

    branch_root_candidates: Tensor
    """Valid vascular roots as a boolean tensor of shape (B, 2) indicating for each branch tip whether it is a valid root."""  # noqa: E501

    edge_index: Tensor  # type: ignore[assignment]+
    """Edges list as a (2, E) tensor, storing the indices of the source and target branches."""

    edge_dir: Tensor
    """The required direction of the source and target branches for each edge as a boolean tensor of shape (E, 2). For each edge, the first column indicates whether the source branch should be oriented from its first tip to its second tip (True) or the contrary (False), and the second column indicates the same for the target branch."""  # noqa: E501

    edge_attr: Tensor | None
    """Edge attributes as a tensor of shape (E, F_e)."""

    edge_p: Tensor | None
    """Ground truth probability of each edge being correct as a tensor of shape (E,)."""  # noqa: E501

    branch_root_p: Tensor | None
    """Ground truth probability of each branch tip being a vascular root as a tensor of shape (B,2)."""  # noqa: E501

    branch_fp_p: Tensor | None
    """Ground truth probability of each branch being a false positive from the segmentation as a tensor of shape (B,)."""  # noqa: E501

    branch_av_p: Tensor | None
    """Ground truth probability of each branch being an artery (as opposed to a vein) as a tensor of shape (B,)."""  # noqa: E501

    branch_dir_p: Tensor | None
    """Ground truth probability of each branch being oriented from their first tip to their second tip (>=0.5) or the contrary (<0.5) as a tensor of shape (B,)."""  # noqa: E501

    branch_subtree_idx: Tensor | None
    """Subtree index of each branch accordingly to the ground truth topology, as a tensor of shape (B,)."""

    name: str
    """Name of the sample, usually the original fundus image file name without extension. (For debug and logging purposes)."""  # noqa: E501

    BRANCH_ATTR = {
        "branch_nodes",
        "branch_curves",
        "branch_root_candidates",
        "branch_root_p",
        "branch_fp_p",
        "branch_av_p",
        "branch_dir_p",
        "branch_subtree_idx",
    }

    EDGE_ATTR = {"edge_dir", "edge_p"}

    def __init__(
        self,
        img: Tensor = None,  # type: ignore
        od_yx: Tensor = None,  # type: ignore
        mac_yx: Tensor = None,  # type: ignore
        vnode_count: int = None,  # type: ignore
        branch_nodes: Tensor = None,  # type: ignore
        branch_curves: list[Tensor] = None,  # type: ignore
        branch_root_candidates: Tensor = None,  # type: ignore
        edge_index: Tensor = None,  # type: ignore
        edge_dir: Tensor = None,  # type: ignore
        edge_attr: Optional[Tensor] = None,
        edge_p: Optional[Tensor] = None,
        branch_root_p: Optional[Tensor] = None,
        branch_fp_p: Optional[Tensor] = None,
        branch_av_p: Optional[Tensor] = None,
        branch_dir_p: Optional[Tensor] = None,
        branch_subtree_idx: Optional[Tensor] = None,
        name: str = "",
    ):
        """Store branch digraph data in PyG format.

        Parameters
        ----------
        img : Tensor
            The fundus image tensor as a 3xHxW tensor.
        od_yx : Tensor
            The (y, x) coordinates of the optic disc center as a tensor of shape (2,).
        mac_yx : Tensor
            The (y, x) coordinates of the macula center as a tensor of shape (2,).
        node_count: int
            Number of nodes in the vascular graph.
        branch_list: Tensor
            Indices of the tips nodes of each branch as a tensor of shape (B, 2), where B is the number of branches.
        branch_curves : list[Tensor]
            The coordinates of the branches curve, as a list of B tensors containing (y, x) coordinates.
        branch_root_candidates: Tensor
            Valid vascular roots as a boolean tensor of shape (B, 2) indicating for each branch tip whether it is a valid root.
        edge_index : Tensor
            Edges list as a (2, E) tensor, storing the indices of the source and target branches.
        edge_dir : Tensor
            The required direction of the source and target branches for each edge as a boolean tensor of shape (E, 2). For each edge, the first column indicates whether the source branch should be oriented from its first tip to its second tip (True) or the contrary (False), and the second column indicates the same for the target branch.
        edge_attr : Tensor, optional
            Edge attributes as a tensor of shape (E, F_e).
        edge_p : Tensor, optional
            Ground truth probability of each edge being correct as a tensor of shape (E,).
        branch_root_p : Tensor, optional
            Ground truth probability of each branch tip being a vascular root as a tensor of shape (B,2).
        branch_fp_p : Tensor, optional
            Ground truth probability of each branch being a false positive from the segmentation as a tensor of shape (B,).
        branch_av_p : Tensor, optional
            Ground truth probability of each branch being an artery (as opposed to a vein) as a tensor of shape (B,).
        branch_dir_p : Tensor, optional
            Ground truth probability of each branch being oriented from their first tip to their second tip (>=0.5) or the contrary (<0.5) as a tensor of shape (B,).
        branch_subtree_idx : Tensor, optional
            Subtree index of each branch accordingly to the ground truth topology, as a tensor of shape (B,).
        """  # noqa: E501
        if img is not None:
            # === DATA INTEGRITY CHECKS ===
            # --- Global fields ---
            assert img.ndim == 3 and img.shape[0] == 3, f"fundus_img must be of shape (3, H, W) but got {img.shape}"
            assert od_yx.shape == (2,), f"od_yx must be of shape (2,) but got {od_yx.shape}"
            assert mac_yx.shape == (2,), f"mac_yx must be of shape (2,) but got {mac_yx.shape}"
            assert isinstance(vnode_count, int) and vnode_count >= 0, (
                f"node_count must be a non-negative integer but got {vnode_count}"
            )

            # --- Branch attributes ---
            B = len(branch_curves)
            assert all(curve.ndim == 2 and curve.shape[1] == 2 for curve in branch_curves), (
                "Each branch curve must be a Nx2 tensor of (y, x) coordinates"
            )
            assert branch_nodes.shape == (B, 2), f"branch_list must be of shape (B, 2) but got {branch_nodes.shape}"
            assert branch_root_candidates.shape == (B, 2), (
                f"branch_root_candidates must be of shape (B, 2) but got {branch_root_candidates.shape}"
            )
            assert branch_root_p is None or branch_root_p.shape == (B, 2), (
                f"branch_root_p must be of shape (B, 2) but got {branch_root_p.shape}"
            )
            assert branch_fp_p is None or branch_fp_p.shape == (B,), (
                f"branch_fp_p must be of shape (B,) but got {branch_fp_p.shape}"
            )
            assert branch_av_p is None or branch_av_p.shape == (B,), (
                f"branch_av_p must be of shape (B,) but got {branch_av_p.shape}"
            )
            assert branch_dir_p is None or branch_dir_p.shape == (B,), (
                f"branch_dir_p must be of shape (B,) but got {branch_dir_p.shape}"
            )
            assert branch_subtree_idx is None or branch_subtree_idx.shape == (B,), (
                f"branch_subtree_idx must be of shape (B,) but got {branch_subtree_idx.shape}"
            )

            # --- Edge attributes ---
            assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
                f"edge_index must be of shape (2, E) but got {edge_index.shape}"
            )
            E = edge_index.shape[1]
            assert edge_dir.shape == (E, 2), f"edge_dir must be of shape (E, 2) but got {edge_dir.shape}"
            assert edge_dir.dtype == torch.bool, "edge_dir must be a boolean tensor"
            assert edge_attr is None or (edge_attr.ndim == 2 and edge_attr.shape[0] == E), (
                f"edge_attr must be of shape (E, F_e) but got {edge_attr.shape}"
            )
            assert edge_p is None or edge_p.shape == (E,), f"edge_p must be of shape (E,) but got {edge_p.shape}"

            # === PREPROCESSING ===
            curves_ = torch.full((B, 20, 2), -1.0)
            pos = []
            for i, curve in enumerate(branch_curves):
                C = curve.shape[0]
                pos.append(curve[C // 2])
                if C == 1:
                    curves_[i, :] = curve[0]
                elif C <= 20:
                    halfC = C // 2
                    repeat = math.ceil(10 / halfC)
                    curves_[i, :10] = torch.tile(curve[:halfC], (repeat, 1))[:10]
                    curves_[i, 10:] = torch.tile(curve[C - halfC :], (repeat, 1))[:10].flip(0)
                else:
                    indices = torch.linspace(0, C - 1, 20).long()
                    curves_[i] = curve[indices]
            pos = torch.stack(pos, dim=0)
        else:
            B = 0
            pos = None
            curves_ = None

        super().__init__(
            edge_index=edge_index,
            edge_dir=edge_dir,
            edge_attr=edge_attr,
            branch_nodes=branch_nodes,
            branch_curves=curves_,
            edge_p=edge_p,
            branch_root_candidates=branch_root_candidates,
            branch_root_p=branch_root_p,
            branch_fp_p=branch_fp_p,
            branch_av_p=branch_av_p,
            branch_dir_p=branch_dir_p,
            branch_subtree_idx=branch_subtree_idx,
            pos=pos,  # Use mid-point as node
            img=img,
            od_yx=od_yx,
            mac_yx=mac_yx,
            vnode_count=vnode_count,
            name=name,
        )
        self.num_nodes = B

    def is_node_attr(self, key: str) -> bool:
        return super().is_node_attr(key) or key in self.BRANCH_ATTR

    def is_edge_attr(self, key: str) -> bool:
        return super().is_edge_attr(key) or key in self.EDGE_ATTR

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "edge_index":
            return self.num_nodes
        elif key == "branch_nodes":
            return self.vnode_count
        else:
            return 0

    @classmethod
    def from_branch_digraph(
        cls,
        digraph: VBranchDigraph,
        fundus_img: Tensor,
        od_yx: npt.NDArray,
        mac_yx: npt.NDArray,
        name: str,
        edge_attr_fn: Optional[BaseEdgeAttrExtractor] = None,
    ) -> Self:
        assert VBranchDigraph.has_all_p(digraph), "branch_digraph must have branch_fp_p and branch_av_p"
        assert digraph.graph is not None, "branch_digraph must have graph constructed"

        not_root = ~digraph.root_mask
        geodata = digraph.graph.geometric_data()
        branch_curves = [torch.from_numpy(curve).float() for curve in geodata.branch_curve(fill_with_nodes=True)]
        branch_subtree_idx = tree_connected_components(torch.from_numpy(digraph.max_parent()))

        valid_root_tips = np.zeros((digraph.branch_count, 2), dtype=np.bool_)
        valid_root_tips[digraph.b1[digraph.root_mask], digraph.b1_tip[digraph.root_mask]] = True
        root_p = np.zeros((digraph.branch_count, 2), dtype=np.float32)
        root_p[digraph.b1[digraph.root_mask], digraph.b1_tip[digraph.root_mask]] = digraph.line_p[digraph.root_mask]

        return cls(
            img=fundus_img,
            od_yx=torch.from_numpy(od_yx).float(),
            mac_yx=torch.from_numpy(mac_yx).float(),
            vnode_count=digraph.graph.node_count,
            branch_nodes=torch.from_numpy(digraph.graph.branch_list).int(),
            branch_curves=branch_curves,
            branch_root_candidates=torch.from_numpy(valid_root_tips),
            edge_index=torch.from_numpy(digraph.b0b1[not_root]).T,
            edge_dir=torch.from_numpy(digraph.b0b1_dir[not_root]),
            edge_attr=torch.from_numpy(edge_attr_fn(digraph)).float() if edge_attr_fn is not None else None,
            edge_p=torch.from_numpy(digraph.line_p[not_root]).float(),
            branch_fp_p=torch.from_numpy(digraph.branch_fp_p).float(),
            branch_av_p=torch.from_numpy(digraph.branch_av_p).float(),
            branch_dir_p=torch.from_numpy(digraph.branch_dir_p).float(),
            branch_subtree_idx=branch_subtree_idx.int(),
            branch_root_p=torch.from_numpy(root_p),
            name=name,
        )

    @classmethod
    def has_gt(cls, instance: Self) -> TypeGuard[VBranchDigraphDataWithGT]:
        """Check if the data instance has ground truth probabilities (i.e. if edge_p, branch_fp_p, branch_av_p and branch_dir_p are not None)."""  # noqa: E501
        return VBranchDigraphDataWithGT.check(instance)


class VBranchDigraphDataWithGT(VBranchDigraphData):
    """Utility class for typechecking to ensure that the data has ground truth probabilities."""

    edge_p: Tensor
    branch_root_p: Tensor
    branch_fp_p: Tensor
    branch_av_p: Tensor
    branch_dir_p: Tensor
    branch_subtree_idx: Tensor

    @classmethod
    def check(cls, inst: VBranchDigraphData) -> TypeGuard[Self]:
        return (
            inst.edge_p is not None
            and inst.branch_root_p is not None
            and inst.branch_fp_p is not None
            and inst.branch_av_p is not None
            and inst.branch_dir_p is not None
            and inst.branch_subtree_idx is not None
        )


class VBranchDigraphBatch(VBranchDigraphData):
    batch: Tensor  # type: ignore[assignment]
    """Tensor of shape (B,) containing for each branch the index of its graph in the batch."""

    batch_size: int
    """Number of graphs in the batch."""

    vnode_count: Tensor
    """Tensor of shape (batch_size,) containing the number of vascular nodes for each graph in the batch."""


class VBranchDigraphDataset(PygDataset):
    def __init__(
        self,
        root: str | Path,
        fundus_paths: list[Path],
        graphs_path: list[Path],
        target_topologies_path: list[tuple[Path, Path]],
        *,
        augment: bool = False,
        verbose: bool = True,
        resize_to: Optional[int] = None,
        transform=None,
        overwrite: Optional[bool] = None,
        graphs: Optional[list[VGraph]] = None,
        target_topologies: Optional[list[tuple[TreeTopology, TreeTopology]]] = None,
        processed_fundus_paths: Optional[list[Path]] = None,
        od_yx: Optional[npt.NDArray] = None,
        mac_yx: Optional[npt.NDArray] = None,
        line_p_smoothing: float = 0.0,
    ):
        assert len(fundus_paths) == len(graphs_path) == len(target_topologies_path), (
            "All input lists must have the same length"
        )
        self._raw_fundus_paths = fundus_paths
        self.resize_to = resize_to
        self.verbose = verbose
        self.augment = augment
        self.line_p_smoothing = line_p_smoothing

        self._graphs_path = graphs_path
        self._target_topologies_path = target_topologies_path
        self.overwrite = overwrite

        super().__init__(root=str(root), transform=transform, force_reload=overwrite is not False)
        if (
            graphs is None
            or target_topologies is None
            or processed_fundus_paths is None
            or od_yx is None
            or mac_yx is None
        ):
            self.fundus_paths, self.target_topologies, self.graphs, self.od_yx, self.mac_yx = self.preload_from_disk()
        else:
            self.fundus_paths = processed_fundus_paths
            self.graphs = graphs
            self.target_topologies = target_topologies
            self.od_yx = od_yx
            self.mac_yx = mac_yx

    @property
    def processed_file_names(self):
        return [str(Path("raw") / (raw_file.stem + ".jpg")) for raw_file in self._raw_fundus_paths]

    def raw_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "raw" / (self._raw_fundus_paths[idx].stem + ".jpg")

    def graph_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "graphs" / (self._raw_fundus_paths[idx].stem + ".npz")

    def art_topo_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "target-topo" / (self._raw_fundus_paths[idx].stem + "_art.npz")

    def vei_topo_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "target-topo" / (self._raw_fundus_paths[idx].stem + "_vei.npz")

    def process(self):
        av2tree = GNNAVSegToTree()

        if (od_mac_file := Path(self.processed_dir) / "od_mac.csv").exists() and self.overwrite is not True:
            od_mac_df = pd.read_csv(od_mac_file).set_index("name")
            if "Unnamed: 0" in od_mac_df.columns:
                od_mac_df.drop(columns=["Unnamed: 0"], inplace=True)
        else:
            od_mac_df = pd.DataFrame(columns=["name", "od_y", "od_x", "mac_y", "mac_x"]).set_index("name")

        to_process = [
            (graph_in, topo_in, raw_in, av2tree, self.processed_dir, self.resize_to)
            for i, (graph_in, topo_in, raw_in) in enumerate(
                zip(self._graphs_path, self._target_topologies_path, self._raw_fundus_paths, strict=True)
            )
            if self.overwrite is True
            or not (raw_out := self.raw_processed_file(i)).exists()
            or not (graph_out := self.graph_processed_file(i)).exists()
            or not (art_out := self.art_topo_processed_file(i)).exists()
            or not (vei_out := self.vei_topo_processed_file(i)).exists()
            or raw_out.stat().st_mtime < raw_in.stat().st_mtime
            or graph_out.stat().st_mtime < graph_in.stat().st_mtime
            or art_out.stat().st_mtime < topo_in[0].stat().st_mtime
            or vei_out.stat().st_mtime < topo_in[1].stat().st_mtime
            or self._raw_fundus_paths[i].stem not in od_mac_df.index
        ]
        if len(to_process) == 0:
            return

        run_parallel = Parallel(n_jobs=-2, return_as="generator_unordered")
        for name, od_yx, mac_yx in tqdm.tqdm(  # type: ignore
            run_parallel(delayed(VBranchDigraphDataset.process_single)(arg) for arg in to_process),
            # (VBranchDigraphDataset.process_single(arg) for arg in to_process),
            total=len(to_process),
            desc="Processing dataset",
            disable=not self.verbose,
        ):
            od_mac_df.loc[name] = [od_yx[0], od_yx[1], mac_yx[0], mac_yx[1]]

        od_mac_df.to_csv(Path(self.processed_dir) / "od_mac.csv")

    @staticmethod
    def process_single(
        opts: tuple[Path, tuple[Path, Path], Path, GNNAVSegToTree, str, Optional[int]],
    ) -> tuple[str, Point | None, Point | None]:
        graph, topo, fundus_path, av2tree, directory, resize_to = opts
        name = fundus_path.stem

        # === Load and crop fundus image ===
        fundus = FundusData(fundus_path)

        r, roi = None, None
        if resize_to is not None:
            fundus, roi = fundus.crop_to_roi(return_roi=True, ensure_square=True)
            r = resize_to / roi.w
            fundus = fundus.resize(r)

        transform: Optional[ResizeTranslateProjection] = None
        if roi is not None and r is not None:
            transform = ResizeTranslateProjection(r, -roi.top_left.numpy() * r)

        fundus.write_image(image=Path(directory) / "raw" / (name + ".jpg"), on_exists="overwrite")

        # === Find Optic Disc and Macula ===
        od_mac = segment(open_image(Path(directory) / "raw" / (name + ".jpg"))).numpy(force=True).argmax(axis=0)
        fundus = fundus.update(od=od_mac == 1, macula=od_mac == 2, reshape_method="resize")

        # === Load and preprocess graph ===
        GEO_ATTRS = [VBranchGeoData.Fields.TANGENTS, VBranchGeoData.Fields.TIPS_TANGENT, VBranchGeoData.Fields.CALIBRES]
        if graph.suffix == ".npz":
            graph = VGraph.load(graph, check_integrity=False)
            graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)
            if transform is not None and resize_to is not None:
                graph.transform(transform, inplace=True)
                graph.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
        else:
            fundus = fundus.update(av=graph, crop_pad=roi, reshape_method="resize")
            graph = av2tree.to_vgraph(fundus)
        graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)
        merge_nodes_by_distance(graph, max_distance=0.5, inplace=True)
        if len(duplicates := graph.branch_duplicates()):
            graph.delete_branch([b for d in duplicates for b in d[1:]], inplace=True)

        # === Load and preprocess GT topology ===
        trees: list[VTree] = []
        for tree_path in topo:
            tree = VTree.load(tree_path, check_integrity=True)
            if transform is not None and resize_to is not None:
                tree.transform(transform, inplace=True)
                tree.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
            merge_nodes_by_distance(tree, max_distance=0.5, inplace=True)
            if len(tree.branch_duplicates()):
                warnings.warn(f"Tree in sample {name} has duplicate branches after processing", stacklevel=1)
            trees.append(tree)

        # === Test for common branch in Artery and Vein trees ===
        merged_tree = trees[0].append(trees[1])
        merge_nodes_by_distance(merged_tree, max_distance=0.5, inplace=True)
        if len(merged_tree.branch_duplicates()):
            warnings.warn(f"Sample {name} has duplicated branches in artery and vein trees", stacklevel=1)

        # === Save processed data ===
        graph.save(Path(directory) / "graphs" / (name + ".npz"), on_exists="overwrite")
        trees[0].save(Path(directory) / "target-topo" / (name + "_art.npz"), on_exists="overwrite")
        trees[1].save(Path(directory) / "target-topo" / (name + "_vei.npz"), on_exists="overwrite")

        return name, fundus.od_center, fundus.macula_center

    def preload_from_disk(
        self,
    ) -> tuple[list[Path], list[tuple[TreeTopology, TreeTopology]], list[VGraph], npt.NDArray, npt.NDArray]:
        N = len(self._raw_fundus_paths)
        raw_paths: list[Path] = [None] * N
        target_topologies: list[tuple[TreeTopology, TreeTopology]] = [None] * N
        graphs: list[VGraph] = [None] * N

        opts = [(i, self.processed_dir, path.stem) for i, path in enumerate(self._raw_fundus_paths)]
        run_parallel = Parallel(n_jobs=-2, return_as="generator_unordered")
        for i, raw_path, target_topology, graph in tqdm.tqdm(  # type: ignore
            run_parallel(delayed(VBranchDigraphDataset.preload_single)(opt) for opt in opts),
            # (VBranchDigraphDataset.preload_single(opt) for opt in opts),
            total=N,
            desc="Preloading dataset",
            disable=not self.verbose,
        ):
            raw_paths[i] = raw_path
            graphs[i] = graph
            target_topologies[i] = target_topology

        df = pd.read_csv(Path(self.processed_dir) / "od_mac.csv", index_col="name", dtype={"name": str})
        df = df.loc[[path.stem for path in raw_paths]]
        od_yx = df[["od_y", "od_x"]].values.astype(np.float32)
        mac_yx = df[["mac_y", "mac_x"]].values.astype(np.float32)
        half = np.array([0, graphs[0].geometric_data().domain.size.x / 2])
        for i, (od, mac) in enumerate(zip(od_yx, mac_yx, strict=True)):
            if np.isnan(mac).any():
                mac_yx[i] = od + half if od[1] < half[1] else od - half

        return raw_paths, target_topologies, graphs, od_yx, mac_yx

    @staticmethod
    def preload_single(
        opt: tuple[int, str, str], check: bool = False
    ) -> tuple[int, Path, tuple[TreeTopology, TreeTopology], VGraph]:
        idx, processed_dir, fundus_name = opt
        raw_path = Path(processed_dir) / "raw" / (fundus_name + ".jpg")
        graph = VGraph.load(Path(processed_dir) / "graphs" / (fundus_name + ".npz"), check_integrity=check)
        tree = VTree.load(Path(processed_dir) / "target-topo" / (fundus_name + "_vei.npz"), check_integrity=check)
        vei_topo = TreeTopology.from_tree(tree, sparse=True, discard_tree=True, expand_labels_by=5)
        tree = VTree.load(Path(processed_dir) / "target-topo" / (fundus_name + "_art.npz"), check_integrity=check)
        art_topo = TreeTopology.from_tree(tree, sparse=True, discard_tree=True, expand_labels_by=5)
        return idx, raw_path, (art_topo, vei_topo), graph

    def len(self):
        return len(self.fundus_paths)

    def get(self, idx: int | str) -> VBranchDigraphData:
        if isinstance(idx, str):
            name = idx
            idx = [_.stem for _ in self._raw_fundus_paths].index(idx)
        else:
            name = self._raw_fundus_paths[idx].stem
        digraph, fundus_img, od_yx, mac_yx = self.get_sample(idx, augment=self.augment, test=False)
        return VBranchDigraphData.from_branch_digraph(
            digraph, torch.from_numpy(fundus_img), od_yx, mac_yx, name, EdgeAttrExtractor()
        )

    def get_sample(
        self, idx: int | str, *, augment: bool = False, test: bool = False
    ) -> tuple[VBranchDigraph, npt.NDArray, npt.NDArray, npt.NDArray]:
        if isinstance(idx, str):
            idx = [_.stem for _ in self._raw_fundus_paths].index(idx)
        fundus_path = self.fundus_paths[idx]
        fundus_image = read_image(fundus_path, cast_to_float=True)
        od_yx = self.od_yx[idx]
        mac_yx = self.mac_yx[idx]

        # === Deteriorate graph ===
        assert self.graphs is not None, "Graphs not loaded"
        graph = self.graphs[idx]
        if augment:
            graph = deteriorate_graph(graph)

        # === Compute BranchDigraph ===
        assert self.target_topologies is not None, "Target topologies not loaded"
        branch_digraph = VBranchDigraph.from_graph(graph, check=False)
        branch_digraph.compute_p_from_gt(*self.target_topologies[idx], check=False, smooth_p=self.line_p_smoothing)

        # === Test ===
        if test:
            lines_msg = branch_digraph.check_lines("report")
            lines_p_msg = branch_digraph.check_line_p("report")
            if lines_msg or lines_p_msg:
                if lines_msg:
                    lines_msg = "\n" + lines_msg
                if lines_p_msg:
                    lines_p_msg = "\n" + lines_p_msg
                warnings.warn(
                    f"Issues found in sample {idx} ({fundus_path.stem}):{lines_msg}{lines_p_msg}", stacklevel=1
                )
            try:
                branch_digraph.optimize_tree(keep_missing_branch=True)
            except Exception as e:
                warnings.warn(f"No optimal tree from sample {idx} ({fundus_path.stem}): {e}", stacklevel=1)

        # === Geometric and color augmentation ===
        sample = branch_digraph, fundus_image, od_yx, mac_yx
        if augment:
            sample = geometric_augment(sample)

        return sample

    def draw_jppype(
        self,
        idx: int | str,
        *,
        test: bool = False,
        augment: bool = False,
        gt_topo: bool = True,
        branch_label=False,
        node_label=False,
    ) -> tuple[Mosaic, VBranchDigraph, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Draw the sample using jppype for visualization. If gt_topo is True, also draw the ground truth topology.
        Parameters
        ----------
        idx : int or str
            Index of the sample to draw, or the name of the fundus image (without extension).
        test : bool, optional
            Whether to run checks and optimizations on the graph before drawing, by default False.
        augment : bool, optional
            Whether to apply data augmentation to the sample before drawing, by default False.
        gt_topo : bool, optional
            Whether to draw the ground truth topology in a separate view, by default True.
        branch_label : bool, optional
            Whether to label the branches with their indices, by default False.
        node_label : bool, optional
            Whether to label the nodes with their indices, by default False.
        Returns
        -------
        mosaic: Mosaic
            A jppype Mosaic object containing the visualizations.
        digraph: VBranchDigraph
            The branch digraph of the sample, after augmentation and checks.
        fundus_img: npt.NDArray
            The fundus image of the sample, after augmentation.
        od_yx: npt.NDArray
            The (y, x) coordinates of the optic disc center.
        mac_yx: npt.NDArray
            The (y, x) coordinates of the macula center.
        """
        if isinstance(idx, str):
            name = idx
            idx = [_.stem for _ in self._raw_fundus_paths].index(name)
        else:
            name = self._raw_fundus_paths[idx].stem
        fundus = FundusData(self.fundus_paths[idx])
        art_tree = VTree.load(Path(self.processed_dir) / "target-topo" / (name + "_art.npz"))
        vei_tree = VTree.load(Path(self.processed_dir) / "target-topo" / (name + "_vei.npz"))
        trees_gt = (art_tree, vei_tree)
        topo_gt = self.target_topologies[idx]

        digraph, fundus_img, od_yx, mac_yx = self.get_sample(idx, augment=augment, test=test)

        m = Mosaic(
            3 if gt_topo else 2,
            cols_titles=[name, "Predicted with GT Topology"] + (["Ground Truth"] if gt_topo else []),
            cell_height=700,
            background=fundus_img,
        )
        draw_graph(
            digraph.graph,
            view=m[0],
            edge="bspline",
            edge_labels=branch_label,
            node_labels=node_label,
        )

        m.views[0].add_graph([], nodes_yx=[od_yx, mac_yx], name="OD/Macula")
        m.views[0]["OD/Macula"].nodes_cmap = {0: "green", 1: "yellow"}

        try:
            solved_tree = digraph.optimize_tree(keep_missing_branch=True)
            draw_tree(solved_tree, view=m[1], branch_color="subtree", bspline_dir=True)
        except Exception:
            warnings.warn(f"Could not optimize tree for sample {name}. Drawing unoptimized tree instead.")

        if gt_topo:

            def overlay_topo(img: npt.NDArray[np.float64], topo: TreeTopology, art: bool) -> npt.NDArray[np.float64]:
                topo = topo.as_dense()
                subtree_map = TopologicalLabel.decode_subtree(topo.branch_map)
                alpha = np.zeros(topo.shape, dtype=np.float64)
                topo_img = np.zeros(img.shape, dtype=np.float64)
                main_color = AV_COLORS[AVLabel.ART if art else AVLabel.VEI]
                colors = iter(color_jitter(main_color, hue=0.1))

                for subtree_id in np.unique(subtree_map):
                    if subtree_id == -1:
                        continue
                    subtree_mask = subtree_map == subtree_id
                    topo_img[subtree_mask] = next(colors) / 255.0
                    subtree_rank_map = topo.rank_map[subtree_mask]
                    alpha[subtree_mask] = 0.8 - 0.7 * (subtree_rank_map / subtree_rank_map.max())

                return img * (1 - alpha[:, :, None]) + topo_img * alpha[:, :, None]

            topo_map = (fundus.image).transpose(1, 2, 0)
            topo_map = overlay_topo(topo_map, topo_gt[0], art=True)
            topo_map = overlay_topo(topo_map, topo_gt[1], art=False)

            m[2].add_image(topo_map, name="background")
            draw_trees(trees_gt, view=m[2], bspline_dir=True)
        return m, digraph, fundus_img, od_yx, mac_yx

    def show_tree_diff(
        self,
        idx: int | str,
        parent_pred: Int1DArray,
        dir_pred: Bool1DArray,
        fp_pred: Optional[Bool1DArray] = None,
        av_pred: Optional[Bool1DArray] = None,
    ) -> tuple[Mosaic, VTree]:
        if isinstance(idx, str):
            name = idx
            idx = [_.stem for _ in self._raw_fundus_paths].index(name)
        else:
            name = self._raw_fundus_paths[idx].stem
        fundus = FundusData(self.fundus_paths[idx])
        art_tree = VTree.load(Path(self.processed_dir) / "target-topo" / (name + "_art.npz"))
        vei_tree = VTree.load(Path(self.processed_dir) / "target-topo" / (name + "_vei.npz"))
        trees_gt = (art_tree, vei_tree)
        topo_gt = self.target_topologies[idx]

        digraph, fundus_img, od_yx, mac_yx = self.get_sample(idx, augment=False, test=False)

        m = Mosaic(
            3,
            cols_titles=[f"GT Tree: {name}", "Predicted Tree", "Reference Topology"],
            cell_height=700,
            background=fundus_img,
        )

        # Draw GT tree
        solved_tree = digraph.optimize_tree(keep_missing_branch=False)
        draw_tree(solved_tree, view=m[0], branch_color="subtree", bspline_dir=True)
        # cmap = m.views[0]["tree"].edges_cmap
        # for b_fp in np.where(digraph.missing_branch())[0]:
        #     cmap[b_fp] = "#ffffff"
        # m.views[0]["tree"].edges_cmap = cmap

        # Draw Predicted tree
        tree = digraph.compute_tree_from_arborescence(parent_pred, dir_pred, fp_pred, keep_missing_branch=False)
        draw_tree(tree, view=m[1], branch_color="subtree", bspline_dir=True, edge_labels=True, node_labels=False)
        # if av_pred is not None:
        #     cmap = {i: AV_COLORS[AVLabel.ART] if av else AV_COLORS[AVLabel.VEI] for i, av in enumerate(av_pred)}
        #     if fp_pred is not None:
        #         for i in np.where(fp_pred)[0]:
        #             cmap[i] = AV_COLORS[AVLabel.BKG]
        #     m.views[1]["tree"].edges_cmap = cmap

        if True:

            def overlay_topo(img: npt.NDArray[np.float64], topo: TreeTopology, art: bool) -> npt.NDArray[np.float64]:
                topo = topo.as_dense()
                subtree_map = TopologicalLabel.decode_subtree(topo.branch_map)
                alpha = np.zeros(topo.shape, dtype=np.float64)
                topo_img = np.zeros(img.shape, dtype=np.float64)
                main_color = AV_COLORS[AVLabel.ART if art else AVLabel.VEI]
                colors = iter(color_jitter(main_color, hue=0.1))

                for subtree_id in np.unique(subtree_map):
                    if subtree_id == -1:
                        continue
                    subtree_mask = subtree_map == subtree_id
                    topo_img[subtree_mask] = next(colors) / 255.0
                    subtree_rank_map = topo.rank_map[subtree_mask]
                    alpha[subtree_mask] = 0.8 - 0.7 * (subtree_rank_map / subtree_rank_map.max())

                return img * (1 - alpha[:, :, None]) + topo_img * alpha[:, :, None]

            topo_map = (fundus.image).transpose(1, 2, 0)
            topo_map = overlay_topo(topo_map, topo_gt[0], art=True)
            topo_map = overlay_topo(topo_map, topo_gt[1], art=False)

            m[2].add_image(topo_map, name="background")
            draw_trees(trees_gt, view=m[2], bspline_dir=True)

        else:
            assert digraph.graph is not None, "Graph must be loaded to infer tree"
            OD = Point(od_yx[0], od_yx[1])
            art_branch = digraph.graph.branch_attr["av"] == AVLabel.ART
            vei_branch = digraph.graph.branch_attr["av"] == AVLabel.VEI
            parent_base = -np.ones(digraph.graph.branch_count, dtype=np.int_)
            dir_base = np.zeros(digraph.graph.branch_count, dtype=np.bool_)
            parent_base[art_branch], dir_base[art_branch] = naive_infer_arborescence(
                digraph.graph, OD, branch_subset=art_branch
            )
            parent_base[vei_branch], dir_base[vei_branch] = naive_infer_arborescence(
                digraph.graph, OD, branch_subset=vei_branch
            )

            tree = digraph.compute_tree_from_arborescence(parent_base, dir_base, fp_pred, keep_missing_branch=True)
            draw_tree(tree, view=m[2], branch_color="subtree", bspline_dir=True)

        return m, tree

    def split(self, indices: Sequence[int], augment: Optional[bool] = None) -> VBranchDigraphDataset:
        """Create a new dataset with only the samples at the specified indices."""
        dataset = copy.copy(self)
        dataset.fundus_paths = [self.fundus_paths[i] for i in indices]
        dataset._raw_fundus_paths = [self._raw_fundus_paths[i] for i in indices]
        dataset.graphs = [self.graphs[i] for i in indices]
        dataset._graphs_path = [self._graphs_path[i] for i in indices]
        dataset.target_topologies = [self.target_topologies[i] for i in indices]
        dataset._target_topologies_path = [self._target_topologies_path[i] for i in indices]
        dataset.od_yx = self.od_yx[indices]
        dataset.mac_yx = self.mac_yx[indices]
        if augment is not None:
            dataset.augment = augment
        return dataset

    def split_loaders(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        *,
        rng_seed: Optional[int] = None,
    ) -> tuple[VBranchDigraphDataset, VBranchDigraphDataset, VBranchDigraphDataset]:
        """Split the dataset into train, validation and test sets and return corresponding DataLoaders."""
        assert train_ratio + val_ratio < 1.0, "train_ratio and val_ratio must sum to less than 1.0"

        subset_paths = {}
        samples_subset = [subset_paths.setdefault(path.parent, len(subset_paths)) for path in self._raw_fundus_paths]

        train_indices = []
        val_indices = []
        test_indices = []
        rs = RandomState(MT19937(SeedSequence(if_none(rng_seed, 123456))))
        for _, samples in np_group_by(np.arange(len(self)), np.array(samples_subset)):
            rs.shuffle(samples)
            num_samples = len(samples)
            train_end = int(train_ratio * num_samples)
            val_end = int((train_ratio + val_ratio) * num_samples)
            train_indices.extend(samples[:train_end])
            val_indices.extend(samples[train_end:val_end])
            test_indices.extend(samples[val_end:])

        train_dataset = self.split(train_indices, augment=True)
        val_dataset = self.split(val_indices, augment=False)
        test_dataset = self.split(test_indices, augment=False)

        return train_dataset, val_dataset, test_dataset

    @classmethod
    def load_from_dirs(
        cls,
        fundus_dir: Path | Sequence[Path],
        target_topology_dir: Path | Sequence[Path],
        graph_dir: Path | Sequence[Path | None] | None = None,
        av_dir: Path | Sequence[Path | None] | None = None,
        *,
        line_p_smoothing: float = 0.0,
        root: Optional[str | Path] = None,
        fundus_ext: str | None | Sequence[str | None] = None,
        av_ext: str | None | Sequence[str | None] = None,
        verbose: bool = True,
        transform=None,
        resize_to: Optional[int] = None,
        overwrite: Optional[bool] = None,
        ignore_recent: Optional[int | datetime] = None,
    ) -> Self:
        GRAPH_EXT, ART_EXT, VEI_EXT = ".npz", "_art.npz", "_vei.npz"

        if isinstance(ignore_recent, datetime):
            ignore_recent = int(ignore_recent.timestamp())

        def discover_paths(fundus_dir, target_topology_dir, graph_dir, av_dir, fundus_ext, av_ext):
            if fundus_ext is None:
                fundus_ext = most_common_image_ext(fundus_dir)

            fundus_paths = fundus_dir.glob(f"*{fundus_ext}")
            if graph_dir is not None:
                graphs = graph_dir.glob(f"*{GRAPH_EXT}")
            elif av_dir is not None:
                if av_ext is None:
                    av_ext = most_common_image_ext(av_dir)
                graphs = av_dir.glob(f"*{av_ext}")
            else:
                raise ValueError("Either graph_dir or av_dir must be provided")
            target_topo_art: Iterable[Path] = target_topology_dir.glob(f"*{ART_EXT}")
            target_topo_vei: Iterable[Path] = target_topology_dir.glob(f"*{VEI_EXT}")

            if ignore_recent is not None:
                target_topo_art = [_ for _ in target_topo_art if _.stat().st_mtime < ignore_recent]

            filenames = sorted(
                {p.stem for p in fundus_paths}
                & {g.stem for g in graphs}
                & {t.stem[:-4] for t in target_topo_art}
                & {t.stem[:-4] for t in target_topo_vei}
            )
            fundus_paths = [fundus_dir / f"{name}{fundus_ext}" for name in filenames]
            if graph_dir is not None:
                graph_paths = [graph_dir / f"{name}{GRAPH_EXT}" for name in filenames]
            else:
                assert av_dir is not None
                graph_paths = [av_dir / f"{name}{av_ext}" for name in filenames]
            target_paths: list[tuple[Path, Path]] = [
                tuple(target_topology_dir / f"{name}{ext}" for ext in [ART_EXT, VEI_EXT]) for name in filenames
            ]
            return fundus_paths, graph_paths, target_paths

        if isinstance(fundus_dir, Sequence):
            N = len(fundus_dir)
            assert isinstance(target_topology_dir, Sequence), "target_topology_dir must be a sequence if fundus_dir is"
            assert len(target_topology_dir) == N, "fundus_dir and target_topology_dir must have the same length"
            if graph_dir is None:
                graph_dir = [None] * N
            assert isinstance(graph_dir, Sequence), "graph_dir must be a sequence if fundus_dir is"
            assert len(graph_dir) == N, "fundus_dir and graph_dir must have the same length"
            if av_dir is None:
                av_dir = [None] * N
            assert isinstance(av_dir, Sequence), "av_dir must be a sequence if fundus_dir is"
            assert len(av_dir) == N, "fundus_dir and av_dir must have the same length"

            if not isinstance(fundus_ext, Sequence):
                fundus_ext = [fundus_ext] * N
            assert isinstance(fundus_ext, Sequence), "fundus_ext must be a sequence if fundus_dir is"
            assert len(fundus_ext) == N, "fundus_ext must have the same length as fundus_dir"

            if not isinstance(av_ext, Sequence):
                av_ext = [av_ext] * N
            assert isinstance(av_ext, Sequence), "av_ext must be a sequence if fundus_dir is"
            assert len(av_ext) == N, "av_ext must have the same length as fundus_dir"

            fundus_paths, graph_paths, target_paths = [], [], []
            for f_paths, g_paths, t_paths in (
                discover_paths(*opts)
                for opts in zip(fundus_dir, target_topology_dir, graph_dir, av_dir, fundus_ext, av_ext, strict=True)
            ):
                fundus_paths.extend(f_paths)
                graph_paths.extend(g_paths)
                target_paths.extend(t_paths)
        else:
            assert isinstance(target_topology_dir, Path), "target_topology_dir must be a Path if fundus_dir is"
            assert isinstance(graph_dir, Path) or graph_dir is None, "graph_dir must be a Path if fundus_dir is"
            assert isinstance(av_dir, Path) or av_dir is None, "av_dir must be a Path if fundus_dir is"
            fundus_paths, graph_paths, target_paths = discover_paths(
                fundus_dir, target_topology_dir, graph_dir, av_dir, fundus_ext, av_ext
            )

        print(f"Found {len(fundus_paths)} branch digraphs...")

        if root is None:
            fundus_hash = hashlib.sha256(str(fundus_dir).encode("utf-8")).hexdigest()
            if resize_to is not None:
                fundus_hash += f"-{resize_to}"
            root = Path(tempfile.gettempdir()) / "fundus-vessels-toolkit" / "datasets-cache" / fundus_hash

        return cls(
            root,
            fundus_paths,
            graph_paths,
            target_paths,
            verbose=verbose,
            resize_to=resize_to,
            transform=transform,
            overwrite=overwrite,
            line_p_smoothing=line_p_smoothing,
        )
