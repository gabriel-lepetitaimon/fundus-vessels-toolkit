import hashlib
import math
import multiprocessing
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Self, Sequence

import numpy as np
import numpy.typing as npt
import torch
import torch_geometric as pyg
import tqdm
from jppype import Mosaic
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset

from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.utils.color import color_jitter
from fundus_toolkits.utils.geometric import Rect
from fundus_toolkits.utils.image import read_image

from ...pipelines.avseg_to_tree import GNNAVSegToTree
from ...utils.data_io import most_common_image_ext
from ...utils.fundus_projections import ResizeTranslateProjection
from ...utils.jppype import AV_COLORS, draw_graph, draw_tree, draw_trees
from ...vascular_data_objects import VBranchGeoData, VTree
from ..tree_topology import TopologicalLabel
from ..vbranch_digraph import TreeTopology, VBranchDigraph, VGraph
from .data_augmentation import deteriorate_graph, geometric_augment


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
                    curves_[i, 10:] = torch.tile(curve[C - halfC :], (repeat, 1))[:10]
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
            edge_first_tip=edge_first_tip,
            fundus_img=fundus_img,
            branch_curves=curves_,
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
    def from_branch_digraph(cls, branch_digraph: VBranchDigraph, fundus_img: torch.Tensor) -> Self:
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
            fundus_img=fundus_img,
            branch_curves=branch_curves,
            edge_p=edge_p,
            branch_av_p=branch_av_p,
            branch_dir=branch_dir,
        )


class VBranchDigraphDataset(PygDataset):
    def __init__(
        self,
        root: str | Path,
        fundus_paths: list[Path],
        graphs: list[Path],
        target_topologies: list[tuple[Path, Path]],
        *,
        verbose: bool = True,
        resize_to: Optional[int] = None,
        transform=None,
        overwrite: Optional[bool] = None,
    ):
        assert len(fundus_paths) == len(graphs) == len(target_topologies), "All input lists must have the same length"
        self._fundus_paths = fundus_paths
        self.resize_to = resize_to
        self.verbose = verbose

        self._graphs = graphs
        self._target_topologies = target_topologies
        self.overwrite = overwrite

        super().__init__(root=str(root), transform=transform, force_reload=overwrite is not False)
        self.fundus_paths, self.target_topologies, self.graphs = self.preload_from_disk()

    @property
    def processed_file_names(self):
        return [str(Path("raw") / (raw_file.stem + ".jpg")) for raw_file in self._fundus_paths]

    def raw_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "raw" / (self._fundus_paths[idx].stem + ".jpg")

    def graph_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "graphs" / (self._fundus_paths[idx].stem + ".npz")

    def art_topo_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "target-topo" / (self._fundus_paths[idx].stem + "_art.npz")

    def vei_topo_processed_file(self, idx: int) -> Path:
        return Path(self.processed_dir) / "target-topo" / (self._fundus_paths[idx].stem + "_vei.npz")

    def process(self):
        av2tree = GNNAVSegToTree()

        to_process = [
            (graph_in, topo_in, raw_in, av2tree, self.processed_dir, self.resize_to)
            for i, (graph_in, topo_in, raw_in) in enumerate(
                zip(self._graphs, self._target_topologies, self._fundus_paths, strict=True)
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
        ]

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(VBranchDigraphDataset.process_single, to_process),
                # (VBranchDigraphDataset.process_single(arg) for arg in args),
                total=len(to_process),
                desc="Processing dataset",
                disable=not self.verbose,
            ):
                pass

    @staticmethod
    def process_single(
        opts: tuple[Path, tuple[Path, Path], Path, GNNAVSegToTree, str, Optional[int]],
    ) -> None:
        graph, topo, fundus_path, av2tree, directory, resize_to = opts

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

        # === Load and preprocess graph ===
        GEO_ATTRS = [VBranchGeoData.Fields.TANGENTS, VBranchGeoData.Fields.TIPS_TANGENT, VBranchGeoData.Fields.CALIBRES]
        if graph.suffix == ".npz":
            graph = VGraph.load(graph)
            graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)
            if transform is not None and resize_to is not None:
                graph.transform(transform, inplace=True)
                graph.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
        else:
            fundus = fundus.update(av=graph, crop_pad=roi, reshape_method="resize")
            graph = av2tree.to_vgraph(fundus)
        graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)

        # === Load and preprocess GT topology ===
        trees = []
        for tree in topo:
            tree = VTree.load(tree)
            if transform is not None and resize_to is not None:
                tree.transform(transform, inplace=True)
                tree.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
            trees.append(tree)

        # === Save processed data ===
        name = fundus_path.stem
        fundus.write_image(image=Path(directory) / "raw" / (name + ".jpg"), on_exists="overwrite")
        graph.save(Path(directory) / "graphs" / (name + ".npz"), on_exists="overwrite")
        trees[0].save(Path(directory) / "target-topo" / (name + "_art.npz"), on_exists="overwrite")
        trees[1].save(Path(directory) / "target-topo" / (name + "_vei.npz"), on_exists="overwrite")

    def preload_from_disk(self) -> tuple[list[Path], list[tuple[TreeTopology, TreeTopology]], list[VGraph]]:
        N = len(self._fundus_paths)
        raw_paths: list[Path] = [None] * N
        target_topologies: list[tuple[TreeTopology, TreeTopology]] = [None] * N
        graphs: list[VGraph] = [None] * N

        opts = [(i, self.processed_dir, path.stem) for i, path in enumerate(self._fundus_paths)]
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            for i, raw_path, target_topology, graph in tqdm.tqdm(
                pool.imap_unordered(VBranchDigraphDataset.preload_single, opts),
                total=N,
                # (VBranchDigraphDataset.process_single(arg) for arg in args),
                desc="Preloading dataset",
                disable=not self.verbose,
            ):
                raw_paths[i] = raw_path
                graphs[i] = graph
                target_topologies[i] = target_topology

        return raw_paths, target_topologies, graphs

    @staticmethod
    def preload_single(opt: tuple[int, str, str]) -> tuple[int, Path, tuple[TreeTopology, TreeTopology], VGraph]:
        idx, processed_dir, fundus_name = opt
        raw_path = Path(processed_dir) / "raw" / (fundus_name + ".jpg")
        graph = VGraph.load(Path(processed_dir) / "graphs" / (fundus_name + ".npz"))
        tree = VTree.load(Path(processed_dir) / "target-topo" / (fundus_name + "_vei.npz"))
        vei_topo = TreeTopology.from_tree(tree, sparse=True, discard_tree=True, expand_labels_by=5)
        tree = VTree.load(Path(processed_dir) / "target-topo" / (fundus_name + "_art.npz"))
        art_topo = TreeTopology.from_tree(tree, sparse=True, discard_tree=True, expand_labels_by=5)
        return idx, raw_path, (art_topo, vei_topo), graph

    def len(self):
        return len(self.fundus_paths)

    def get(self, idx) -> VBranchDigraphData:
        digraph, fundus_img = self.get_sample(idx, augment=False, test=True)
        return VBranchDigraphData.from_branch_digraph(digraph, torch.from_numpy(fundus_img))

    def get_sample(self, idx: int, *, augment: bool = False, test: bool = False) -> tuple[VBranchDigraph, npt.NDArray]:
        fundus_path = self.fundus_paths[idx]
        fundus_image = read_image(fundus_path, cast_to_float=True)

        # === Deteriorate graph ===
        assert self.graphs is not None, "Graphs not loaded"
        graph = self.graphs[idx]
        # if augment:
        #     graph = deteriorate_graph(graph)

        # === Compute BranchDigraph ===
        assert self.target_topologies is not None, "Target topologies not loaded"
        branch_digraph = VBranchDigraph.from_graph(graph, check=False)
        branch_digraph.compute_p_from_gt(*self.target_topologies[idx], check=False)

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
                branch_digraph.optimize_tree(keep_missing_branch=False)
            except Exception as e:
                warnings.warn(f"No optimal tree from sample {idx} ({fundus_path.stem}): {e}", stacklevel=1)

        # === Geometric and color augmentation ===
        if augment:
            branch_digraph, fundus_image = geometric_augment((branch_digraph, fundus_image))

        return branch_digraph, fundus_image

    def draw_jppype(
        self, idx: int, *, test: bool = False, augment: bool = False
    ) -> tuple[Mosaic, VBranchDigraph, npt.NDArray]:
        name = self._fundus_paths[idx].stem
        fundus = FundusData(self.fundus_paths[idx])
        art_tree = VTree.load(Path(self.processed_dir) / "target-topo" / (name + "_art.npz"))
        vei_tree = VTree.load(Path(self.processed_dir) / "target-topo" / (name + "_vei.npz"))
        trees_gt = (art_tree, vei_tree)
        topo_gt = self.target_topologies[idx]

        digraph, fundus_img = self.get_sample(idx, augment=augment, test=test)

        m = Mosaic(
            3,
            cols_titles=["Predicted", "Predicted with GT Topology", "Ground Truth " + name],
            cell_height=700,
            background=fundus_img,
        )
        draw_graph(
            digraph.graph,
            view=m[0],
            edge_labels=True,
            node_labels=True,
        )
        try:
            solved_tree = digraph.optimize_tree(keep_missing_branch=False)
            draw_tree(solved_tree, view=m[1], branch_color="subtree", bspline_dir=True, edge="skeleton")
        except Exception:
            warnings.warn(f"Could not optimize tree for sample {name}. Drawing unoptimized tree instead.")

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
        return m, digraph, fundus_img

    @classmethod
    def load_from_dirs(
        cls,
        fundus_dir: Path | Sequence[Path],
        target_topology_dir: Path | Sequence[Path],
        graph_dir: Path | Sequence[Path | None] | None = None,
        av_dir: Path | Sequence[Path | None] | None = None,
        *,
        root: Optional[str | Path] = None,
        fundus_ext: str | None | Sequence[str | None] = None,
        av_ext: str | None | Sequence[str | None] = None,
        verbose: bool = True,
        transform=None,
        resize_to: Optional[int] = None,
        overwrite: Optional[bool] = None,
    ) -> Self:
        GRAPH_EXT, ART_EXT, VEI_EXT = ".npz", "_art.npz", "_vei.npz"

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
            target_topo_art = target_topology_dir.glob(f"*{ART_EXT}")
            target_topo_vei = target_topology_dir.glob(f"*{VEI_EXT}")

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
        )
