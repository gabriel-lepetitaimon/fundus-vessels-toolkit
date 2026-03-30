from __future__ import annotations

import copy
import hashlib
import shutil
import tarfile
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Self, Sequence, TypeGuard

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import tqdm
from attr import dataclass
from fundus_data_toolkit.functional import open_image
from joblib import Parallel, delayed
from numpy.random import MT19937, RandomState, SeedSequence
from torch import Tensor
from torch_geometric.data import Data as PygData
from torch_geometric.data import Dataset as PygDataset

from fundus_odmac_toolkit.models.segmentation import segment
from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.utils.color import color_jitter
from fundus_toolkits.utils.geometric import Point, Rect
from fundus_toolkits.utils.image import read_image
from fundus_vessels_toolkit.utils.tree import tree_connected_components

from ...pipelines.avseg_to_tree import GNNAVSegToTree
from ...segment_to_graph.graph_simplification import merge_nodes_by_distance
from ...segment_to_graph.tree_topology import TopologicalLabel
from ...segment_to_graph.vbranch_digraph import (
    # BaseEdgeAttrExtractor,
    # EdgeAttrExtractor,
    TreeTopology,
    VBranchDigraph,
    VGraph,
)
from ...utils import if_none
from ...utils.data_io import most_common_image_ext
from ...utils.fundus_projections import ResizeTranslateProjection
from ...utils.numpy import np_group_by
from ...utils.typing import Bool1DArray, Int1DArray
from ...vascular_data_objects import VBranchGeoData, VTree
from .data import BranchDigraphData
from .data_augmentation import deteriorate_graph, geometric_augment

if TYPE_CHECKING:
    from ...utils.jppype import Mosaic


GRAPH_EXT, ART_EXT, VEI_EXT = ".npz", "_art.npz", "_vei.npz"


@dataclass
class SampleSource:
    fundus: Path
    target_topologies: tuple[Path, Path]
    graphes: dict[str, Path]
    date: datetime

    @classmethod
    def from_paths(cls, fundus_path: Path, target_topology_stem: Path, graphes_path: dict[str, Path]):
        art_topo = target_topology_stem.with_suffix(ART_EXT)
        vei_topo = target_topology_stem.with_suffix(VEI_EXT)
        target_topologies = (art_topo, vei_topo)
        date = max(
            art_topo.stat().st_mtime, vei_topo.stat().st_mtime, max(p.stat().st_mtime for p in graphes_path.values())
        )
        return cls(
            fundus=fundus_path,
            target_topologies=target_topologies,
            graphes=graphes_path,
            date=datetime.fromtimestamp(date),
        )


@dataclass
class SampleInfo:
    fundus: Path
    target_topologies: tuple[Path, Path]
    graphes: dict[str, Path]
    od_center: Point | None
    macula_center: Point | None


class BranchDigraphDataset(PygDataset):
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

    def preload_from_disk(
        self,
    ) -> tuple[list[Path], list[tuple[TreeTopology, TreeTopology]], list[VGraph], npt.NDArray, npt.NDArray]:
        N = len(self._raw_fundus_paths)
        raw_paths: list[Path] = [Path()] * N
        target_topologies: list[tuple[TreeTopology, TreeTopology]] = [None] * N  # type: ignore[list-item]
        graphs: list[VGraph] = [None] * N  # type: ignore[list-item]

        opts = [(i, self.processed_dir, path.stem) for i, path in enumerate(self._raw_fundus_paths)]
        run_parallel = Parallel(n_jobs=-2, return_as="generator_unordered")
        for i, raw_path, target_topology, graph in tqdm.tqdm(  # type: ignore
            run_parallel(delayed(BranchDigraphDataset.preload_single)(opt) for opt in opts),
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

    def get(self, idx: int | str) -> BranchDigraphData:
        if isinstance(idx, str):
            name = idx
            idx = [_.stem for _ in self._raw_fundus_paths].index(idx)
        else:
            name = self._raw_fundus_paths[idx].stem
        digraph, fundus_img, od_yx, mac_yx = self.get_sample(idx, augment=self.augment, test=False)
        return BranchDigraphData.from_branch_digraph(digraph, torch.from_numpy(fundus_img), od_yx, mac_yx, name)

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
        from ...utils.jppype import Mosaic, draw_graph, draw_tree, draw_trees

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
        assert digraph.graph is not None, "Graph must be constructed to draw"

        m = Mosaic(
            3 if gt_topo else 2,
            cols_titles=[name, "Predicted with GT Topology"] + (["Ground Truth"] if gt_topo else []),
            cell_height=700,
            background=fundus_img,
        )
        draw_graph(
            digraph.graph,
            view=m.views[0],
            edge="bspline",
            edge_labels=branch_label,
            node_labels=node_label,
        )

        m.views[0].add_graph([], nodes_yx=[od_yx, mac_yx], name="OD/Macula")
        m.views[0]["OD/Macula"].nodes_cmap = {0: "green", 1: "yellow"}  # type: ignore

        try:
            solved_tree = digraph.optimize_tree(keep_missing_branch=True)
            draw_tree(solved_tree, view=m[1], branch_color="subtree", bspline_dir=True)
        except Exception:
            warnings.warn(f"Could not optimize tree for sample {name}. Drawing unoptimized tree instead.", stacklevel=2)

        if gt_topo:
            topo_map = TreeTopology.av_overlay(fundus.image, topo_gt[0], topo_gt[1])
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
        from ...utils.jppype import AV_COLORS, Mosaic, draw_tree, draw_trees

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
        solved_tree = digraph.optimize_tree(keep_missing_branch=True)
        draw_tree(solved_tree, view=m[0], branch_color="subtree", bspline_dir=True)
        # cmap = m.views[0]["tree"].edges_cmap
        # for b_fp in np.where(digraph.missing_branch())[0]:
        #     cmap[b_fp] = "#ffffff"
        # m.views[0]["tree"].edges_cmap = cmap

        # Draw Predicted tree
        tree = digraph.compute_tree_from_arborescence(parent_pred, dir_pred, fp_pred, keep_missing_branch=True)
        draw_tree(tree, view=m[1], branch_color="subtree", bspline_dir=True, edge_labels=True, node_labels=False)
        if av_pred is not None:
            cmap = {i: AV_COLORS[AVLabel.ART] if av else AV_COLORS[AVLabel.VEI] for i, av in enumerate(av_pred)}
            if fp_pred is not None:
                for i in np.where(fp_pred)[0]:
                    cmap[i] = AV_COLORS[AVLabel.BKG]
            m.views[1]["tree"].edges_cmap = cmap  # type: ignore

        topo_map = TreeTopology.av_overlay(fundus.image, topo_gt[0], topo_gt[1])
        m[2].add_image(topo_map, name="background")
        draw_trees(trees_gt, view=m[2], bspline_dir=True)

        return m, tree

    def split(self, indices: Sequence[int], augment: Optional[bool] = None) -> BranchDigraphDataset:
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
    ) -> tuple[BranchDigraphDataset, BranchDigraphDataset, BranchDigraphDataset]:
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

    @classmethod
    def discover_paths(
        cls,
        fundus_dir: Path,
        target_topology_dir: Path,
        graph_dir: Path | dict[str, Path],
        *,
        fundus_ext: str | None,
        av_ext: str | None,
        ignore_recent: Optional[int | datetime] = None,
    ) -> list[SampleSource]:
        if fundus_ext is None:
            fundus_ext = most_common_image_ext(fundus_dir)

        if isinstance(ignore_recent, datetime):
            ignore_recent = int(ignore_recent.timestamp())

        fundus_paths = fundus_dir.glob(f"*{fundus_ext}")
        if not isinstance(graph_dir, dict):
            graph_dir = {"": graph_dir}

        graphes = {}  # {"stem": {"graph_type": Path()} }
        for graph_type, dir_path in graph_dir.items():
            graph_files = dir_path.glob(f"*{GRAPH_EXT}")
            if av_ext is None:
                av_ext = most_common_image_ext(dir_path, raise_if_not_found=False)
            if av_ext:
                img_files = dir_path.glob(f"*{av_ext}")
                graph_files = ({f.stem: f for f in img_files} | {f.stem: f for f in graph_files}).values()
            for file in graph_files:
                graphes.setdefault(file.stem, {}).set(graph_type, file)

        target_topo_art: Iterable[Path] = target_topology_dir.glob(f"*{ART_EXT}")
        target_topo_vei: Iterable[Path] = target_topology_dir.glob(f"*{VEI_EXT}")

        if ignore_recent is not None:
            target_topo_art = [_ for _ in target_topo_art if _.stat().st_mtime < ignore_recent]

        filenames = sorted(
            {p.stem for p in fundus_paths}
            & set(graphes.keys())
            & {t.stem[:-4] for t in target_topo_art}
            & {t.stem[:-4] for t in target_topo_vei}
        )

        return [
            SampleSource.from_paths(
                fundus_path=fundus_dir / f"{name}{fundus_ext}",
                target_topology_stem=target_topology_dir / name,
                graphes_path=graphes[name],
            )
            for name in filenames
        ]

    @classmethod
    def bundle(
        cls,
        output_archive: Path,
        fundus_dir: Path,
        target_topology_dir: Path,
        graph_dir: Path | dict[str, Path] | None = None,
        av_dir: Path | dict[str, Path] | None = None,
    ):
        av2tree = GNNAVSegToTree()

        with tempfile.TemporaryDirectory() as tmp_dir_path:
            tmp_dir = Path(tmp_dir_path)
            if (od_mac_file := tmp_dir / "od_mac.csv").exists() and self.overwrite is not True:
                od_mac_df = pd.read_csv(od_mac_file).set_index("name")
                if "Unnamed: 0" in od_mac_df.columns:
                    od_mac_df.drop(columns=["Unnamed: 0"], inplace=True)
            else:
                od_mac_df = pd.DataFrame(columns=["name", "od_y", "od_x", "mac_y", "mac_x"]).set_index("name")

            to_process = [
                (graph_in, topo_in, raw_in, av2tree, tmp_dir, self.resize_to)
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
            process_single = delayed(BranchDigraphDataset.process_single)
            for name, od_yx, mac_yx in tqdm.tqdm(  # type: ignore
                run_parallel(process_single(*arg) for arg in to_process),
                total=len(to_process),
                desc="Processing dataset",
                disable=not self.verbose,
            ):
                od_mac_df.loc[name] = [od_yx[0], od_yx[1], mac_yx[0], mac_yx[1]]

            od_mac_df.to_csv(Path(self.processed_dir) / "od_mac.csv")

    @classmethod
    def process_sample_from_source(
        cls,
        sample: SampleSource,
        output_dir: Path,
        *,
        av2tree: GNNAVSegToTree,
        resize_to: Optional[int],
    ) -> SampleInfo:
        name = sample.fundus_path.stem

        # === Load and crop fundus image ===
        fundus = FundusData(sample.fundus_path)

        r, roi = None, None
        if resize_to is not None:
            fundus, roi = fundus.crop_to_roi(return_roi=True, ensure_square=True)
            r = resize_to / roi.w
            fundus = fundus.resize(r)

        transform: Optional[ResizeTranslateProjection] = None
        if roi is not None and r is not None:
            transform = ResizeTranslateProjection(r, -roi.top_left.numpy() * r)

        fundus_path = output_dir / "raw" / (name + ".jpg")
        fundus.write_image(image=fundus_path, on_exists="overwrite")

        # === Find Optic Disc and Macula ===
        od_mac = segment(open_image(fundus_path)).numpy(force=True).argmax(axis=0)  # type: ignore
        fundus = fundus.update(od=od_mac == 1, macula=od_mac == 2, reshape_method="resize")

        # === Load, preprocess and save graphes ===
        GEO_ATTRS = [VBranchGeoData.Fields.TANGENTS, VBranchGeoData.Fields.TIPS_TANGENT, VBranchGeoData.Fields.CALIBRES]

        if not isinstance(sample.graphes, dict):
            graphes = {"": sample.graphes}

        for graph_version_name, graph_path in graphes.items():
            if isinstance(graph_path, VGraph):
                graph = graph_path.copy()
            elif graph_path.suffix == ".npz":
                # Load from graph file
                graph = VGraph.load(graph_path, check_integrity=False)
                if transform is not None and resize_to is not None:
                    graph.transform(transform, inplace=True)
                    graph.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
            else:
                # Parse AV segmentation to graph
                graph = av2tree.to_vgraph(fundus.update(av=graph_path, crop_pad=roi, reshape_method="resize"))

            # Remove duplicated branches and nodes, and remove useless attributes
            graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)
            merge_nodes_by_distance(graph, max_distance=0.5, inplace=True)
            if len(duplicates := graph.branch_duplicates()):
                graph.delete_branch([b for d in duplicates for b in d[1:]], inplace=True)

            # Save processed graph
            graph_dir = Path(output_dir) / "graphs"
            graphes: dict[str, Path] = {}
            if graph_version_name:
                graph_dir = graph_dir / graph_version_name
            graph_path = graph_dir / (name + ".npz")
            graph.save(graph_path, on_exists="overwrite")
            graphes[graph_version_name] = graph_path

        # === Load and preprocess GT topology ===
        trees: list[VTree] = []
        for tree_path in sample.topo:
            tree = VTree.load(tree_path, check_integrity=True)
            if transform is not None and resize_to is not None:
                tree.transform(transform, inplace=True)
                tree.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
            merge_nodes_by_distance(tree, max_distance=0.5, inplace=True)
            if len(tree.branch_duplicates()):
                warnings.warn(f"Tree in sample {name} has duplicate branches after processing", stacklevel=1)
            trees.append(tree)

        # Test for common branch in Artery and Vein trees
        merged_tree = trees[0].append(trees[1])
        merge_nodes_by_distance(merged_tree, max_distance=0.5, inplace=True)
        if len(merged_tree.branch_duplicates()):
            warnings.warn(f"Sample {name} has duplicated branches in artery and vein trees", stacklevel=1)

        # Rasterize topologies
        art_topo = TreeTopology.from_tree(trees[0], expand_labels_by=5)
        vei_topo = TreeTopology.from_tree(trees[1], expand_labels_by=5)

        # Save processed topologies
        topo_path = {av: Path(output_dir) / "target-topo" / f"{name}_{av}.npz" for av in ["art", "vei"]}
        art_topo.save(topo_path["art"], on_exists="overwrite")
        vei_topo.save(topo_path["vei"], on_exists="overwrite")
        return SampleInfo(
            fundus_path=fundus_path,
            topo=(topo_path["art"], topo_path["vei"]),
            graphes=graphes,
            od_center=fundus.od_center,
            macula_center=fundus.macula_center,
        )


class BranchDigraphDatasetArchive:
    def __init__(self, archive_path: Path):
        self.archive_path = archive_path
        self.infos = self.read_infos()
        self._tmp_dir = None

    @property
    def archive_dir(self) -> Path:
        if self._tmp_dir is None:
            return self.extract()
        return self._tmp_dir

    def extract(self, to: Optional[Path] = None) -> Path:
        if to is None:
            if self._tmp_dir is None:
                self._tmp_dir = Path(tempfile.mkdtemp())
            to = self._tmp_dir

        if not (to / "infos.csv").exists():
            with tarfile.open(self.archive_path, "r") as tar:
                tar.extractall(to)
        return to

    def close(self):
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir)
            self._tmp_dir = None

    def __delete__(self, instance):
        self.close()

    # === SAMPLES INFORMATIONS ===
    def read_infos(self) -> pd.DataFrame:
        if not self.archive_path.exists():
            return pd.DataFrame(columns=["name", "od_y", "od_x", "mac_y", "mac_x", "date"]).set_index("name")
        with tarfile.open(self.archive_path, "r") as tar:
            infos = pd.read_csv(tar.extractfile("infos.csv")).set_index("name")  # type: ignore
        return infos

    def __len__(self):
        return len(self.infos)
