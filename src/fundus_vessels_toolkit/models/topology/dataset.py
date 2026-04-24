from __future__ import annotations

import copy
import hashlib
import shutil
import tarfile
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Self, Sequence, overload

import numpy as np
from scipy.__config__ import show
import tqdm
from joblib import Parallel, delayed
from numpy.random import MT19937, RandomState, SeedSequence
from pydantic import BaseModel, Field, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass
from skimage.morphology import binary_erosion, disk
from torch_geometric.data import Dataset as PygDataset

from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.utils.data_io import most_common_image_ext, overwrite_or_newer
from fundus_toolkits.utils.geometric import Point, Rect

from ...pipelines.avseg_to_tree import AVSegToTreeBase, GNNAVSegToTree
from ...segment_to_graph.graph_simplification import merge_nodes_by_distance
from ...segment_to_graph.vbranch_digraph import (
    TreeTopology,
    VBranchDigraph,
    VGraph,
)
from ...utils import if_none
from ...utils.fundus_projections import ResizeTranslateProjection
from ...utils.numpy import np_group_by
from ...utils.typing import Bool1DArray, Int1DArray, Int1DArrayLike
from ...vascular_data_objects import VBranchGeoData, VTree
from .data import BranchDigraphData
from .data_augmentation import AugmentationOpts

if TYPE_CHECKING:
    from ...utils.jppype import Mosaic


GRAPH_EXT, ART_EXT, VEI_EXT = ".npz", "_art.npz", "_vei.npz"


@dataclass
class BranchDigraphSample:
    """Data class storing in-memory a sample of a BranchDigraphDataset."""

    name: str
    fundus: FundusData
    art_topology: TreeTopology
    vei_topology: TreeTopology
    graphes: dict[str, VGraph]

    @property
    def target_topologies(self) -> tuple[TreeTopology, TreeTopology]:
        return self.art_topology, self.vei_topology


type DatasetType = Literal["train", "validation", "test"] | None


@pydantic_dataclass
class SampleInfo:
    """Data class representing a processed sample of a BranchDigraphDataset."""

    name: str
    fundus: Path
    od: Path
    macula: Path
    art_topology: Path
    vei_topology: Path
    graphes: dict[str, Path]
    date: datetime
    od_center: tuple[float, float] | None = Field(default=None)
    macula_center: tuple[float, float] | None = Field(default=None)
    dataset: str = Field(default="")
    dataset_type: DatasetType = Field(default=None)

    @property
    def target_topologies(self) -> tuple[Path, Path]:
        return self.art_topology, self.vei_topology

    def load(self, image: bool = True, discard_gt_tree: bool = True) -> BranchDigraphSample:
        """Load the sample from disk into memory."""
        if image:
            fundus = FundusData(image=self.fundus, od=self.od, macula=self.macula)
        else:
            fundus = FundusData.empty_like(self.fundus)
        if self.od_center is not None:
            fundus = fundus.update(od_center=Point(*self.od_center))
        if self.macula_center is not None:
            fundus = fundus.update(macula_center=Point(*self.macula_center))

        return BranchDigraphSample(
            name=self.name,
            fundus=fundus,
            art_topology=TreeTopology.load(self.art_topology, tree=not discard_gt_tree),
            vei_topology=TreeTopology.load(self.vei_topology, tree=not discard_gt_tree),
            graphes={k: VGraph.load(p) for k, p in self.graphes.items()},
        )

    def prefix(self, prefix: Path | str) -> Self:
        """Return a new BranchDigraphSampleInfo with the paths prefixed by the given path."""
        sample = copy.copy(self)
        sample.fundus = prefix / self.fundus
        sample.od = prefix / self.od
        sample.macula = prefix / self.macula
        sample.art_topology = prefix / self.art_topology
        sample.vei_topology = prefix / self.vei_topology
        sample.graphes = {k: prefix / p for k, p in sample.graphes.items()}
        return sample

    def relative_to(self, path: Path) -> Self:
        """Return a new BranchDigraphSampleInfo with the paths relative to the given path."""
        sample = copy.copy(self)
        sample.fundus = sample.fundus.relative_to(path)
        sample.od = sample.od.relative_to(path)
        sample.macula = sample.macula.relative_to(path)
        sample.art_topology = sample.art_topology.relative_to(path)
        sample.vei_topology = sample.vei_topology.relative_to(path)
        sample.graphes = {k: p.relative_to(path) for k, p in sample.graphes.items()}
        return sample

    def all_files(self, relative_to: Optional[Path | str] = None) -> list[Path]:
        """Return a list of all files associated with this sample, optionally relative to a given path."""

        paths = [self.fundus, self.od, self.macula, self.art_topology, self.vei_topology] + list(self.graphes.values())
        return [p.relative_to(relative_to) for p in paths] if relative_to is not None else paths

    MANIFEST_FILENAME = "manifest.json"

    @classmethod
    def decode(cls, infos: bytes | list | Path | str) -> list[SampleInfo]:
        if isinstance(infos, str):
            infos = Path(infos)
        if isinstance(infos, Path):
            if not infos.exists():
                return []

            if infos.name.endswith((".tar.gz", ".tar")):
                with tarfile.open(infos, "r") as tar:
                    file = tar.extractfile(cls.MANIFEST_FILENAME)
                    if file is None:
                        return []
                    return cls.decode(file.read())
            else:
                if infos.is_dir():
                    infos = infos / cls.MANIFEST_FILENAME
                if not infos.exists():
                    return []
                with open(infos, "rb") as f:
                    return cls.decode(f.read())

        adapter = TypeAdapter(list[SampleInfo])
        try:
            if isinstance(infos, bytes):
                return adapter.validate_json(infos)
            else:
                return adapter.validate_python(infos)
        except Exception as e:
            warnings.warn(f"Failed to decode sample infos: {e}", stacklevel=1)
            return []

    @classmethod
    def encode(
        cls, samples: list[SampleInfo], save: Optional[str | Path] = None, root_path: Optional[Path] = None
    ) -> bytes:
        adapter = TypeAdapter(list[SampleInfo])

        if root_path is not None:
            samples = [s.relative_to(root_path) for s in samples]

        json_bytes = adapter.dump_json(samples)

        if save is not None:
            save = Path(save)
            if not save.suffix == ".json":
                save = save / cls.MANIFEST_FILENAME
            save.parent.mkdir(parents=True, exist_ok=True)
            with open(save, "wb") as f:
                f.write(json_bytes)

        return json_bytes


@dataclass
class SampleSource:
    """Data class representing the source files of a sample of a BranchDigraphDataset, and providing methods to process them into a BranchDigraphSampleInfo."""  # noqa: E501

    fundus: Path
    """Path to the fundus image file."""

    od: Path
    """Path to the optic disc segmentation map."""

    macula: Path
    """Path to the macula segmentation map."""

    target_topologies: tuple[Path, Path]
    """Paths to the target topology files for the artery and vein trees. The files should be in a format loadable by VTree.load()."""  # noqa: E501

    graphes: dict[str, Path]
    """Paths to the graph files, with keys representing different versions of the graph (e.g. generated by different models). The files should be in a format loadable by VGraph.load() or an image parsable by AVSegToTreeBase."""  # noqa: E501

    date: datetime
    """Date of the sample, used for caching purposes. Should be the latest modification date among the source files."""

    dataset: str
    """Optional name of the dataset this sample belongs to."""

    _od_center: tuple[float, float] | None | EllipsisType = Field(default=..., repr=False)
    _mac_center: tuple[float, float] | None | EllipsisType = Field(default=..., repr=False)

    @property
    def name(self) -> str:
        """Name of the sample, derived from the fundus image file name."""
        return self.fundus.stem

    @classmethod
    def from_paths(
        cls,
        fundus_path: Path,
        od_path: Path,
        macula_path: Path,
        target_topology_stem: Path,
        graphes_path: dict[str, Path],
        dataset: str = "",
    ):
        """Create a BranchDigraphSampleSource from the given paths.

        Parameters
        ----------
        fundus_path : Path
            Path to the fundus image file.

        target_topology_stem : Path
            Stem path to the target topology files for the artery and vein trees. The retrieved paths will be ``target_topology_stem+ART_EXT`` and  ``target_topology_stem+VEI_EXT``.

        graphes_path : dict[str, Path]
            Paths to the graph files, with keys representing different versions of the graph (e.g. generated by different models). The files should be in a format loadable by VGraph.load() or an image parsable by AVSegToTreeBase.

        dataset : str, optional
            Optional name of the dataset this sample belongs to.
        """  # noqa: E501
        art_topo = Path(str(target_topology_stem) + ART_EXT)
        vei_topo = Path(str(target_topology_stem) + VEI_EXT)
        target_topologies = (art_topo, vei_topo)
        date = max(
            art_topo.stat().st_mtime, vei_topo.stat().st_mtime, max(p.stat().st_mtime for p in graphes_path.values())
        )
        return cls(
            fundus=fundus_path,
            od=od_path,
            macula=macula_path,
            target_topologies=target_topologies,
            graphes=graphes_path,
            date=datetime.fromtimestamp(date),
            dataset=dataset,
        )

    def output_paths(self, output_dir: Path) -> SampleInfo:
        """Return the output paths for the processed sample, based on the given output directory."""
        name = self.name
        fundus_path = output_dir / "raw" / (name + ".jpg")
        graphes = {
            version: (
                output_dir / "graphs" / version / (name + ".npz")
                if version
                else output_dir / "graphs" / (name + ".npz")
            )
            for version in self.graphes.keys()
        }
        topo_path = {av: output_dir / "target-topo" / f"{name}_{av}.npz" for av in ["art", "vei"]}
        od = output_dir / "od" / (name + ".png")
        macula = output_dir / "macula" / (name + ".png")

        return SampleInfo(
            name=name,
            fundus=fundus_path,
            od=od,
            macula=macula,
            art_topology=topo_path["art"],
            vei_topology=topo_path["vei"],
            graphes=graphes,
            od_center=None,
            macula_center=None,
            date=self.date,
            dataset=self.dataset,
        )

    def already_processed(
        self,
        existing_samples: dict[str, SampleInfo] | SampleInfo,
        overwrite: Optional[bool | datetime] = None,
    ) -> SampleInfo | None:
        """Check if the sample is already processed and up-to-date in the output directory, based on the existing samples and the overwrite policy."""  # noqa: E501
        existing_sample = existing_samples.get(self.name) if isinstance(existing_samples, dict) else existing_samples

        # A sample will not be reprocessed if:
        # 1. A sample exist and all its files exist on disk
        if existing_sample is None or not all(f.exists() for f in existing_sample.all_files()):
            return None

        # 2. All graphes versions are present in the existing sample
        if isinstance(self.graphes, dict):
            if not all(version in existing_sample.graphes for version in self.graphes.keys()):
                return None

        # 3. overwrite is not True and this source is older than the overwrite threshold
        match overwrite:
            case True:
                return None
            case False:
                return existing_sample
            case datetime() as max_datetime:
                if self.date > max_datetime:
                    return None

        # 4. This source is older than the existing sample
        return existing_sample if self.date <= existing_sample.date else None

    def compute_od_mac(self, overwrite: Optional[bool | datetime] = None) -> Self:
        """Compute the optic disc and macula centers from the fundus image if they are not already provided."""
        from fundus_odmac_toolkit.models.segmentation import segment as segment_od_mac
        from fundus_data_toolkit.functional import open_image

        fundus = FundusData.empty_like(self.fundus)

        save_od = overwrite_or_newer(self.fundus, self.od, overwrite)
        save_mac = overwrite_or_newer(self.fundus, self.macula, overwrite)
        if save_od or save_mac:
            od_mac = segment_od_mac(open_image(self.fundus)).numpy(force=True).argmax(axis=0)  # type: ignore
            fundus.update(od=od_mac == 1, macula=od_mac == 2, reshape_method="resize", inplace=True)
            assert fundus.od_center is not None and fundus.macula_center is not None

            if save_od:
                fundus.write_image(od=self.od, on_exists="overwrite")
            if save_mac:
                fundus.write_image(macula=self.macula, on_exists="overwrite")
            self._od_center = fundus.od_center if not fundus.od_center.is_nan() else None
            self._mac_center = fundus.macula_center if not fundus.macula_center.is_nan() else None
        else:
            if self._od_center is ...:
                fundus.update(od=self.od, inplace=True)
                assert fundus.od_center is not None
                self._od_center = fundus.od_center if not fundus.od_center.is_nan() else None
            if self._mac_center is ...:
                fundus.update(macula=self.macula, inplace=True)
                assert fundus.macula_center is not None
                self._mac_center = fundus.macula_center if not fundus.macula_center.is_nan() else None

        return self

    def process(
        self,
        output_dir: Path,
        *,
        resize_to: Optional[int] = None,
        av2tree: Optional[AVSegToTreeBase] = None,
        mask_optic_disc: bool = True,
        overwrite: Optional[bool | datetime] = None,
    ) -> SampleInfo:
        """Process the sample from the source files and save the processed data to disk. The fundus image may be resized to the specified size, and the graphes may be parsed from AV segmentation if av2tree is provided.

        Parameters
        ----------
        output_dir : Path
            Directory to save the processed data. The following files will be saved:
            - {output_dir}/raw/{name}.jpg: the processed fundus image.
            - {output_dir}/graphs/{graph_version_name}/{name}.npz: the processed graphes, with one file for each version. If graph_version_name is an empty string, the files will be saved directly under graphs/.
            - {output_dir}/target-topo/{name}_art.npz: the processed artery tree topology.
            - {output_dir}/target-topo/{name}_vei.npz: the processed vein tree topology.
            - {output_dir}/od/{name}.png: the optic disc mask.

        resize_to : Optional[int]
            If specified, the fundus image will be resized to have the specified width, and the graphes and topologies will be transformed accordingly. If not specified, the original size will be kept.

        av2tree : Optional[AVSegToTreeBase]
            The method used to parse AV segmentation images to a graph. Only used if the graphes provided are image files instead of graph files.

        overwrite: Optional[bool]
            Whether to overwrite the processed files if they already exist. If None, the files will be overwritten if the source files are newer than the processed files, and kept otherwise.

        Returns
        -------
        BranchDigraphSampleInfo
            The information of the processed sample, including the paths to the processed files and the coordinates of the optic disc and macula centers.
        """  # noqa: E501
        output_paths = self.output_paths(output_dir)

        # === 1. Load and crop fundus image ===
        fundus = FundusData(self.fundus)

        r, roi = None, None
        if resize_to is not None:
            fundus, roi = fundus.crop_to_roi(return_roi=True, ensure_square=True)
            r = resize_to / roi.w
            fundus = fundus.resize(r)

        transform: Optional[ResizeTranslateProjection] = None
        if roi is not None and r is not None:
            transform = ResizeTranslateProjection(r, -roi.top_left.numpy() * r)

        if overwrite_or_newer(self.fundus, output_paths.fundus, overwrite):
            fundus.write_image(image=output_paths.fundus, on_exists="overwrite")

        # === 2. Load or compute and crop od and macula ===
        # Ensure the od and macula segmentation exists
        self.compute_od_mac(overwrite=overwrite)
        overwrite_od = overwrite_or_newer(self.od, output_paths.od, overwrite)

        if mask_optic_disc or overwrite_od:  # Load, crop and save OD
            fundus.update(od=self.od, inplace=True, crop_pad=roi, reshape_method="resize")
            if overwrite_od:
                fundus.write_image(od=output_paths.od, on_exists="overwrite")
        if overwrite_or_newer(self.macula, output_paths.macula, overwrite):  # Load, crop and save macula
            fundus.update(macula=self.macula, inplace=True, crop_pad=roi, reshape_method="resize")
            fundus.write_image(macula=output_paths.macula, on_exists="overwrite")

        # === 3. Load, preprocess and save graphes ===
        GEO_ATTRS = [VBranchGeoData.Fields.TANGENTS, VBranchGeoData.Fields.TIPS_TANGENT, VBranchGeoData.Fields.CALIBRES]
        graphes: dict[str, Path] = copy.copy(self.graphes) if isinstance(self.graphes, dict) else {"": self.graphes}

        for graph_version, graph_path in graphes.items():
            graphes[graph_version] = graph_out_path = output_paths.graphes[graph_version]
            if not overwrite_or_newer(graph_path, graph_out_path, overwrite):
                continue

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
                if av2tree is None:
                    av2tree = GNNAVSegToTree()
                fundus.update(av=graph_path, crop_pad=roi, reshape_method="resize", inplace=True)
                if mask_optic_disc:
                    selem = disk(fundus.od_diameter * 0.2, dtype=bool)
                    mask = ~binary_erosion(fundus.od, selem)  # type: ignore
                    fundus.update(av=fundus.av * mask, inplace=True)
                graph = av2tree.to_vgraph(fundus)

            # Remove duplicated branches and nodes, and remove useless attributes
            graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)
            merge_nodes_by_distance(graph, max_distance=0.5, inplace=True)
            if len(duplicates := graph.branch_duplicates()):
                graph.delete_branch([b for d in duplicates for b in d[1:]], inplace=True)

            # Save processed graph
            graph.save(graph_out_path, on_exists="overwrite")

        # === 4. Load and preprocess GT topology ===
        if any(
            overwrite_or_newer(src, dst, overwrite)
            for src, dst in zip(self.target_topologies, output_paths.target_topologies, strict=True)
        ):
            trees: list[VTree] = []
            for tree_path in self.target_topologies:
                tree = VTree.load(tree_path, check_integrity=True)
                if transform is not None and resize_to is not None:
                    tree.transform(transform, inplace=True)
                    tree.geometric_data()._domain = Rect.from_size((resize_to, resize_to))
                merge_nodes_by_distance(tree, max_distance=0.5, inplace=True)
                if len(tree.branch_duplicates()):
                    warnings.warn(f"Tree in sample {self.name} has duplicate branches after processing", stacklevel=1)
                trees.append(tree)

            # Test for common branch in Artery and Vein trees
            merged_tree = trees[0].append(trees[1])
            merge_nodes_by_distance(merged_tree, max_distance=0.5, inplace=True)
            if len(merged_tree.branch_duplicates()):
                warnings.warn(f"Sample {self.name} has duplicated branches in artery and vein trees", stacklevel=1)

            # Rasterize topologies
            art_topo = TreeTopology.from_tree(trees[0], expand_labels_by=5)
            vei_topo = TreeTopology.from_tree(trees[1], expand_labels_by=5)

            # Save processed topologies
            art_topo.save(output_paths.art_topology, on_exists="overwrite")
            vei_topo.save(output_paths.vei_topology, on_exists="overwrite")

        return output_paths


class BranchDigraphDatasetConfig(BaseModel):
    graph_version: str | dict[str, float] = Field(default_factory=dict)
    preload: bool | Literal["without-image"] = Field(default=False)
    augment: bool | AugmentationOpts = Field(default=False)
    verbose: bool = Field(default=True)
    # line_p_smoothing: NotRequired[float] = 0.0


class BranchDigraphDataset(PygDataset):
    def __init__(
        self,
        src_path: Path | str,
        root: Optional[str | Path] = None,
        *,
        transform=None,
        cfg: Optional[BranchDigraphDatasetConfig] = None,
    ):
        """Dataset of branch digraphs for fundus images, with multiple graph versions and target topologies.

        Parameters
        ----------
        src_path : Path | str
            Path to the processed dataset directory or archive, which should contain an appropriate manifest along with the processed data.
        root : Optional[str | Path], optional
            Root directory to store the dataset.
        """  # noqa: E501
        src_path = Path(src_path)
        self.src_path = src_path
        self.samples_info = SampleInfo.decode(src_path)
        self.cfg: BranchDigraphDatasetConfig = cfg or BranchDigraphDatasetConfig()

        if root is None:
            if src_path.is_dir():
                root = str(src_path)
            else:
                fundus_hash = hashlib.sha256(str(src_path).encode("utf-8")).hexdigest()
                root = str(Path(tempfile.gettempdir()) / "fundus-vessels-toolkit" / "datasets-cache" / fundus_hash)
        else:
            root = str(root) if isinstance(root, Path) else root
        self.root_specified = root is not None

        force_reload = False
        root_manifest = Path(root) / SampleInfo.MANIFEST_FILENAME
        if root_manifest.exists():
            src_manifest = src_path
            if not src_path.name.endswith((".tar", "tar.gz")):
                src_manifest = src_path / SampleInfo.MANIFEST_FILENAME
            force_reload = root_manifest.stat().st_mtime <= src_manifest.stat().st_mtime

        super().__init__(root=root, transform=transform, force_reload=force_reload)
        self.samples_info = [info.prefix(self.processed_dir) for info in self.samples_info]

        if (preload := self.cfg.preload) is not False:
            self.preload(with_image=preload != "without-image")
        else:
            self._preloaded_samples = None

    def preload(self, with_image: bool = False) -> Self:
        """Preload the samples into memory. If with_image is False, only the graph and topology data will be preloaded, and the fundus images will be loaded on demand when calling get_sample()."""  # noqa: E501
        self._preloaded_samples = [
            samples_info.load(image=with_image)
            for samples_info in tqdm.tqdm(self.samples_info, desc="Preloading dataset", disable=not self.cfg.verbose)
        ]
        return self

    def copy(self) -> Self:
        """Return a copy of the dataset."""
        new = copy.copy(self)
        new.cfg = new.cfg.model_copy(deep=True)
        new.samples_info = copy.copy(self.samples_info)
        if self._preloaded_samples is not None:
            new._preloaded_samples = copy.copy(self._preloaded_samples)
        return new

    @classmethod
    def load_from_dirs(
        cls,
        fundus_dir: Path | list[Path],
        target_topology_dir: Path | list[Path],
        graph_dir: Path | dict[str, Path] | list[Path] | list[dict[str, Path]],
        dataset_name: Optional[str | list[str]] = None,
        *,
        output_dir: Optional[Path | str] = None,
        resize_to: Optional[int] = None,
        mask_optic_disc: bool = False,
        overwrite: Optional[bool] = None,
        fundus_ext: str | None = None,
        av_ext: str | None = None,
        av2tree: Optional[AVSegToTreeBase] = None,
        ignore_recent: Optional[int | datetime] = None,
        cfg: Optional[BranchDigraphDatasetConfig] = None,
        n_workers: int = 0,
    ) -> Self:
        cfg = cfg or BranchDigraphDatasetConfig()
        verbose = cfg.verbose
        # === List source files ===
        if isinstance(fundus_dir, list):
            N = len(fundus_dir)
            assert isinstance(target_topology_dir, list) and len(target_topology_dir) == N, (
                "If fundus_dir is a list, target_topology_dir must be a list of the same length"
            )
            assert isinstance(graph_dir, list) and len(graph_dir) == N, (
                "If fundus_dir is a list, graph_dir must be a list of the same length"
            )
            if isinstance(dataset_name, str):
                dataset_name = [dataset_name] * N
            elif dataset_name is None:
                dataset_name = [f_dir.parent.name for f_dir in fundus_dir]
            else:
                assert isinstance(dataset_name, list) and len(dataset_name) == N, (
                    "If fundus_dir is a list, dataset_name should be either a string or a list of the same length"
                )

            samples_src = []
            for f_dir, t_dir, g_dir, d_name in zip(
                fundus_dir, target_topology_dir, graph_dir, dataset_name, strict=True
            ):
                samples_src.extend(
                    cls.discover_paths(
                        f_dir,
                        t_dir,
                        g_dir,
                        dataset=d_name,
                        fundus_ext=fundus_ext,
                        av_ext=av_ext,
                        ignore_recent=ignore_recent,
                    )
                )
        else:
            assert not isinstance(target_topology_dir, list), (
                "If fundus_dir is not a list, target_topology_dir should not be a list"
            )  # noqa: E501
            assert not isinstance(graph_dir, list), "If fundus_dir is not a list, graph_dir should not be a list"
            if dataset_name is None:
                dataset_name = ""
            assert isinstance(dataset_name, str), "If fundus_dir is not a list, dataset_name should a string"

            samples_src = cls.discover_paths(
                fundus_dir,
                target_topology_dir,
                graph_dir,
                dataset=dataset_name,
                fundus_ext=fundus_ext,
                av_ext=av_ext,
                ignore_recent=ignore_recent,
            )

        if verbose:
            print(f"Found {len(samples_src)} branch digraphs...")

        # === Generate output directory ===
        if output_dir is None:
            fundus_hash = hashlib.sha256(str(graph_dir).encode("utf-8")).hexdigest()
            if resize_to is not None:
                fundus_hash += f"-{resize_to}"
            output_dir = Path(tempfile.gettempdir()) / "fundus-vessels-toolkit" / "datasets-cache" / fundus_hash
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if overwrite is True and output_dir.exists():
            shutil.rmtree(output_dir)

        # === Process samples ===
        # Skip samples that are already processed and up-to-date
        existing_samples = {s.name: s for s in SampleInfo.decode(output_dir)} if not overwrite else {}

        samples: list[SampleInfo] = []
        for i, sample_src in reversed(list(enumerate(samples_src))):
            existing_sample = sample_src.already_processed(existing_samples, overwrite)
            if existing_sample is not None:
                samples.append(existing_sample)
                samples_src.pop(i)

        # Process remaining samples in parallel
        if len(samples_src) > 0:
            for sample_src in tqdm.tqdm(samples_src, desc="Computing OD/Macula centers", disable=not verbose):
                sample_src.compute_od_mac(overwrite=overwrite)

            process = partial(
                SampleSource.process,
                output_dir=output_dir,
                resize_to=resize_to,
                mask_optic_disc=mask_optic_disc,
                av2tree=av2tree,
                overwrite=overwrite,
            )
            if n_workers == 0:
                new_samples: list[SampleInfo] = [
                    process(sample_src)
                    for sample_src in tqdm.tqdm(
                        samples_src, total=len(samples_src), desc="Processing samples", disable=not verbose
                    )
                ]  # type: ignore
            else:
                run_parallel = Parallel(n_jobs=-2, return_as="generator_unordered")
                new_samples: list[SampleInfo] = list(
                    tqdm.tqdm(
                        run_parallel(delayed(process)(sample_src) for sample_src in samples_src),
                        total=len(samples_src),
                        desc="Processing samples",
                        disable=not verbose,
                    )
                )  # type: ignore
            samples.extend(new_samples)

        # Save samples info
        samples = sorted(samples, key=lambda s: s.name)
        if len(samples_src) > 0:
            SampleInfo.encode(samples, save=output_dir, root_path=output_dir)

        return cls(output_dir, cfg=cfg)

    @property
    def processed_dir(self) -> str:
        """Directory where the processed data is stored. If the src_path is an archive, this will be a temporary directory where the archive is extracted."""  # noqa: E501
        return self.root if self.root_specified else super().processed_dir

    def process(self) -> None:
        """Process the raw data files and save the processed data to disk. This method should be called before using the dataset for training or evaluation."""  # noqa: E501
        processed_dir = Path(self.processed_dir)
        if self.src_path.name.endswith((".tar", ".tar.gz")):
            try:
                with tarfile.open(self.src_path, "r:*") as tar:
                    tar.extractall(path=processed_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to extract dataset archive {self.src_path}: {e}") from e
        elif processed_dir != self.src_path:
            shutil.copytree(self.src_path, processed_dir, dirs_exist_ok=True)

    @property
    def processed_file_names(self):
        return [file for sample in self.samples_info for file in sample.all_files()]

    def len(self):
        return len(self.samples_info)

    @overload
    def get(
        self,
        idx: int | str,
        *,
        version: Optional[str] = None,
        augment: Optional[bool | AugmentationOpts] = None,
        return_digraph: Literal[False] = False,
    ) -> BranchDigraphData: ...
    @overload
    def get(
        self,
        idx: int | str,
        *,
        version: Optional[str] = None,
        augment: Optional[bool | AugmentationOpts] = None,
        return_digraph: Literal[True],
    ) -> tuple[BranchDigraphData, VBranchDigraph]: ...
    def get(
        self,
        idx: int | str,
        *,
        version: Optional[str] = None,
        augment: Optional[bool | AugmentationOpts] = None,
        return_digraph: bool = False,
    ) -> BranchDigraphData | tuple[BranchDigraphData, VBranchDigraph]:
        sample = self.get_sample(idx)
        graphes = list(sample.graphes.values())
        versions = list(sample.graphes.keys())
        if version is not None and version in versions:
            graph_version = version
        elif isinstance(self.cfg.graph_version, dict):
            pick_p = [self.cfg.graph_version.get(v, 0.0) for v in versions]
            p_total = sum(pick_p)
            if p_total == 0.0:
                graph_version = versions[np.random.randint(len(graphes))]
            else:
                graph_version = versions[np.random.choice(len(graphes), p=np.array(pick_p) / p_total)]
        elif self.cfg.graph_version is None or self.cfg.graph_version not in sample.graphes:
            graph_version = versions[np.random.randint(len(graphes))]
        else:  # version is a valid key in sample.graphes
            graph_version = self.cfg.graph_version
        graph = sample.graphes[graph_version]

        return BranchDigraphData.from_graph(
            graph,
            sample.fundus,
            sample.target_topologies,
            return_digraph=return_digraph,
            augment=self.cfg.augment if augment is None else augment,
            name=sample.name + (f"/{graph_version}" if graph_version else ""),
            od_center=sample.fundus.od_center if sample.fundus.has_od_center else None,
            mac_center=sample.fundus.macula_center if sample.fundus.has_macula_center else None,
        )

    def get_sample(self, idx: int | str, *, discard_gt_tree: bool = True) -> BranchDigraphSample:
        if isinstance(idx, str):
            idx = [s.name for s in self.samples_info].index(idx)
        if self._preloaded_samples is not None:
            sample = copy.copy(self._preloaded_samples[idx])
            if not sample.fundus.has_image:
                sample.fundus = sample.fundus.update(image=self.samples_info[idx].fundus)
            return sample
        else:
            return self.samples_info[idx].load(discard_gt_tree=discard_gt_tree)

    def jppype_show(
        self,
        idx: int | str,
        *,
        version: Optional[str] = None,
        augment: Optional[bool | AugmentationOpts] = None,
        branch_label=False,
        node_label=False,
    ) -> tuple[Mosaic, BranchDigraphSample, BranchDigraphData]:
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

        sample = self.get_sample(idx, discard_gt_tree=False)
        sample_data = self.get(idx, augment=augment, version=version)

        if "/" in sample_data.name:
            name, graph_version = sample_data.name.split("/", 1)
        else:
            name, graph_version = sample_data.name, ""

        m = Mosaic(
            3,
            cols_titles=[sample_data.name, "with GT Topology", "Ground Truth"],
            cell_height=700,
            background=sample.fundus.image,
        )
        draw_graph(
            sample.graphes[graph_version],
            view=m.views[0],
            edge="bspline",
            edge_labels=branch_label,
            node_labels=node_label,
        )

        digraph = sample_data.to_digraph(graph=True)
        assert digraph.graph is not None

        m.views[1].add_image(sample_data.img, "background")
        m.views[1].add_graph([], nodes_yx=[sample_data.od_yx.numpy(), sample_data.mac_yx.numpy()], name="OD/Macula")
        m.views[1]["OD/Macula"].nodes_cmap = {0: "blue", 1: "yellow"}  # type: ignore

        solved_tree = digraph.optimize_tree(keep_missing_branch=True)
        solved_tree_layer = draw_tree(
            solved_tree, view=m[1], branch_color="subtree", bspline_dir=True, interactive=True
        )
        solved_tree_layer.edges_cmap = {
            b - 1: color if b > len(digraph.branch_fp()) or not digraph.branch_fp()[b - 1] else "#777777"
            for b, color in solved_tree_layer.edges_cmap.items()  # type: ignore
        }

        topo_map = TreeTopology.av_overlay(sample.fundus.image, sample.art_topology, sample.vei_topology)
        m[2].add_image(topo_map, name="background")
        draw_trees((sample.art_topology.tree, sample.vei_topology.tree), view=m[2], bspline_dir=True)
        return m, sample, sample_data

    def show_tree_diff(
        self,
        idx: int | str,
        parent_pred: Int1DArray,
        dir_pred: Bool1DArray,
        fp_pred: Optional[Bool1DArray] = None,
        av_pred: Optional[Bool1DArray] = None,
        *,
        show_gt_graph: bool = True,
        version: Optional[str] = None,
        gt_digraph: Optional[VBranchDigraph] = None,
        branch_label: bool = False,
    ) -> tuple[Mosaic, VTree]:
        from ...utils.jppype import AV_COLORS, Mosaic, draw_tree, draw_trees

        if isinstance(idx, str):
            if version is None:
                idx, version = idx.split("/", 1)

        sample = self.get_sample(idx, discard_gt_tree=False)
        if gt_digraph is None:
            _, gt_digraph = self.get(idx, return_digraph=True, version=version)
        assert VBranchDigraph.has_all_p(gt_digraph)

        m = Mosaic(
            3 if show_gt_graph else 2,
            cols_titles=[f"Reference Topology: {sample.name}", "Predicted Tree"]
            + (["GT Tree"] if show_gt_graph else []),
            cell_height=700,
            background=sample.fundus.image,
        )

        B = len(parent_pred)

        # Draw GT tree
        solved_tree = gt_digraph.optimize_tree(keep_missing_branch=True)
        if show_gt_graph:
            draw_tree(
                solved_tree,
                view=m[2],
                branch_color="subtree",
                bspline_dir=True,
                interactive=True,
            )

        # Draw Predicted tree
        tree = gt_digraph.compute_tree_from_arborescence(parent_pred, dir_pred, fp_pred, keep_missing_branch=True)
        branch_dir_cmap = {}
        for b in range(gt_digraph.branch_count):
            if tree.branch_dirs(b) != gt_digraph.branch_dir[b] and not gt_digraph.branch_fp()[b]:
                branch_dir_cmap[b] = "#d2ff1d"
        for b in range(gt_digraph.branch_count, tree.branch_count):
            branch_dir_cmap[b] = AV_COLORS[AVLabel.BKG]
        if av_pred is not None:
            color_legend = {
                (True, 1): AV_COLORS[AVLabel.ART],
                (False, 2): AV_COLORS[AVLabel.VEI],
                (True, 2): "#fc249b",
                (False, 1): "#1c94e3",
                (True, 0): "white",
                (False, 0): "white",
            }
            branch_cmap = {i: color_legend[(av, gt_digraph.branch_av_class()[i])] for i, av in enumerate(av_pred)}
            if fp_pred is not None:
                for i in np.where(fp_pred)[0]:
                    branch_cmap[i] = AV_COLORS[AVLabel.BKG]
            for i in range(B, tree.branch_count):
                if tree.branch_tree[i] >= 0:
                    branch_cmap[i] = branch_cmap[tree.branch_tree[i]]
                else:
                    branch_cmap[i] = AV_COLORS[AVLabel.BKG]
        node_cmap = {}
        for node in tree.nodes():
            if node.out_degree <= 0:
                node_cmap[node.id] = branch_cmap[node.incoming_branch_ids[0]] if node.in_degree != 0 else "grey"
            else:
                for b in node.outgoing_branch_ids:
                    if b >= B or gt_digraph.branch_fp()[b] or (fp_pred is not None and fp_pred[b]):
                        continue
                    parent = tree.branch_tree[b]
                    while parent >= B:
                        parent = tree.branch_tree[parent]
                    gt_parent = solved_tree.branch_tree[b]
                    while gt_parent >= B:
                        gt_parent = solved_tree.branch_tree[gt_parent]
                    if gt_parent != parent:
                        node_cmap[node.id] = "#d2ff1d"
                        break
                else:
                    node_cmap[node.id] = branch_cmap[node.outgoing_branch_ids[0]]

        draw_tree(
            tree,
            view=m[1],
            branch_color=branch_cmap,
            edge_labels=branch_label,
            node_labels=False,
            node_cmap=node_cmap,
            interactive=True,
            bspline_dir=branch_dir_cmap,
        )

        topo_map = TreeTopology.av_overlay(sample.fundus.image, *sample.target_topologies)
        m[0].add_image(topo_map, name="background")
        draw_trees((sample.art_topology.tree, sample.vei_topology.tree), view=m[0], bspline_dir=True)

        return m, tree

    @classmethod
    def concatenate(cls, datasets: Sequence[BranchDigraphDataset]) -> BranchDigraphDataset:
        """Concatenate multiple datasets into a single dataset."""
        assert len(datasets) > 0, "At least one dataset must be provided for concatenation"
        dataset_out = datasets[0].copy()
        for d in datasets[1:]:
            dataset_out.samples_info.extend(d.samples_info)
            if dataset_out._preloaded_samples is not None:
                samples = d._preloaded_samples
                if samples is None:
                    samples = [d.samples_info[i].load() for i in range(len(d))]
                dataset_out._preloaded_samples.extend(samples)
        return dataset_out

    def split(self, indices: Sequence[int]) -> BranchDigraphDataset:
        """Create a new dataset with only the samples at the specified indices."""
        dataset = self.copy()
        dataset.samples_info = [self.samples_info[i] for i in indices]
        if self._preloaded_samples is not None:
            dataset._preloaded_samples = [self._preloaded_samples[i] for i in indices]
        return dataset

    def split_by_dataset(self) -> dict[str, BranchDigraphDataset]:
        """Split the dataset into subsets based on the dataset name annotation in the samples, and return a dict mapping dataset names to the corresponding subsets."""  # noqa: E501
        subsets = {}
        for i, sample_info in enumerate(self.samples_info):
            subsets.setdefault(sample_info.dataset, []).append(i)
        return {dataset: self.split(indices) for dataset, indices in subsets.items()}

    def select_dataset(self, dataset: str | Sequence[str]) -> BranchDigraphDataset:
        """Select a subset of the dataset with the specified dataset name annotation in the samples, and return a new dataset containing only the selected samples."""  # noqa: E501
        if isinstance(dataset, str):
            dataset = [dataset]
        indices = [i for i, sample_info in enumerate(self.samples_info) if sample_info.dataset in dataset]
        return self.split(indices)

    def split_sets(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        *,
        rng_seed: Optional[int] = None,
        cfg: Optional[BranchDigraphDatasetConfig] = None,
    ) -> tuple[BranchDigraphDataset, BranchDigraphDataset, BranchDigraphDataset]:
        """Split the dataset into train, validation and test sets and return corresponding DataLoaders."""
        assert train_ratio + val_ratio < 1.0, "train_ratio and val_ratio must sum to less than 1.0"

        train_indices = []
        val_indices = []
        test_indices = []

        # === Affect samples with a dataset type annotation ===
        annotated_samples = np.zeros(len(self.samples_info), dtype=bool)
        for i, sample in enumerate(self.samples_info):
            if sample.dataset_type is not None:
                annotated_samples[i] = True
                match sample.dataset_type:
                    case "train":
                        train_indices.append(i)
                    case "validation":
                        val_indices.append(i)
                    case "test":
                        test_indices.append(i)

        # === Randomly split remaining samples ===
        rs = RandomState(MT19937(SeedSequence(if_none(rng_seed, 123456))))
        remaining_indices = np.where(~annotated_samples)[0]
        _lookup = {}
        samples_subset = [_lookup.setdefault(self.samples_info[i].dataset, len(_lookup)) for i in remaining_indices]
        for _, samples_idx in np_group_by(remaining_indices, np.array(samples_subset)):
            rs.shuffle(samples_idx)
            num_samples = len(samples_idx)
            train_end = int(train_ratio * num_samples)
            val_end = int((train_ratio + val_ratio) * num_samples)
            train_indices.extend(train_idx := samples_idx[:train_end])
            val_indices.extend(val_idx := samples_idx[train_end:val_end])
            test_indices.extend(test_idx := samples_idx[val_end:])
            # Save split annotation for reproducibility
            for i in train_idx:
                self.samples_info[i].dataset_type = "train"
            for i in val_idx:
                self.samples_info[i].dataset_type = "validation"
            for i in test_idx:
                self.samples_info[i].dataset_type = "test"

        train_dataset = self.split(train_indices)
        val_dataset = self.split(val_indices)
        test_dataset = self.split(test_indices)
        for test_set in (val_dataset, test_dataset):
            test_set.cfg.augment = False

        return train_dataset, val_dataset, test_dataset

    def set_dataset_type(self, dataset_type: DatasetType | dict[str | Int1DArrayLike, DatasetType]) -> None:
        """Set the dataset type annotation for the samples in the dataset. This can be used to annotate samples as belonging to the train, validation or test set, which can then be used for splitting the dataset with split_loaders().

        Parameters
        ----------
        dataset_type : DatasetType or dict[str, DatasetType]
            The dataset type annotation to set for the samples. If a single DatasetType is provided, it will be applied to all samples. If a dict is provided, the keys should be dataset names and the values should be the corresponding DatasetType annotations. Samples with a dataset name not in the dict will not be annotated.
        """  # noqa: E501
        if isinstance(dataset_type, dict):
            for samples, type in dataset_type.items():
                if isinstance(samples, str):
                    for sample in self.samples_info:
                        if sample.dataset == samples:
                            sample.dataset_type = type
                else:
                    samples_idx = np.asarray(samples)
                    for i in samples_idx:
                        self.samples_info[i].dataset_type = type
        else:
            for sample in self.samples_info:
                sample.dataset_type = dataset_type
        self.save_manifest()

    def save_manifest(self) -> None:
        SampleInfo.encode(self.samples_info, save=Path(self.processed_dir), root_path=Path(self.processed_dir))

    @classmethod
    def discover_paths(
        cls,
        fundus_dir: Path,
        target_topology_dir: Path,
        graph_dir: Path | dict[str, Path],
        od_dir: Optional[Path] = None,
        macula_dir: Optional[Path] = None,
        *,
        dataset: str = "",
        fundus_ext: str | None = None,
        av_ext: str | None = None,
        od_ext: str | None = None,
        macula_ext: str | None = None,
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
                graphes.setdefault(file.stem, {}).update({graph_type: file})

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

        if od_dir is None:
            od_dir = fundus_dir.parent / "1-od"
        if od_ext is None:
            od_ext = most_common_image_ext(od_dir) if od_dir.exists() else ".png"

        if macula_dir is None:
            macula_dir = fundus_dir.parent / "1-macula"
        if macula_ext is None:
            macula_ext = most_common_image_ext(macula_dir) if macula_dir.exists() else ".png"

        return [
            SampleSource.from_paths(
                fundus_path=fundus_dir / f"{name}{fundus_ext}",
                target_topology_stem=target_topology_dir / name,
                graphes_path=graphes[name],
                dataset=dataset,
                od_path=od_dir / f"{name}{od_ext}",
                macula_path=macula_dir / f"{name}{macula_ext}",
            )
            for name in filenames
        ]

    def bundle(self, output_archive: Path | str, overwrite: bool = False) -> None:
        """Bundle the processed dataset into a tar.gz archive for easier storage and sharing.

        Parameters
        ----------
        output_archive : Path
            Path to save the output archive. Should end with .tar.gz or .tar.
        """
        output_archive = Path(output_archive)
        if not output_archive.name.endswith((".tar.gz", ".tar")):
            raise ValueError("Output archive should have .tar.gz or .tar extension")

        if output_archive.exists():
            if overwrite:
                output_archive.unlink()
            else:
                raise FileExistsError(
                    f"Output archive {output_archive} already exists. Set overwrite=True to overwrite it."
                )

        with tarfile.open(output_archive, "w") as tar:
            # Add sample info CSV
            with tempfile.NamedTemporaryFile(suffix=".json") as tmp_file:
                SampleInfo.encode(self.samples_info, save=tmp_file.name, root_path=Path(self.processed_dir))
                tar.add(tmp_file.name, arcname=SampleInfo.MANIFEST_FILENAME)

            # Add processed files
            for sample in self.samples_info:
                for file in sample.all_files():
                    tar.add(file, arcname=str(file.relative_to(self.processed_dir)))
