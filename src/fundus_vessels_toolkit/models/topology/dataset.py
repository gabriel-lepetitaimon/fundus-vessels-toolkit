from __future__ import annotations

import copy
import hashlib
import shutil
import tarfile
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, NotRequired, Optional, Self, Sequence, TypedDict, Unpack

import numpy as np
import numpy.typing as npt
import tqdm
from fundus_data_toolkit.functional import open_image
from joblib import Parallel, delayed
from numpy.random import MT19937, RandomState, SeedSequence
from pydantic import Field, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass
from torch_geometric.data import Dataset as PygDataset

from fundus_odmac_toolkit.models.segmentation import segment
from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.utils.data_io import most_common_image_ext, overwrite_or_newer
from fundus_toolkits.utils.geometric import Point, Rect
from fundus_toolkits.utils.image import read_image

from ...pipelines.avseg_to_tree import AVSegToTreeBase, GNNAVSegToTree
from ...segment_to_graph.graph_simplification import merge_nodes_by_distance
from ...segment_to_graph.vbranch_digraph import (
    TreeTopology,
    VGraph,
)
from ...utils import if_none
from ...utils.fundus_projections import ResizeTranslateProjection
from ...utils.numpy import np_group_by
from ...utils.typing import Bool1DArray, Int1DArray
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

    def compute_od_mac_center(self) -> Self:
        """Compute the optic disc and macula centers from the fundus image if they are not already provided."""
        if self.od_center is not None and self.macula_center is not None:
            return self

        fundus = FundusData(self.fundus)
        od_mac = segment(open_image(self.fundus)).numpy(force=True).argmax(axis=0)  # type: ignore
        fundus = fundus.update(od=od_mac == 1, macula=od_mac == 2, reshape_method="resize")
        if fundus.od_center is not None:
            self.od_center = fundus.od_center if not fundus.od_center.is_nan() else None
        if fundus.macula_center is not None:
            self.macula_center = fundus.macula_center if not fundus.macula_center.is_nan() else None
        return self

    def load(self, image: bool = True, discard_gt_tree: bool = True) -> BranchDigraphSample:
        """Load the sample from disk into memory."""
        fundus = read_image(self.fundus, cast_to_float=True)
        fundus = FundusData(image=fundus) if image else FundusData(shape=(fundus.shape[-2], fundus.shape[-1]))
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

    def prefix(self, prefix: Path) -> Self:
        """Return a new BranchDigraphSampleInfo with the paths prefixed by the given path."""
        return self.__class__(
            name=self.name,
            fundus=prefix / self.fundus,
            art_topology=prefix / self.art_topology,
            vei_topology=prefix / self.vei_topology,
            graphes={k: prefix / p for k, p in self.graphes.items()},
            od_center=self.od_center,
            macula_center=self.macula_center,
            date=self.date,
            dataset=self.dataset,
        )

    def relative_to(self, path: Path) -> Self:
        """Return a new BranchDigraphSampleInfo with the paths relative to the given path."""
        return self.__class__(
            name=self.name,
            fundus=self.fundus.relative_to(path),
            art_topology=self.art_topology.relative_to(path),
            vei_topology=self.vei_topology.relative_to(path),
            graphes={k: p.relative_to(path) for k, p in self.graphes.items()},
            od_center=self.od_center,
            macula_center=self.macula_center,
            date=self.date,
            dataset=self.dataset,
        )

    def all_files(self, relative_to: Optional[Path | str] = None) -> list[Path]:
        """Return a list of all files associated with this sample, optionally relative to a given path."""

        paths = [self.fundus, self.art_topology, self.vei_topology] + list(self.graphes.values())
        return [p.relative_to(relative_to) for p in paths] if relative_to is not None else paths

    MANIFEST_FILENAME = "manifest.json"

    @classmethod
    def decode(cls, infos: bytes | list | Path | str) -> list[SampleInfo]:
        adapter = TypeAdapter(list[SampleInfo])

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
                    return adapter.validate_json(file.read())
            else:
                if infos.is_dir():
                    infos = infos / cls.MANIFEST_FILENAME
                if not infos.exists():
                    return []
                with open(infos, "rb") as f:
                    return adapter.validate_json(f.read())
        elif isinstance(infos, bytes):
            return adapter.validate_json(infos)
        else:
            return adapter.validate_python(infos)

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

    target_topologies: tuple[Path, Path]
    """Paths to the target topology files for the artery and vein trees. The files should be in a format loadable by VTree.load()."""  # noqa: E501

    graphes: dict[str, Path]
    """Paths to the graph files, with keys representing different versions of the graph (e.g. generated by different models). The files should be in a format loadable by VGraph.load() or an image parsable by AVSegToTreeBase."""  # noqa: E501

    date: datetime
    """Date of the sample, used for caching purposes. Should be the latest modification date among the source files."""

    dataset: str
    """Optional name of the dataset this sample belongs to."""

    @property
    def name(self) -> str:
        """Name of the sample, derived from the fundus image file name."""
        return self.fundus.stem

    @classmethod
    def from_paths(
        cls, fundus_path: Path, target_topology_stem: Path, graphes_path: dict[str, Path], dataset: str = ""
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
        """  # noqa: E501
        art_topo = Path(str(target_topology_stem) + ART_EXT)
        vei_topo = Path(str(target_topology_stem) + VEI_EXT)
        target_topologies = (art_topo, vei_topo)
        date = max(
            art_topo.stat().st_mtime, vei_topo.stat().st_mtime, max(p.stat().st_mtime for p in graphes_path.values())
        )
        return cls(
            fundus=fundus_path,
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

        return SampleInfo(
            name=name,
            fundus=fundus_path,
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
        output_dir: Path,
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

    def process(
        self,
        output_dir: Path,
        *,
        resize_to: Optional[int],
        av2tree: Optional[AVSegToTreeBase] = None,
        od_mac_center: bool | tuple[Point, Point] = True,
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

        resize_to : Optional[int]
            If specified, the fundus image will be resized to have the specified width, and the graphes and topologies will be transformed accordingly. If not specified, the original size will be kept.

        av2tree : Optional[AVSegToTreeBase]
            The method used to parse AV segmentation images to a graph. Only used if the graphes provided are image files instead of graph files.

        overwrite: Optional[bool]
            Whether to overwrite the processed files if they already exist. If None, the files will be overwritten if the source files are newer than the processed files, and kept otherwise.

        od_mac_center: bool | tuple[Point, Point]
            Whether to find the optic disc and macula centers from the fundus image and save them in the processed sample. If True, the centers will be found using segmentation. If a tuple of Points is provided, they will be used as the centers directly. If False, the centers will not be included in the processed sample.
        Returns
        -------
        BranchDigraphSampleInfo
            The information of the processed sample, including the paths to the processed files and the coordinates of the optic disc and macula centers.
        """  # noqa: E501
        output_paths = self.output_paths(output_dir)

        # === Load and crop fundus image ===
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

        # === Load, preprocess and save graphes ===
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
                graph = av2tree.to_vgraph(fundus.update(av=graph_path, crop_pad=roi, reshape_method="resize"))

            # Remove duplicated branches and nodes, and remove useless attributes
            graph.geometric_data().clear_attribute(all_except=GEO_ATTRS)
            merge_nodes_by_distance(graph, max_distance=0.5, inplace=True)
            if len(duplicates := graph.branch_duplicates()):
                graph.delete_branch([b for d in duplicates for b in d[1:]], inplace=True)

            # Save processed graph
            graph.save(graph_out_path, on_exists="overwrite")

        # === Load and preprocess GT topology ===
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

        if isinstance(od_mac_center, tuple):
            output_paths.od_center, output_paths.macula_center = od_mac_center
        elif od_mac_center is True:
            output_paths.compute_od_mac_center()

        return output_paths


class BranchDigraphDatasetArgs(TypedDict):
    graph_version: NotRequired[str | dict[str, float]]
    preload: NotRequired[bool | Literal["without-image"]]
    augment: NotRequired[bool | AugmentationOpts]
    verbose: NotRequired[bool]
    # line_p_smoothing: NotRequired[float] = 0.0


class BranchDigraphDataset(PygDataset):
    def __init__(
        self,
        src_path: Path,
        root: Optional[str | Path] = None,
        *,
        transform=None,
        **kwargs: Unpack[BranchDigraphDatasetArgs],
    ):
        """Dataset of branch digraphs for fundus images, with multiple graph versions and target topologies.

        Parameters
        ----------
        src_path : Path
            Path to the processed dataset directory or archive, which should contain an appropriate manifest along with the processed data.
        root : Optional[str | Path], optional
            Root directory to store the dataset.
        """  # noqa: E501

        self.src_path = src_path
        self.augment = kwargs.get("augment", False)
        self.graph_version = kwargs.get("graph_version", None)
        # self.line_p_smoothing = line_p_smoothing
        self.samples_info = SampleInfo.decode(src_path)
        self.verbose = kwargs.get("verbose", True)

        if root is None:
            if src_path.is_dir():
                root = str(src_path)
        else:
            root = str(root) if isinstance(root, Path) else root
        self.root_specified = root is not None
        super().__init__(root=root, transform=transform)

        if (preload := kwargs.get("preload", False)) is not False:
            self._preloaded_samples = [
                samples_info.load(image=preload != "without-image")
                for samples_info in tqdm.tqdm(self.samples_info, desc="Preloading dataset", disable=not self.verbose)
            ]
        else:
            self._preloaded_samples = None

    def copy(self) -> Self:
        """Return a copy of the dataset."""
        new = copy.copy(self)
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
        *,
        output_dir: Optional[Path | str] = None,
        resize_to: Optional[int] = None,
        overwrite: Optional[bool] = None,
        fundus_ext: str | None = None,
        av_ext: str | None = None,
        av2tree: Optional[AVSegToTreeBase] = None,
        ignore_recent: Optional[int | datetime] = None,
        **kwargs: Unpack[BranchDigraphDatasetArgs],
    ) -> Self:
        verbose = kwargs.get("verbose", True)
        # === List source files ===
        if isinstance(fundus_dir, list):
            N = len(fundus_dir)
            assert isinstance(target_topology_dir, list) and len(target_topology_dir) == N, (
                "If fundus_dir is a list, target_topology_dir must be a list of the same length"
            )
            assert isinstance(graph_dir, list) and len(graph_dir) == N, (
                "If fundus_dir is a list, graph_dir must be a list of the same length"
            )
            samples_src = []
            for f_dir, t_dir, g_dir in zip(fundus_dir, target_topology_dir, graph_dir, strict=True):
                samples_src.extend(
                    cls.discover_paths(
                        f_dir,
                        t_dir,
                        g_dir,
                        dataset=f_dir.name,
                        fundus_ext=fundus_ext,
                        av_ext=av_ext,
                        ignore_recent=ignore_recent,
                    )
                )
        else:
            assert not isinstance(target_topology_dir, list), (
                "If fundus_dir is not a list, target_topology_dir should not be a list"
            )  # noqa: E501
            assert not isinstance(graph_dir, list), "If fundus_dir is not a list, graph_dir should not be a list"  # noqa: E501
            samples_src = cls.discover_paths(
                fundus_dir,
                target_topology_dir,
                graph_dir,
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
        existing_timestamps = {s.name: s.date for s in existing_samples.values()}

        samples: list[SampleInfo] = []
        for i, sample_src in reversed(list(enumerate(samples_src))):
            existing_sample = sample_src.already_processed(output_dir, existing_samples, overwrite)
            if existing_sample is not None:
                samples.append(existing_sample)
                samples_src.pop(i)

        # Process remaining samples in parallel
        if len(samples_src) > 0:
            run_parallel = Parallel(n_jobs=-2, return_as="generator_unordered")
            sample_process = delayed(SampleSource.process)
            new_samples: list[SampleInfo] = [
                s
                for s in tqdm.tqdm(
                    run_parallel(
                        sample_process(
                            sample_src,
                            output_dir=output_dir,
                            resize_to=resize_to,
                            av2tree=av2tree,
                            od_mac_center=False,
                            overwrite=existing_timestamps.get(sample_src.name) if overwrite is None else overwrite,
                        )
                        for sample_src in samples_src
                    ),
                    total=len(samples_src),
                    desc="Processing samples",
                    disable=not verbose,
                )
            ]  # type: ignore

            for sample in tqdm.tqdm(new_samples, desc="Computing OD/Macula centers", disable=not verbose):
                if (existing_sample := existing_samples.get(sample.name)) is not None:
                    sample.od_center = existing_sample.od_center
                    sample.macula_center = existing_sample.macula_center
                else:
                    sample.compute_od_mac_center()

            samples.extend(new_samples)

        # Save samples info
        samples = sorted(samples, key=lambda s: s.name)
        if len(samples_src) > 0:
            SampleInfo.encode(samples, save=output_dir)

        return cls(output_dir, **kwargs)

    @property
    def processed_dir(self) -> str:
        """Directory where the processed data is stored. If the src_path is an archive, this will be a temporary directory where the archive is extracted."""  # noqa: E501
        return self.root if self.root_specified else super().processed_dir

    def process(self) -> None:
        """Process the raw data files and save the processed data to disk. This method should be called before using the dataset for training or evaluation."""  # noqa: E501
        processed_dir = Path(self.processed_dir)
        if self.src_path.name.endswith((".tar", ".tar.gz")):
            with tarfile.open(self.src_path, "r:*") as tar:
                tar.extractall(path=processed_dir)
        elif self.processed_dir != self.src_path:
            shutil.copytree(self.src_path, processed_dir, dirs_exist_ok=True)
        self.sample_info = [info.prefix(processed_dir) for info in self.samples_info]

    @property
    def processed_file_names(self):
        return [file for sample in self.samples_info for file in sample.all_files(relative_to=self.processed_dir)]

    def len(self):
        return len(self.samples_info)

    def get(self, idx: int | str) -> BranchDigraphData:
        sample = self.get_sample(idx)
        graphes = list(sample.graphes.values())
        versions = list(sample.graphes.keys())
        if isinstance(self.graph_version, dict):
            pick_p = [self.graph_version.get(v, 0.0) for v in versions]
            p_total = sum(pick_p)
            if p_total == 0.0:
                graph_version = versions[np.random.randint(len(graphes))]
            else:
                graph_version = versions[np.random.choice(len(graphes), p=np.array(pick_p) / p_total)]
        elif self.graph_version is None or self.graph_version not in sample.graphes:
            graph_version = versions[np.random.randint(len(graphes))]
        else:  # version is a valid key in sample.graphes
            graph_version = self.graph_version
        graph = sample.graphes[graph_version]

        return BranchDigraphData.from_graph(
            graph,
            sample.fundus,
            sample.target_topologies,
            augment=self.augment,
            name=sample.name + (f"/{graph_version}" if graph_version else ""),
            od_center=sample.fundus.od_center,
            mac_center=sample.fundus.macula_center,
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
        sample_data = self.get(idx)

        if "/" in sample_data.name:
            name, graph_version = sample_data.name.split("/", 1)
        else:
            name, graph_version = sample_data.name, ""

        m = Mosaic(
            3,
            cols_titles=[sample_data.name, "with GT Topology", "Ground Truth"],
            cell_height=700,
            background=sample_data.img,
        )
        draw_graph(
            sample.graphes[graph_version],
            view=m.views[0],
            edge="bspline",
            edge_labels=branch_label,
            node_labels=node_label,
        )

        digraph = sample_data.to_branch_digraph(graph=True)
        assert digraph.graph is not None

        m.views[0].add_graph([], nodes_yx=[sample_data.od_yx.numpy(), sample_data.mac_yx.numpy()], name="OD/Macula")
        m.views[0]["OD/Macula"].nodes_cmap = {0: "green", 1: "yellow"}  # type: ignore

        # draw_graph(
        #     digraph.graph,
        #     view=m.views[1],
        #     edge="bspline",
        #     edge_labels=branch_label,
        #     node_labels=node_label,
        # )
        solved_tree = digraph.optimize_tree(keep_missing_branch=True)
        draw_tree(solved_tree, view=m[1], branch_color="subtree", bspline_dir=True)

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

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        *,
        rng_seed: Optional[int] = None,
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

        return train_dataset, val_dataset, test_dataset

    def set_dataset_type(self, dataset_type: DatasetType | dict[str, DatasetType]) -> None:
        """Set the dataset type annotation for the samples in the dataset. This can be used to annotate samples as belonging to the train, validation or test set, which can then be used for splitting the dataset with split_loaders().

        Parameters
        ----------
        dataset_type : DatasetType or dict[str, DatasetType]
            The dataset type annotation to set for the samples. If a single DatasetType is provided, it will be applied to all samples. If a dict is provided, the keys should be dataset names and the values should be the corresponding DatasetType annotations. Samples with a dataset name not in the dict will not be annotated.
        """  # noqa: E501
        if isinstance(dataset_type, dict):
            for sample in self.samples_info:
                if sample.dataset in dataset_type:
                    sample.dataset_type = dataset_type[sample.dataset]
        else:
            for sample in self.samples_info:
                sample.dataset_type = dataset_type

    @classmethod
    def discover_paths(
        cls,
        fundus_dir: Path,
        target_topology_dir: Path,
        graph_dir: Path | dict[str, Path],
        *,
        dataset: str = "",
        fundus_ext: str | None = None,
        av_ext: str | None = None,
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

        return [
            SampleSource.from_paths(
                fundus_path=fundus_dir / f"{name}{fundus_ext}",
                target_topology_stem=target_topology_dir / name,
                graphes_path=graphes[name],
                dataset=dataset,
            )
            for name in filenames
        ]

    def bundle(self, output_archive: Path, overwrite: bool = False) -> None:
        """Bundle the processed dataset into a tar.gz archive for easier storage and sharing.

        Parameters
        ----------
        output_archive : Path
            Path to save the output archive. Should end with .tar.gz or .tar.
        """
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
