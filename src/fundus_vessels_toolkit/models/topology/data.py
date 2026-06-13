from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Self, TypeGuard, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch_geometric.data import Data as PygData
from torch_geometric.typing import OptTensor

from fundus_toolkits import FundusData
from fundus_toolkits.utils.geometric import Point, Rect

from ...segment_to_graph.geometry_parsing import populate_tangent
from ...segment_to_graph.vbranch_digraph import (
    TreeTopology,
    VBranchDigraph,
    VGraph,
    _VBranchDigraphWithAVProba,
)
from ...utils import if_none
from ...utils.tree import tree_connected_components
from ...vascular_data_objects import VBranchGeoData
from ...vascular_data_objects.vgeometric_data import VGeometricData
from .data_augmentation import AugmentationCfg, deteriorate_graph


class BranchDigraphData(PygData):
    img: Tensor
    """Fundus image as a tensor of shape (3, H, W)."""

    od_yx: Tensor
    """(y, x) coordinates of the optic disc center as a tensor of shape (2,)."""

    mac_yx: Tensor
    """(y, x) coordinates of the macula center as a tensor of shape (2,)."""

    vnode_count: int
    """Number of nodes in the vascular graph."""

    vnode_coord: Tensor
    """(y, x) coordinates of the vascular nodes as a tensor of shape (N, 2)."""

    branch_nodes: Tensor
    """Indices of the two tip nodes of each branch in the graph, as a tensor of shape (B, 2)."""

    branch_curves: Tensor
    """Branch curves with a constant N number of points, as a tensor of shape (B, N, 2)."""

    branch_root_candidates: Tensor
    """Valid vascular roots as a boolean tensor of shape (B, 2) indicating for each branch tip whether it is a valid root."""  # noqa: E501

    edge_index: Tensor  # type: ignore[assignment]+
    """Edges list as a (2, E) tensor, storing the indices of the source and target branches."""

    edge_dir: Tensor
    """The required direction of the source and target branches for each edge as a boolean tensor of shape (E, 2). For each edge, the first column indicates whether the source branch should be oriented from its first tip to its second tip (True) or the contrary (False), and the second column indicates the same for the target branch."""  # noqa: E501

    branch_tip_pos: Tensor | None
    """Ground truth (y, x) coordinates of the branch tips as a tensor of shape (B, 2, 2)."""

    branch_tip_tan: Tensor | None
    """Ground truth tangent vectors of the branch tips as a tensor of shape (B, 2, 2)."""

    branch_tip_calibre: Tensor | None
    """Ground truth calibres of the branch tips as a tensor of shape (B, 2)."""

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

    edge_attr: None

    BRANCH_ATTR = {
        "branch_nodes",
        "branch_curves",
        "branch_root_candidates",
        "branch_tip_pos",
        "branch_tip_tan",
        "branch_tip_calibre",
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
        vnode_coord: Tensor = None,  # type: ignore
        branch_nodes: Tensor = None,  # type: ignore
        branch_curves: list[Tensor] = None,  # type: ignore
        branch_root_candidates: Tensor = None,  # type: ignore
        edge_index: Tensor = None,  # type: ignore
        edge_dir: Tensor = None,  # type: ignore
        branch_tip_pos: Optional[Tensor] = None,  # type: ignore
        branch_tip_tan: Optional[Tensor] = None,  # type: ignore
        branch_tip_calibre: Optional[Tensor] = None,  # type: ignore
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
        branch_tip_pos: Tensor, optional
            Ground truth (y, x) coordinates of the branch tips as a tensor of shape (B, 2, 2).
        branch_tip_tan: Tensor, optional
            Ground truth tangent vectors of the branch tips as a tensor of shape (B, 2, 2).
        branch_tip_calibre: Tensor, optional
            Ground truth calibres of the branch tips as a tensor of shape (B, 2).
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

            assert vnode_coord.shape == (vnode_count, 2), (
                f"vnode_coord must be of shape ({vnode_count}, 2) but got {vnode_coord.shape}"
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
            assert branch_tip_pos is None or branch_tip_pos.shape == (B, 2, 2), (
                f"branch_tip_pos must be of shape (B, 2, 2) but got {branch_tip_pos.shape}"
            )
            assert branch_tip_tan is None or branch_tip_tan.shape == (B, 2, 2), (
                f"branch_tip_tan must be of shape (B, 2, 2) but got {branch_tip_tan.shape}"
            )
            assert branch_tip_calibre is None or branch_tip_calibre.shape == (B, 2), (
                f"branch_tip_calibre must be of shape (B, 2) but got {branch_tip_calibre.shape}"
            )

            # --- Edge attributes ---
            assert edge_index.ndim == 2 and edge_index.shape[0] == 2, (
                f"edge_index must be of shape (2, E) but got {edge_index.shape}"
            )
            E = edge_index.shape[1]
            assert edge_dir.shape == (E, 2), f"edge_dir must be of shape (E, 2) but got {edge_dir.shape}"
            assert edge_dir.dtype == torch.bool, "edge_dir must be a boolean tensor"
            assert edge_p is None or edge_p.shape == (E,), f"edge_p must be of shape (E,) but got {edge_p.shape}"

            # === PREPROCESSING ===
            curves_ = torch.full((B, 20, 2), -1.0)
            pos = []
            for i, curve in enumerate(branch_curves):
                C = curve.shape[0]
                pos.append(curve[C // 2])
                if C == 1:
                    curves_[i, :] = curve[0]
                else:
                    # Bilinear resampling of curve index
                    idx = torch.linspace(0, C - 1, 20)
                    low_idx = torch.floor(idx).long()
                    high_idx = torch.minimum(low_idx + 1, torch.tensor(C - 1))
                    alpha = (idx - low_idx).unsqueeze(1)
                    curves_[i] = (1 - alpha) * curve[low_idx] + alpha * curve[high_idx]

            pos = torch.stack(pos, dim=0)
        else:
            B = 0
            pos = None
            curves_ = None

        super().__init__(
            edge_index=edge_index,
            edge_dir=edge_dir,
            branch_nodes=branch_nodes,
            branch_curves=curves_,
            branch_root_candidates=branch_root_candidates,
            branch_tip_pos=branch_tip_pos,
            branch_tip_tan=branch_tip_tan,
            branch_tip_calibre=branch_tip_calibre,
            edge_p=edge_p,
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
            vnode_coord=vnode_coord,
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

    @property
    def branch_count(self) -> int:
        return self.branch_nodes.shape[0]

    @classmethod
    def from_branch_digraph(
        cls,
        digraph: VBranchDigraph,
        fundus_img: Tensor | npt.NDArray,
        od_yx: npt.NDArray,
        mac_yx: npt.NDArray,
        name: str,
    ) -> Self:
        # assert VBranchDigraph.has_all_p(digraph), "branch_digraph must have branch_fp_p and branch_av_p"
        assert digraph.graph is not None, "branch_digraph must have graph constructed"
        # with watch("BranchDigraphData.from_branch_digraph") as p:
        if isinstance(fundus_img, np.ndarray):
            fundus_img = torch.from_numpy(fundus_img)

        not_root = ~digraph.root_mask
        geodata = digraph.graph.geometric_data().copy()
        branch_curves = [
            torch.from_numpy(curve).float() for curve in geodata.branch_curve(fill_with_nodes=True, min_length=2)
        ]

        branch_tip_pos = geodata.tip_coord()
        branch_tip_tan = geodata.tip_tangent() if geodata.has_branch_data(VBranchGeoData.Fields.TIPS_TANGENT) else None
        CAL = VBranchGeoData.Fields.TIPS_CALIBRE
        branch_tip_calibre = geodata.tip_data(CAL) if geodata.has_branch_data(CAL) else None

        valid_root_tips = np.zeros((digraph.branch_count, 2), dtype=np.bool_)
        valid_root_tips[digraph.b1[digraph.root_mask], digraph.b1_tip[digraph.root_mask]] = True

        @dataclass(frozen=True)
        class GTInfo:
            edge_p: OptTensor = None
            branch_fp_p: OptTensor = None
            branch_av_p: OptTensor = None
            branch_dir_p: OptTensor = None
            branch_root_p: OptTensor = None
            branch_subtree_idx: OptTensor = None

        if VBranchDigraph.has_all_p(digraph):
            root_p = np.zeros((digraph.branch_count, 2), dtype=np.float32)
            root_p[digraph.b1[digraph.root_mask], digraph.b1_tip[digraph.root_mask]] = digraph.line_p[digraph.root_mask]
            branch_subtree_idx = tree_connected_components(torch.from_numpy(digraph.max_parent()))
            gt_info = GTInfo(
                edge_p=torch.from_numpy(digraph.line_p[not_root]).float(),
                branch_fp_p=torch.from_numpy(digraph.branch_fp_p).float(),
                branch_av_p=torch.from_numpy(digraph.branch_av_p).float(),
                branch_dir_p=torch.from_numpy(digraph.branch_dir_p).float(),
                branch_root_p=torch.from_numpy(root_p).float(),
                branch_subtree_idx=branch_subtree_idx.int(),
            )
        else:
            gt_info = GTInfo()

        return cls(
            img=fundus_img,
            od_yx=torch.from_numpy(od_yx).float(),
            mac_yx=torch.from_numpy(mac_yx).float(),
            vnode_count=digraph.graph.node_count,
            vnode_coord=torch.from_numpy(geodata.node_coord()).float(),
            edge_index=torch.from_numpy(digraph.b0b1[not_root]).T,
            edge_dir=torch.from_numpy(digraph.b0b1_dir[not_root]),
            branch_nodes=torch.from_numpy(digraph.graph.branch_list).int(),
            branch_curves=branch_curves,
            branch_root_candidates=torch.from_numpy(valid_root_tips),
            branch_tip_pos=torch.from_numpy(branch_tip_pos).float() if branch_tip_pos is not None else None,
            branch_tip_tan=torch.from_numpy(branch_tip_tan).float() if branch_tip_tan is not None else None,
            branch_tip_calibre=torch.from_numpy(branch_tip_calibre).float() if branch_tip_calibre is not None else None,
            **asdict(gt_info),
            name=name,
        )

    @overload
    @classmethod
    def from_graph(
        cls,
        graph: VGraph,
        fundus: FundusData | npt.NDArray,
        gt_topology: Optional[tuple[TreeTopology, TreeTopology]] = None,
        *,
        return_digraph: Literal[False] = False,
        augment: bool | AugmentationCfg = False,
        name: Optional[str] = None,
        od_center: Optional[Point] = None,
        mac_center: Optional[Point] = None,
    ) -> Self: ...
    @overload
    @classmethod
    def from_graph(
        cls,
        graph: VGraph,
        fundus: FundusData | npt.NDArray,
        gt_topology: Optional[tuple[TreeTopology, TreeTopology]] = None,
        *,
        return_digraph: Literal[True],
        augment: bool | AugmentationCfg = False,
        name: Optional[str] = None,
        od_center: Optional[Point] = None,
        mac_center: Optional[Point] = None,
    ) -> tuple[Self, VBranchDigraph]: ...
    @classmethod
    def from_graph(
        cls,
        graph: VGraph,
        fundus: FundusData | npt.NDArray,
        gt_topology: Optional[tuple[TreeTopology, TreeTopology]] = None,
        *,
        return_digraph: bool = False,
        augment: bool | AugmentationCfg = False,
        name: Optional[str] = None,
        od_center: Optional[Point] = None,
        mac_center: Optional[Point] = None,
    ) -> Self | tuple[Self, VBranchDigraph]:
        """Alternative constructor to create a BranchDigraphData from a VGraph and a fundus image. Note that this method will not be able to fill all the fields of the data, especially those related to the ground truth probabilities and the branch curves, which are not stored in the VGraph."""  # noqa: E501
        # with watch("BranchDigraphData.from_graph") as p:
        #    with p.sub("parse cfg"):
        augment_opts = AugmentationCfg.parse(augment)

        #    with p.sub("graph preprocessing"):
        graph = graph.copy()
        graph.clear_all_branch_attr()
        graph.clear_all_branch_attr()
        if augment_opts.deteriorate_graph:
            #        with p.sub("graph deterioration"):
            graph = deteriorate_graph(graph, opts=augment_opts.deterioration_opts, inplace=True)

        #    with p.sub("VBranchDigraph.from_graph"):
        branch_digraph = VBranchDigraph.from_graph(graph, check=False)
        if gt_topology is not None:
            # with p.sub("compute_p_from_gt"):
            branch_digraph.compute_p_from_gt(*gt_topology, check=False)

        # with p.sub("read fundus and preprocess"):
        if isinstance(fundus, FundusData):
            fundus_img = fundus.image
            if od_center is None and fundus.has_od_center:
                od_center = fundus.od_center
            if mac_center is None:
                mac_center = fundus.inferred_macula_center()
        else:
            fundus_img = fundus
        fundus_shape = (fundus_img.shape[1], fundus_img.shape[2])

        if od_center is None:
            od_center = Point.from_tuple(fundus_shape) // 2
            if mac_center is None:
                mac_center = Point(fundus_shape[0] // 2, fundus_shape[1])  # Dummy position on the right of the OD
        elif mac_center is None:
            if od_center.x < fundus_shape[1] // 2:
                mac_center = Point(od_center.y, od_center.x + fundus_shape[1] // 2)
            else:
                mac_center = Point(od_center.y, od_center.x - fundus_shape[1] // 2)

        branch_digraph.graph.geometric_data().clear_attribute(all_except="CALIBRE")
        if augment_opts.geometric:
            # with p.sub("Geometric Augmentation") as p_aug:
            # with p_aug.sub("generate transform"):
            t = augment_opts.generate_transform(shape=fundus_shape)
            # with p_aug.sub("transform graph"):
            branch_digraph.graph.transform(t, warped_domain="same", inplace=True)
            # with p_aug.sub("warp fundus"):
            fundus_img, _ = t.warp(fundus_img.transpose((1, 2, 0)), warped_domain="same")
            fundus_img = fundus_img.transpose((2, 0, 1))
            # with p_aug.sub("transform OD and macula centers"):
            od_yx, mac_yx = t.transform(np.array([od_center, mac_center]))
        else:
            od_yx, mac_yx = od_center.numpy(), mac_center.numpy()
            # with p.sub("recompute tangents"):
        populate_tangent(branch_digraph.graph, tips=True, inplace=True)

        data = cls.from_branch_digraph(
            digraph=branch_digraph,
            fundus_img=fundus_img,
            od_yx=od_yx,
            mac_yx=mac_yx,
            name=if_none(name, "graph_based_sample"),
        )
        return (data, branch_digraph) if return_digraph else data

    @overload
    def to_digraph(self, *, graph: bool = False, gt_proba: Literal[True]) -> _VBranchDigraphWithAVProba: ...
    @overload
    def to_digraph(self, *, graph: bool = False, gt_proba: Optional[Literal[False]] = None) -> VBranchDigraph: ...
    def to_digraph(self, *, graph: bool = False, gt_proba: Optional[bool] = None) -> VBranchDigraph:
        """Convert the data back to a VBranchDigraph.

        Parameters
        ----------
        graph : bool, optional
            Whether to construct the VGraph and set it in the digraph.graph attribute. Note that the geometric data of the graph will be filled with the branch curves stored in the data, which may not be very accurate as they are just sampled points on the original branch curve. (By default False).

        gt_proba : bool or None, optional
            Whether to set the edge_p, branch_fp_p, branch_av_p and branch_dir_p attributes of the digraph based on the data.
            - If None (by default), the probabilities will be set if they are present in the data, and not set otherwise.
            - If True, the probabilities will be set and an error will be raised if they are not present in the data.
            - If False, the probabilities will not be set regardless of whether they are present in the data or not.

        """  # noqa: E501
        digraph = self.lines.to_digraph()
        if graph:
            geodata = VGeometricData(
                nodes_coord=self.vnode_coord.numpy(force=True),
                branches_curve=[curve.numpy(force=True) for curve in self.branch_curves],
                domain=Rect.from_size(self.img.shape[-2:]),  # type: ignore
            )
            digraph.graph = VGraph(
                branch_list=self.branch_nodes.numpy(force=True), geometric_data=geodata, check_integrity=True
            )
        if BranchDigraphData.has_gt(self) and gt_proba is not False:
            digraph.line_p = self.line_p.numpy(force=True)
            digraph.branch_fp_p = self.branch_fp_p.numpy(force=True)
            digraph.branch_av_p = self.branch_av_p.numpy(force=True)
            digraph.branch_dir_p = self.branch_dir_p.numpy(force=True)
        elif gt_proba is True:
            raise ValueError("Data instance has no ground truth probabilities but gt_proba is set to True.")
        return digraph

    @property
    def edge_lines(self):
        return DigraphLines(self.edge_index, self.edge_dir, self.branch_nodes)

    @property
    def lines(self) -> DigraphLines:
        """The concatenation of edge lines (linking two branches) and root lines (linking the virtual root node to a branch). The root lines are added based on the valid root candidates indicated in the batch data, and their score is given by the root affinity score of their target branch."""  # noqa: E501
        root_branch, root_tip = torch.where(self.branch_root_candidates)
        root_lines = torch.stack([-torch.ones_like(root_branch), root_branch], dim=0)
        root_dir = root_tip == 0
        root_dir = torch.stack([torch.zeros_like(root_dir), root_dir], dim=-1)
        return DigraphLines(
            edge_index=torch.cat([self.edge_index, root_lines], dim=1),
            edge_dir=torch.cat([self.edge_dir, root_dir], dim=0),
            branch_nodes=self.branch_nodes,
        )

    @property
    def line_p(self) -> Optional[Tensor]:
        """Ground truth probability of each line (edge or root) being correct. For edge lines, the probability is given by the edge_p attribute of the data, while for root lines, the probability is given by the branch_root_p attribute of the target branch and tip of the root line."""  # noqa: E501
        if self.edge_p is None or self.branch_root_p is None:
            return None
        root_branch, root_tip = torch.where(self.branch_root_candidates)
        root_line_p = self.branch_root_p[root_branch, root_tip]
        return torch.cat([self.edge_p, root_line_p], dim=0)

    @classmethod
    def has_gt(cls, instance: Self) -> TypeGuard[_BranchDigraphDataWithGT]:
        """Check if the data instance has ground truth probabilities (i.e. if edge_p, branch_fp_p, branch_av_p and branch_dir_p are not None)."""  # noqa: E501
        return _BranchDigraphDataWithGT.check(instance)


@dataclass(frozen=True)
class DigraphLines:
    edge_index: Tensor
    edge_dir: Tensor
    branch_nodes: Tensor
    mask: Optional[Tensor] = None
    whole_mask: Optional[Tensor] = None

    @classmethod
    def from_batch(cls, batch: BranchDigraphBatch):
        return cls(edge_index=batch.edge_index, edge_dir=batch.edge_dir, branch_nodes=batch.branch_nodes)

    def to_digraph(self) -> VBranchDigraph:
        line_list = np.empty((len(self), 4), dtype=np.int64)
        line_list[:, 0] = self.b0.numpy(force=True)
        line_list[:, 1] = self.b0_dir.numpy(force=True).astype(np.int64)
        line_list[:, 2] = self.b1.numpy(force=True)
        line_list[:, 3] = 1 - self.b1_dir.numpy(force=True)
        return VBranchDigraph(line_list=line_list, branch_count=self.branch_nodes.shape[0])

    def __bool__(self):
        return self.edge_index.shape[1] > 0

    def __len__(self):
        return self.edge_index.shape[1]

    def __getitem__(self, idx):
        if self.whole_mask is None:
            whole_mask = torch.zeros(self.edge_index.shape[1], dtype=torch.bool, device=self.edge_index.device)
            whole_mask[idx] = True
        else:
            whole_mask = torch.zeros_like(self.whole_mask)
            whole_mask[self.whole_mask][idx] = True
        return DigraphLines(
            edge_index=self.edge_index[:, idx],
            edge_dir=self.edge_dir[idx],
            branch_nodes=self.branch_nodes,
            mask=self.mask[idx] if self.mask is not None else None,
            whole_mask=whole_mask,
        )

    @property
    def n_lines(self) -> int:
        """Number of lines"""
        return self.edge_index.shape[1]

    @property
    def b0(self) -> Tensor:
        """Integer tensor of shape (N_line,) containing the index of the source branch of each line"""
        return self.edge_index[0]

    @property
    def b1(self) -> Tensor:
        """Integer tensor of shape (N_line,) containing the index of the target branch of each line"""
        return self.edge_index[1]

    @property
    def b0_dir(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating for each line the required direction of its source branch (True: from tip0 to tip1, False: from tip1 to tip0)"""  # noqa: E501
        return self.edge_dir[:, 0]

    @property
    def b1_dir(self) -> Tensor:
        """Boolean tensor of shape (N_line,) indicating for each line the required direction of its target branch (True: from tip0 to tip1, False: from tip1 to tip0)"""  # noqa: E501
        return self.edge_dir[:, 1]

    @property
    def b1_node(self) -> Tensor:
        """Integer tensor of shape (N_line, 2) containing the index of the tail node for each target branch."""
        return torch.gather(self.branch_nodes[self.b1], 1, 1 - self.b1_dir[:, None].long()).squeeze()

    @property
    def edge_tip(self) -> Tensor:
        """Tensor of shape (N_line, 2) containing for each line the indices of the tips of its source and target branches involved in the line, ordered from tip0 to tip1. For example, if a line links the tip0 of branch A to the tip1 of branch B, the corresponding edge_tip will be [0, 1]."""  # noqa: E501
        return torch.stack([~self.edge_dir[:, 0], self.edge_dir[:, 1]], dim=-1).int()

    @property
    def tip0(self) -> Tensor:
        """Int tensor of shape (N_line,) indicating for each line the tip indices of the source branches."""  # noqa: E501
        return (~self.edge_dir[:, 0]).int()

    @property
    def tip1(self) -> Tensor:
        """Int tensor of shape (N_line,) indicating for each line the tip indices of the target branches."""  # noqa: E501
        return self.edge_dir[:, 1].int()


########################################################################################################################
# === TYPE CHECKING UTILITIES ===
########################################################################################################################
class _BranchDigraphDataWithGT(BranchDigraphData):
    """Utility class for typechecking to ensure that the data has ground truth probabilities."""

    edge_p: Tensor
    line_p: Tensor
    branch_root_p: Tensor
    branch_fp_p: Tensor
    branch_av_p: Tensor
    branch_dir_p: Tensor
    branch_subtree_idx: Tensor

    @classmethod
    def check(cls, inst: BranchDigraphData) -> TypeGuard[Self]:
        return (
            inst.edge_p is not None
            and inst.branch_root_p is not None
            and inst.branch_fp_p is not None
            and inst.branch_av_p is not None
            and inst.branch_dir_p is not None
            and inst.branch_subtree_idx is not None
        )


class BranchDigraphBatch(BranchDigraphData):
    batch: Tensor  # type: ignore[assignment]
    """Tensor of shape (B,) containing for each branch the index of its graph in the batch."""

    batch_size: int
    """Number of graphs in the batch."""

    vnode_count: Tensor
    """Tensor of shape (batch_size,) containing the number of vascular nodes for each graph in the batch."""

    @classmethod
    def has_gt(cls, instance: Self) -> TypeGuard[_BranchDigraphBatchWithGT]:
        """Check if the batch instance has ground truth probabilities (i.e. if edge_p, branch_fp_p, branch_av_p and branch_dir_p are not None)."""  # noqa: E501
        return _BranchDigraphBatchWithGT.check(instance)


class _BranchDigraphBatchWithGT(_BranchDigraphDataWithGT):
    """Utility class for typechecking to ensure that the batch has ground truth probabilities."""

    batch: Tensor  # type: ignore[assignment]
    """Tensor of shape (B,) containing for each branch the index of its graph in the batch."""

    batch_size: int
    """Number of graphs in the batch."""

    vnode_count: Tensor
    """Tensor of shape (batch_size,) containing the number of vascular nodes for each graph in the batch."""
