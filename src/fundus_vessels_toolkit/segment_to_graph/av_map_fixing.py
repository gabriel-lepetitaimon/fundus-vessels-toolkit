import numpy as np
import numpy.typing as npt
from skimage.segmentation import expand_labels

from fundus_vessels_toolkit.vascular_data_objects.vgraph import VGraph

from ..utils.rasterization import rasterize_topology
from ..vascular_data_objects.vbranch_geodata import VBranchGeoData
from ..vascular_data_objects.vtree import VTree


def fix_av_map(
    av_map: npt.NDArray[np.uint8],
    trees: tuple[VTree, VTree],
    expand_labels_by: int = 0,
    force_initial_segmentation: bool = True,
    discard_av: bool = False,
) -> npt.NDArray[np.uint8]:
    """
    Fix the AV classification to match the given trees. The vessel segmentation is not modified, only the AV classification.

    Parameters
    ----------
    av_map : npt.NDArray[np.uint8]
        The AV map to fix.

    trees : tuple[VTree, VTree]
        The arterioles and venules trees.

    expand_labels_by : int, optional
        The number of pixels to expand the labels by. Default is 0.

    Returns
    -------
    npt.NDArray[np.float32]
        The fixed AV map.
    """  # noqa: E501
    a_map = rasterize_tree(trees[0]) > 0
    v_map = rasterize_tree(trees[1]) > 0

    if expand_labels_by > 0:
        from skimage.morphology import binary_dilation, disk

        a_map = binary_dilation(a_map, disk(expand_labels_by))
        v_map = binary_dilation(v_map, disk(expand_labels_by))

    if force_initial_segmentation:
        seg_mask = av_map == 0
        a_map[seg_mask] = False
        v_map[seg_mask] = False

    tree_av_map = a_map + np.uint8(2) * v_map

    if not discard_av:
        mask = tree_av_map == 0
        tree_av_map[mask] = av_map[mask]

    return tree_av_map


def rasterize_tree(
    tree: VTree,
    *,
    expand: int = 0,
    fill_junctions: bool = True,
    bezier_interpolate: float = 0.5,
    boundaries_field: VBranchGeoData.Key = VBranchGeoData.Fields.BOUNDARIES,
) -> npt.NDArray[np.uint64]:
    """
    Rasterize the given vessel tree into a mask with branch indices.

    Parameters
    ----------
    tree : VTree
        The vessel tree to rasterize.

    expand : int, optional
        The number of pixels to expand the labels by. Default is 0.

    fill_junctions : bool, optional
        Whether to fill junctions in the rasterization. Default is True.

    bezier_interpolate : float, optional
        The Bézier interpolation factor. If null, disables Bézier interpolation. Default is 0.5.

    boundaries_field : VBranchGeoData.Key, optional
        The field in the branch geodata to use for boundaries. Default is VBranchGeoData.Fields.BOUNDARIES.

    Returns
    -------
    npt.NDArray[np.uint64]
        The rasterized branch index map.
    """
    geodata = tree.geometric_data()
    labels_map, _ = rasterize_topology(
        branch_list=tree.branch_list,
        branch_tree=tree.branch_tree,
        branch_dirs=tree.branch_dirs(),
        curves=geodata.branch_curve(),
        boundaries=[
            _.data if _ is not None else np.empty((0, 2, 2), dtype=np.int_)
            for _ in geodata.branch_data(boundaries_field)
        ],
        nodes_yx=geodata.node_coord(),
        shape=geodata.domain.shape,
        bezier_interpolate=bezier_interpolate,
        fill_junctions=fill_junctions,
    )

    if expand > 0:
        labels_map = expand_labels(labels_map, distance=expand)

    return labels_map


def draw_missing_connections(graph: VGraph, out: npt.NDArray, fill_value: int = 1):
    """
    Draw missing connections in the graph by filling in the gaps in the out array.

    This functions is useless now that rasterize_tree uses bezier interpolation.

    Parameters
    ----------
    graph : VGraph
        The vessel graph to draw missing connections for.
    out : npt.NDArray
        The output array to fill with missing connections.
    fill_value :
        The value to fill in the gaps. Default is 1.
    """
    for branch in graph.branches():
        mean_calibre = None
        curve = branch.curve()
        calibres = c.data if (c := branch.geodata(VBranchGeoData.Fields.CALIBRES)) is not None else None
        n1, n2 = [node.coord() for node in branch.nodes()]
        for bezier in branch.bspline().filling_curves(n1, n2, smoothing=0.5):
            if bezier[0] == bezier[-1]:
                continue

            mid_points = bezier.evaluate(np.linspace(0, 1, min(10, int(bezier.arc_length(fast_approximation=True)))))
            mid_points = np.round(mid_points).astype(int)
            if (
                np.any(mid_points < 0)
                or np.any(mid_points >= np.array(out.shape))
                or np.all(out[mid_points[:, 0], mid_points[:, 1]] != 0)
            ):
                continue

            mean_calibre = 2.0
            if calibres is not None:
                tip_calibres = []
                if bezier[0] != n1:
                    p0 = np.all(curve == bezier[0], axis=1)
                    if p0.any():
                        tip_calibres.append(calibres[np.argmax(p0)])
                if bezier[-1] != n2:
                    p1 = np.all(curve == bezier[-1], axis=1)
                    if p1.any():
                        tip_calibres.append(calibres[np.argmax(p1)])
                mean_calibre = max(2.0, float(0.75 * np.mean(tip_calibres))) if len(tip_calibres) > 0 else 2.0
            bezier.rasterize(out, width=mean_calibre, fill_value=fill_value)
