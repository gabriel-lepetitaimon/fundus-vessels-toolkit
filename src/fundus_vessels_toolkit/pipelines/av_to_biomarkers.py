__all__ = ["FundusAVSegToBiomarkers"]

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from ..vascular_data_objects import FundusData, VTree
from .avseg_to_tree import AVSegToTreeBase


class FundusAVSegToBiomarkers:
    def __init__(self, avseg_to_tree: AVSegToTreeBase):
        self.avseg_to_tree = avseg_to_tree

    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        fundus = self.avseg_to_tree.prepare_data(fundus, av=av, od=od)

        trees = self.to_av_trees(fundus, av=av, od=od)
        biomarkers = {}
        for vtype, tree in zip(["art", "vei"], trees, strict=True):
            biomarkers |= {f"{vtype}_{k}": v for k, v in self.tree_to_biomarkers(tree, fundus).items()}
        return biomarkers

    def to_av_trees(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VTree, VTree]:
        trees = self.avseg_to_tree(fundus, av=av, od=od)

        return trees

    def tree_to_biomarkers(self, tree: VTree, fundus_data: Optional[FundusData] = None) -> Dict[str, pd.DataFrame]:
        from ..vparameters.bifurcations import parametrize_bifurcations
        from ..vparameters.branches import parametrize_branches

        return {
            "bifurcations": parametrize_bifurcations(tree, fundus_data=fundus_data),
            "branches": parametrize_branches(tree),
        }

    def inspect_bifurcations(self, fundus_data: FundusData, *, show_tangents=True, label_branches=False):
        import panel as pn
        from bokeh.models.widgets.tables import NumberFormatter
        from IPython.display import display
        from ipywidgets import GridBox, Layout

        pn.extension("tabulator")

        from ..vparameters.bifurcations import parametrize_bifurcations

        trees = self.to_av_trees(fundus_data)
        views = []
        tables = []
        bifuractions_data = []

        def focus_on(id):
            def callback(event):
                coord = trees[id].nodes_coord()[bifuractions_data[id]["node"][event.row]]
                views[id].goto(coord[::-1], scale=3)

            return callback

        for i, tree in enumerate(trees):
            bifurcations = parametrize_bifurcations(tree, fundus_data=fundus_data)
            bifuractions_data.append(bifurcations)
            nodes_rank = tree.nodes_attr["rank"].copy()
            branches_rank = tree.branches_attr["rank"].copy()

            # Draw the image view
            view = fundus_data.draw()
            view["graph"] = tree.jppype_layer(
                edge_map=True, boundaries=True, node_labels=True, edge_labels=label_branches
            )
            nodes_rank[nodes_rank > 7] = -1
            view["graph"].nodes_cmap = nodes_rank.map(RANK_COLOR_MAP).to_dict()
            branches_rank[branches_rank > 7] = -1
            view["graph"].edges_cmap = branches_rank.map(RANK_COLOR_MAP).to_dict()
            if show_tangents:
                show = np.zeros((tree.branches_count, 2), dtype=bool)
                invert = np.zeros((tree.branches_count, 2), dtype=bool)
                b0, b1, b2 = [bifurcations[_].to_numpy() for _ in ("branch0", "branch1", "branch2")]
                show[b0, np.where(tree.branch_dirs(b0), 1, 0)] = True
                show[b1, np.where(tree.branch_dirs(b1), 0, 1)] = True
                show[b2, np.where(tree.branch_dirs(b2), 0, 1)] = True
                invert[b0, np.where(tree.branch_dirs(b0), 1, 0)] = True
                # invert[b2, np.where(tree.branch_dirs(b2), 0, 1)] = True
                view["tangents"] = tree.geometric_data().jppype_branches_tips_tangents(
                    show_only=show, invert_direction=invert, scaling=10
                )

            # Draw the table
            formatter = {}
            for col in ("dist_od", "dist_macula", "d0", "d1", "d2"):
                formatter[col] = NumberFormatter(format="0")
            for col in ("θ1", "θ2", "θ_branching", "θ_assymetry"):
                formatter[col] = NumberFormatter(format="0°")
            for col in ("assymetry_ratio", "branching_coefficient"):
                formatter[col] = NumberFormatter(format="0.00a")
            if label_branches:
                bifurcations.rename(columns={"branch0": "B0", "branch1": "B1", "branch2": "B2"}, inplace=True)
            else:
                bifurcations.drop(columns=["branch0", "branch1", "branch2"], inplace=True)
            table = pn.widgets.Tabulator(
                bifurcations,
                show_index=False,
                disabled=True,
                layout="fit_data",
                formatters=formatter,
                editors={k: None for k in bifurcations.columns},
                text_align={k: "center" for k in bifurcations.columns},
                width=860,
                height=300,
                frozen_columns=["node", "B0"],
                align="center",
            )
            table.on_click(focus_on(i))

            views.append(view)
            tables.append(table)

        display(GridBox(views, layout=Layout(grid_template_columns="repeat(2, 1fr)", grid_template_rows="400px")))
        return (
            pn.Row(tables[0], pn.Spacer(width=10), tables[1], height=400, scroll=True),
            trees,
            bifuractions_data,
        )

    def inspect_branches(self, fundus_data: FundusData):
        import panel as pn
        from bokeh.models.widgets.tables import NumberFormatter, ScientificFormatter
        from IPython.display import display
        from ipywidgets import GridBox, Layout
        from matplotlib import colormaps as mpl_cmaps
        from matplotlib.colors import to_hex

        pn.extension("tabulator")

        from ..vparameters.bifurcations import parametrize_bifurcations
        from ..vparameters.branches import parametrize_branches

        trees = self.to_av_trees(fundus_data)
        views = []
        tables = []
        branch_infos = []
        color_coding = [None, None]

        CMAP = "cividis"

        def set_edge_encoding(id, column):
            if column == color_coding[id] or not column.startswith("τ") and column not in ("rank", "strahler"):
                return
            cmap = mpl_cmaps[CMAP]
            data = tables[id].value[column]
            data -= data.min()
            data /= data.max()
            if column in ("rank", "strahler"):
                data = 1 - data
            views[id]["graph"].edges_cmap = data.map(lambda x: to_hex(cmap(x))).to_dict()
            color_coding[id] = column

        def on_table_clicked(id):
            def callback(event):
                coord = trees[id].geometric_data().branch_midpoint(branch_infos[id]["branch"][event.row])
                views[id].goto(coord[::-1], scale=2)
                set_edge_encoding(id, event.column)

            return callback

        for i, tree in enumerate(trees):
            parametrize_bifurcations(tree, fundus_data=fundus_data)  # To compute rank and strahler
            branch_info = parametrize_branches(tree)
            branch_infos.append(branch_info)
            nodes_rank = tree.nodes_attr["rank"].copy()
            branches_rank = tree.branches_attr["rank"].copy()

            # Draw the image view
            view = fundus_data.draw()
            view["graph"] = tree.jppype_layer(edge_map=True, boundaries=True, node_labels=False, edge_labels=True)
            nodes_rank[nodes_rank > 7] = -1
            view["graph"].nodes_cmap = nodes_rank.map(RANK_COLOR_MAP).to_dict()
            branches_rank[branches_rank > 7] = -1
            view["graph"].edges_cmap = branches_rank.map(RANK_COLOR_MAP).to_dict()

            # Draw the table
            branch_info = branch_info.rename(
                columns={
                    "mean_calibre": "μCal",
                    "std_calibre": "σCal",
                    "mean_curvature": "μCurv",
                    "std_curvature": "σCurv",
                    "n_curvatures_roots": "n_curv",
                }
            )
            branch_info.index.names = ["index"]
            formatter = {}
            for col in ("μCal", "σCal"):
                formatter[col] = ScientificFormatter(precision=2)
            for col in ("μCurv", "σCurv", "τHart", "τGrisan", "τTrucco", "τHart_bspline", "τGrisan_bspline"):
                formatter[col] = ScientificFormatter(precision=3)
            for col in ("n_curv", "n_bspline"):
                formatter[col] = NumberFormatter(format="0")
            for col in ("assymetry_ratio", "branching_coefficient"):
                formatter[col] = NumberFormatter(format="0.00e")
            table = pn.widgets.Tabulator(
                branch_info,
                show_index=False,
                disabled=True,
                layout="fit_data",
                formatters=formatter,
                editors={k: None for k in branch_info.columns},
                text_align={k: "center" for k in branch_info.columns},
                width=860,
                height=300,
                frozen_columns=["branch"],
                align="center",
            )
            table.on_click(on_table_clicked(i))
            table.style.background_gradient(
                cmap=CMAP,
                subset=["τHart", "τGrisan", "τTrucco", "τHart_bspline", "τGrisan_bspline", "n_curv", "n_bspline"],
            )

            views.append(view)
            tables.append(table)
            set_edge_encoding(i, "rank")

        display(GridBox(views, layout=Layout(grid_template_columns="repeat(2, 1fr)", grid_template_rows="400px")))
        return (
            pn.Row(tables[0], pn.Spacer(width=10), tables[1], height=400, scroll=True),
            trees,
            branch_infos,
        )


RANK_COLOR_MAP = {
    -1: "#f9ecdd",
    0: "#3a37ff",
    1: "#6f00ff",
    2: "#f36bd6",
    3: "#ff8282",
    4: "#ff9f79",
    5: "#ffcc92",
    6: "#ebe7b3",
    7: "#f9ecdd",
}
