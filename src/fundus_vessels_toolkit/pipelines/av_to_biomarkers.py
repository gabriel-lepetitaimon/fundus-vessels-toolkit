__all__ = ["FundusAVSegToBiomarkers"]

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from ..vascular_data_objects import FundusData, VTree
from .avseg_to_tree import FundusAVSegToTree


class FundusAVSegToBiomarkers:
    def __init__(self, max_vessel_diameter=20):
        self.avseg_to_tree = FundusAVSegToTree(max_vessel_diameter=max_vessel_diameter)

    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Dict[str, pd.DataFrame]:
        trees = self.to_av_trees(fundus, av=av, od=od)
        biomarkers = {}
        for vtype, tree in zip(["art", "vei"], trees, strict=True):
            biomarkers |= {f"{vtype}_{k}": v for k, v in self.tree_to_biomarkers(tree).items()}
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

    def tree_to_biomarkers(self, tree: VTree) -> Dict[str, pd.DataFrame]:
        from ..vparameters.bifurcations import parametrize_bifurcations
        from ..vparameters.branches import parametrize_branches

        return {
            "bifurcations": parametrize_bifurcations(tree),
            "branches": parametrize_branches(tree),
        }

    def inspect_bifurcations(self, fundus_data: FundusData, *, show_tangents=True):
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
            bifurcations = parametrize_bifurcations(tree)
            bifuractions_data.append(bifurcations)

            # Draw the image view
            view = fundus_data.draw()
            view["graph"] = tree.jppype_layer(edge_map=True, boundaries=True, node_labels=True, edge_labels=True)
            if show_tangents:
                show = np.zeros((tree.branches_count, 2), dtype=bool)
                invert = np.zeros((tree.branches_count, 2), dtype=bool)
                b0, b1, b2 = [bifurcations[_].to_numpy() for _ in ("branch0", "branch1", "branch2")]
                show[b0, np.where(tree.branch_dirs(b0), 1, 0)] = True
                show[b1, np.where(tree.branch_dirs(b1), 0, 1)] = True
                show[b2, np.where(tree.branch_dirs(b2), 0, 1)] = True
                invert[b1, np.where(tree.branch_dirs(b1), 0, 1)] = True
                invert[b2, np.where(tree.branch_dirs(b2), 0, 1)] = True
                view["tangents"] = tree.geometric_data().jppype_branches_tips_tangents(
                    show_only=show, invert_direction=invert, scaling=10
                )

            # Draw the table
            formatter = {}
            for col in ("d0", "d1", "d2"):
                formatter[col] = NumberFormatter(format="0")
            for col in ("θ1", "θ2", "θ_branching", "θ_assymetry"):
                formatter[col] = NumberFormatter(format="0°")
            for col in ("assymetry_ratio", "branching_coefficient"):
                formatter[col] = NumberFormatter(format="0.00a")
            bifurcations = bifurcations.rename(columns={"branch0": "B0", "branch1": "B1", "branch2": "B2"})
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
        from bokeh.models.widgets.tables import NumberFormatter
        from IPython.display import display
        from ipywidgets import GridBox, Layout

        pn.extension("tabulator")

        from ..vparameters.branches import parametrize_branches

        trees = self.to_av_trees(fundus_data)
        views = []
        tables = []
        branch_infos = []

        def focus_on(id):
            def callback(event):
                coord = trees[id].geometric_data().branch_midpoint(branch_infos[id]["branch"][event.row])
                views[id].goto(coord[::-1], scale=2)

            return callback

        for i, tree in enumerate(trees):
            branch_info = parametrize_branches(tree)
            branch_infos.append(branch_info)

            # Draw the image view
            view = fundus_data.draw()
            view["graph"] = tree.jppype_layer(edge_map=True, boundaries=True, node_labels=True, edge_labels=True)

            # Draw the table
            branch_info = branch_info.rename(
                columns={
                    "mean_calibre": "μCal",
                    "std_calibre": "σCal",
                    "mean_curvature": "μCurv",
                    "n_curvatures_roots": "n_curv",
                }
            )
            formatter = {}
            for col in ("μCal", "σCal", "μCurv", "τHart", "τGrisan"):
                formatter[col] = NumberFormatter(format="0.00a")
            for col in ("n_curv",):
                formatter[col] = NumberFormatter(format="0")
            for col in ("assymetry_ratio", "branching_coefficient"):
                formatter[col] = NumberFormatter(format="0.00a")
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
            table.on_click(focus_on(i))

            views.append(view)
            tables.append(table)

        display(GridBox(views, layout=Layout(grid_template_columns="repeat(2, 1fr)", grid_template_rows="400px")))
        return (
            pn.Row(tables[0], pn.Spacer(width=10), tables[1], height=400, scroll=True),
            trees,
            branch_infos,
        )
