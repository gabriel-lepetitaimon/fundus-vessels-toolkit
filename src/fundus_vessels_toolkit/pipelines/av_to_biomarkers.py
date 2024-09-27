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
        import plotly.graph_objects as go
        from IPython.display import display
        from ipywidgets import GridBox, Layout

        from ..vparameters.bifurcations import parametrize_bifurcations

        trees = self.to_av_trees(fundus_data)
        widgets = []
        bifuractions_data = []
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
            cols = list(bifurcations.columns)
            for col in ("d0", "d1", "d2", "θ1", "θ2", "θ_branching", "θ_assymetry"):
                bifurcations[col] = bifurcations[col].astype(int)
            for col in ("assymetry_ratio", "branching_coefficient"):
                bifurcations[col] = bifurcations[col].astype(float).round(2)
            table = go.FigureWidget()
            table.add_table(
                header=dict(values=cols),
                cells=dict(values=[bifurcations[col] for col in cols]),
            )

            widgets.extend([table, view])

        display(
            GridBox(
                widgets,
                layout=Layout(grid_template_columns="repeat(2, 1fr)", grid_template_rows="repeat(600px, 1fr)"),
            )
        )

        return trees, bifuractions_data

    def inspect_branches(self, fundus_data: FundusData):
        import plotly.graph_objects as go
        from IPython.display import display
        from ipywidgets import GridBox, Layout

        from ..vparameters.branches import parametrize_branches

        trees = self.to_av_trees(fundus_data)
        widgets = []
        branch_infos = []
        for i, tree in enumerate(trees):
            branch_info = parametrize_branches(tree)
            branch_infos.append(branch_info)

            # Draw the image view
            view = fundus_data.draw()
            view["graph"] = tree.jppype_layer(edge_map=True, boundaries=True, node_labels=True, edge_labels=True)

            # Draw the table
            branch_info["calibre"] = (
                branch_info["mean_calibre"].round(1).astype(str) + "±" + branch_info["std_calibre"].round(1).astype(str)
            )
            branch_info.drop(columns=["std_calibre", "mean_calibre"], inplace=True)
            for col in ("mean_curvature", "τHart", "τGrisan"):
                branch_info[col] = branch_info[col].astype(float).apply("{:.2e}".format)

            for col in ("n_curvatures_roots",):
                branch_info[col] = branch_info[col].astype(int)

            cols = list(branch_info.columns)
            cols.remove("calibre")
            cols.insert(1, "calibre")

            table = go.FigureWidget()
            table.add_table(
                header=dict(values=cols),
                cells=dict(values=[branch_info[col] for col in cols]),
            )

            widgets.extend([table, view])

        display(
            GridBox(
                widgets,
                layout=Layout(grid_template_columns="repeat(2, 1fr)", grid_template_rows="repeat(600px, 1fr)"),
            )
        )

        return trees, branch_infos
