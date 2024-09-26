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
        from ..vparameters.branches import tortuosity_parameters

        return {
            "bifurcations": parametrize_bifurcations(tree),
            "branches": tortuosity_parameters(tree.geometric_data()),
        }

    def inspect_bifurcations(self, fundus_data: FundusData):
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
