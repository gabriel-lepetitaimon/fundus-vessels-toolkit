from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Self, Tuple

import numpy as np

from ..utils.geometric import Point


class VBranchGeoAttr(ABC):
    """``VBranchGeoAttr`` is an abstract class defining a geometric attribute related to branch of a vascular graph."""

    def is_invalid(self, ctx: Dict[str, Any]) -> str:
        """Check that the attribute data is valid. If not return an error message.

        Parameters
        ----------
        ctx : Dict[str, Any]
            The context in which the attribute data is checked.

        Returns
        -------
        str
            An error message if the attribute data is invalid, otherwise an empty string.
        """
        return ""

    @abstractmethod
    def merge(self, others: List[Self], ctx: Dict[str, Any]) -> Self:
        """Merge the parametric data with another parametric data.

        Parameters
        ----------
        other : VBranchParametricData
            The parametric data to merge.

        Returns
        -------
        VBranchParametricData
            The merged parametric data.
        """
        pass

    @abstractmethod
    def flip(self) -> Self:
        """Flip the parametric data.

        Returns
        -------
        VBranchParametricData
            The flipped parametric data.
        """
        pass

    @abstractmethod
    def split(self, split_position: Point, split_id: int, ctx: Dict[str, Any]) -> Tuple[Self, Self]:
        """Split the parametric data at a given position.

        Parameters
        ----------
        split_position : Point
            The position at which to split the parametric data.

        Returns
        -------
        Tuple[VBranchParametricData, VBranchParametricData]
            The parametric data of the two branches after the split.
        """
        pass


class VBranchCurveAttr(VBranchGeoAttr):
    """``VBranchParametricData`` is a class that stores
    the parametric data of a vascular graph."""

    def __init__(self, array: np.ndarray) -> None:
        super().__init__()
        self.array = array

    def is_invalid(self, ctx: Dict[str, Any]) -> str:
        """Create a parametric data object from an array.

        Parameters
        ----------
        array : np.ndarray
            The array to create the parametric data from.

        Returns
        -------
        VBranchParametricData
            The parametric data object.
        """
        curve = ctx["curve"]
        if self.array.shape[0] != curve.shape[0]:
            return (
                f"Branch curve data must have the same length as the branch curve: {curve.shape[0]}."
                f"(Data provided has shape: {self.array.shape}.)"
            )
        return ""

    def merge(self, others: List[VBranchCurveAttr], ctx: Dict[str, Any]) -> VBranchCurveAttr:
        """Merge the parametric data with another parametric data.

        Parameters
        ----------
        other : VBranchParametricData
            The parametric data to merge.

        Returns
        -------
        VBranchParametricData
            The merged parametric data.
        """
        self.array = np.concatenate([self.array] + [_.array for _ in others], axis=0)
        return self

    def flip(self) -> VBranchCurveAttr:
        """Flip the parametric data.

        Returns
        -------
        VBranchParametricData
            The flipped parametric data.
        """
        self.array = np.flip(self.array, axis=0)
        return self

    def split(
        self, split_position: Point, split_id: int, ctx: Dict[str, Any]
    ) -> Tuple[VBranchCurveAttr, VBranchCurveAttr]:
        """Split the parametric data at a given position.

        Parameters
        ----------
        split_position : Point
            The position at which to split the parametric data.

        Returns
        -------
        Tuple[VBranchParametricData, VBranchParametricData]
            The parametric data of the two branches after the split.
        """
        return (VBranchCurveAttr(self.array[:split_id]), VBranchCurveAttr(self.array[split_id:]))
