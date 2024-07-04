from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Self, Tuple

import numpy as np

from ..utils.bezier import BSpline
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


@dataclass
class VGeoAttr:
    name: str
    version: Optional[str] = ""
    branch_geo_attr_type: type = VBranchGeoAttr

    def __str__(self) -> str:
        name = self.name
        if self.version:
            name += f"[{self.version}]"
        return name

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, VGeoAttr)
            and self.name == other.name
            and (self.version == other.version or self.version is None or other.version is None)
        )

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def from_str(s: str, attr_type=VBranchGeoAttr) -> VGeoAttr:
        if "[" in s:
            name, version = s.split("[", 1)
            version = version.strip("] ")
            name = name.strip()
        else:
            name = s
            version = ""
        return VGeoAttr(name, version, attr_type)


####################################################################################################
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


####################################################################################################
class VBranchTangents(VBranchGeoAttr):
    """``VBranchTangents`` is a class that stores the tangents of a branch of a vascular graph."""

    def __init__(self, array: np.ndarray) -> None:
        super().__init__()
        assert array.ndim == 2 and array.shape[1] == 2, "Tangents must be a 2D array with 2 columns."
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
                f"Branch tangents data must have the same length as the branch curve: {curve.shape[0]}."
                f"(Data provided has shape: {self.array.shape}.)"
            )
        return ""

    def merge(self, others: List[VBranchTangents], ctx: Dict[str, Any]) -> VBranchTangents:
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

    def flip(self) -> VBranchTangents:
        """Flip the parametric data.

        Returns
        -------
        VBranchParametricData
            The flipped parametric data.
        """
        self.array = -np.flip(self.array, axis=0)
        return self

    def split(
        self, split_position: Point, split_id: int, ctx: Dict[str, Any]
    ) -> Tuple[VBranchTangents, VBranchTangents]:
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
        return (VBranchTangents(self.array[:split_id]), VBranchTangents(self.array[split_id:]))


####################################################################################################
class VBranchBSpline(VBranchGeoAttr):
    """``VBranchBSpline`` is a class that stores the B-spline representation of a branch of a vascular graph."""

    def __init__(self, bspline: np.ndarray | BSpline) -> None:
        super().__init__()
        if not isinstance(bspline, BSpline):
            bspline = BSpline.from_array(bspline)
        self.bspline = bspline

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
        if self.bspline[0].p0 != curve[0] or self.bspline[-1].p1 != curve[-1]:
            return (
                f"The B-spline representation of the branch must start at the first point of the curve "
                f"and end at the last point of the curve. "
                f"(Data provided: {self.bspline[0].p0} != {curve[0]} or {self.bspline[-1].p1} != {curve[-1]})"
            )
        return ""

    def merge(self, others: List[VBranchBSpline], ctx: Dict[str, Any]) -> VBranchBSpline:
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
        self.bspline = sum(others, start=self.bspline)
        return self

    def flip(self) -> VBranchBSpline:
        """Flip the parametric data.

        Returns
        -------
        VBranchParametricData
            The flipped parametric data.
        """
        self.bspline = self.bspline.flip()
        return self

    def split(self, split_position: Point, split_id: int, ctx: Dict[str, Any]) -> Tuple[VBranchBSpline, VBranchBSpline]:
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
        raise NotImplementedError
