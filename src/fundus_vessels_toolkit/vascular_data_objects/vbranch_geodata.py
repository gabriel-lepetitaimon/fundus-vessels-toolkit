from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, NamedTuple, Optional, Self, Tuple, Type, TypeAlias

import numpy as np

from ..utils.bezier import BSpline
from ..utils.geometric import Point


class VBranchGeoDataBase(ABC):
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


####################################################################################################
class VBranchCurveData(VBranchGeoDataBase):
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

    def merge(self, others: List[VBranchCurveData], ctx: Dict[str, Any]) -> VBranchCurveData:
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

    def flip(self) -> VBranchCurveData:
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
    ) -> Tuple[VBranchCurveData, VBranchCurveData]:
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
        return (VBranchCurveData(self.array[:split_id]), VBranchCurveData(self.array[split_id:]))


####################################################################################################
class VBranchTangents(VBranchGeoDataBase):
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
class VBranchBSpline(VBranchGeoDataBase):
    """``VBranchBSpline`` is a class that stores the B-spline representation of a branch of a vascular graph."""

    def __init__(self, bspline: np.ndarray | BSpline) -> None:
        super().__init__()
        if not isinstance(bspline, BSpline):
            bspline = BSpline.from_array(bspline)
        self.bspline: BSpline = bspline

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
        if curve is None or not len(curve):
            if len(self.bspline):
                return "The B-spline representation of a curve-less branch must be empty."
            else:
                return ""
        elif len(self.bspline) == 0:
            return "The B-spline representation of a branch must have at least one segment."
        if self.bspline[0].p0 != Point(*curve[0]) or self.bspline[-1].p1 != Point(*curve[-1]):
            return (
                f"The B-spline representation of the branch must start at the first point of the curve "
                f"and end at the last point of the curve. "
                f"(Data provided: "
                f"{self.bspline[0].p0} != {Point(*curve[0])} or {self.bspline[-1].p1} != {Point(*curve[-1])})"
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

    def __repr__(self) -> str:
        return f"VBranchBSpline({self.bspline})"


####################################################################################################
class VBranchGeoDescriptor(NamedTuple):
    name: str
    geo_type: Type[VBranchGeoDataBase]

    @staticmethod
    def parse(
        name: str | VBranchGeoDescriptor | Type[VBranchGeoDataBase], geo_type: Type[VBranchGeoDataBase]
    ) -> VBranchGeoDescriptor:
        if isinstance(name, VBranchGeoDescriptor):
            if geo_type is not None and name.geo_type is not geo_type:
                raise ValueError(f"Invalid geo type for descriptor {name.name}. Expected {geo_type}.")
            return name
        return VBranchGeoDescriptor(VBranchGeoDescriptor.parse_name(name), geo_type)

    @staticmethod
    def parse_name(name: str | VBranchGeoDescriptor | Type[VBranchGeoDataBase]) -> str:
        if isinstance(name, str):
            return name
        if isinstance(name, VBranchGeoDescriptor):
            return name.name
        if isinstance(name, Type) and name is not VBranchGeoDataBase and issubclass(name, VBranchGeoDataBase):
            return name.__name__
        raise ValueError(f"Invalid name type: {type(name)}.")


#: The type of curated dictionary of geometric data for branches.
VBranchGeoDict: TypeAlias = Dict[str, List[VBranchGeoDataBase | None]]

#: All the types which may be converted to a VBranchGeoData object.
VBranchGeoDataLike: TypeAlias = np.ndarray | BSpline | VBranchGeoDataBase

#: All the types which may be used to refer to a VBranchGeoData object.
VBranchGeoDataKey: TypeAlias = VBranchGeoDescriptor | str | Type[VBranchGeoDataBase]

#: The type of uncurated dictionary of geometric data for branches.
VBranchGeoDictUncurated: TypeAlias = Dict[VBranchGeoDataKey, List[Optional[VBranchGeoDataLike]]]


class VBranchGeoData:
    """``VBranchGeoData`` is a utility class [...] ."""

    Type: TypeAlias = Type[VBranchGeoDataBase]
    Base = VBranchGeoDataBase
    Descriptor = VBranchGeoDescriptor

    BSpline = VBranchBSpline
    Curve = VBranchCurveData
    Tangents = VBranchTangents

    @staticmethod
    def from_data(
        data: np.ndarray | BSpline | VBranchGeoDataBase, typehint: Optional[Type[VBranchGeoDataBase]] = None
    ) -> VBranchGeoDataBase:
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
        if isinstance(data, BSpline):
            data = VBranchGeoData.BSpline(data)
        if isinstance(data, np.ndarray):
            data = VBranchGeoData.Curve(data)
        if typehint is not None and not isinstance(data, typehint):
            raise ValueError(f"Invalid VBranchGeoData type: {type(data)}. Expected {typehint}.")
        raise ValueError(f"Invalid VBranchGeoData type: {type(data)}.")

    @staticmethod
    def parse_geo_dict(
        data: VBranchGeoDictUncurated, vgeo_data_object
    ) -> Tuple[VBranchGeoDict, Dict[str, VBranchGeoDescriptor]]:
        """Parse a dictionary of geometric data.

        Parameters
        ----------
        data : VBranchGeoDictExtended
            The dictionary of geometric data to parse.

        Returns
        -------
        VBranchGeoDict
            The parsed geometric data.
        """
        branches_geo_data: VBranchGeoDict = {}
        branches_geo_descriptors: Dict[str, VBranchGeoDescriptor] = {}

        for geo_desc, branches_data in data.items():
            geo_type = None
            if isinstance(geo_desc, VBranchGeoDescriptor):
                geo_type = geo_desc.geo_type
            elif isinstance(geo_desc, Type) and issubclass(geo_desc, VBranchGeoDataBase):
                geo_type = geo_desc

            for branch_id, attr_data in enumerate(branches_data):
                try:
                    attr_data = VBranchGeoData.from_data(attr_data, geo_type)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid type for attribute {geo_desc} of branch {branch_id}. " + str(e)
                    ) from None
                if geo_type is None:
                    geo_type = type(attr_data)

                is_invalid = attr_data.is_invalid(vgeo_data_object._branch_ctx(branch_id))
                if is_invalid:
                    raise ValueError(f"Invalid attribute {geo_desc} of branch {branch_id}.\n{is_invalid}")

            geo_desc = VBranchGeoDescriptor.parse(geo_desc, geo_type)
            branches_geo_data[geo_desc.name] = branches_data
            branches_geo_descriptors[geo_desc.name] = geo_desc

        return branches_geo_data, branches_geo_descriptors
