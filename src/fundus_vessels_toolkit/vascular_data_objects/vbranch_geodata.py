from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Self, Tuple, Type, TypeAlias

import numpy as np

from ..utils.bezier import BSpline
from ..utils.data_io import NumpyDict, load_numpy_dict, save_numpy_dict
from ..utils.fundus_projections import FundusProjection
from ..utils.geometric import Point

_registered_vbranch_geo_data_types: Dict[str, Type[VBranchGeoDataBase]] = {}


class MetaVBranchGeoDataBase(ABCMeta):
    """
    Meta class for the abstract class ``VBranchGeoDataBase``.

    This meta class automatically registers all subclasses of ``VBranchGeoDataBase`` in the module.
    """

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != "VBranchGeoDataBase" and name not in _registered_vbranch_geo_data_types:
            _registered_vbranch_geo_data_types[name] = cls
        return cls


class VBranchGeoDataBase(ABC, metaclass=MetaVBranchGeoDataBase):
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

    def transform(self, projection: FundusProjection, ctx: Dict[str, Any]) -> Self:
        """Transform the parametric data using a projection.

        Parameters
        ----------
        projection : FundusProjection
            The projection to apply to the parametric data.

        Returns
        -------
        VBranchParametricData
            The transformed parametric data.
        """
        raise NotImplementedError


####################################################################################################
class VBranchCurveData(VBranchGeoDataBase):
    """``VBranchParametricData`` is a class that stores
    the parametric data of a vascular graph."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = data

    def is_invalid(self, ctx: Dict[str, Any]) -> str:
        curve = ctx["curve"]
        if self.data.shape[0] != curve.shape[0]:
            return (
                f"Branch curve data must have the same length as the branch curve: {curve.shape[0]}."
                f"(Data provided has shape: {self.data.shape}.)"
            )
        return ""

    def merge(self, others: List[VBranchCurveData], ctx: Dict[str, Any]) -> VBranchCurveData:
        self.data = np.concatenate([self.data] + [_.data for _ in others], axis=0)
        return self

    def flip(self) -> VBranchCurveData:
        self.data = np.flip(self.data, axis=0)
        return self

    def split(
        self, split_position: Point, split_id: int, ctx: Dict[str, Any]
    ) -> Tuple[VBranchCurveData, VBranchCurveData]:
        return (VBranchCurveData(self.data[:split_id]), VBranchCurveData(self.data[split_id:]))


####################################################################################################
class VBranchTangents(VBranchGeoDataBase):
    """``VBranchTangents`` is a class that stores the tangents of a branch of a vascular graph."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        assert data.ndim == 2 and data.shape[1] == 2, "Tangents must be a 2D array with 2 columns."
        self.data = data

    def is_invalid(self, ctx: Dict[str, Any]) -> str:
        curve = ctx["curve"]
        if self.data.shape[0] != curve.shape[0]:
            return (
                f"Branch tangents data must have the same length as the branch curve: {curve.shape[0]}."
                f"(Data provided has shape: {self.data.shape}.)"
            )
        return ""

    def merge(self, others: List[VBranchTangents], ctx: Dict[str, Any]) -> VBranchTangents:
        self.data = np.concatenate([self.data] + [_.data for _ in others], axis=0)
        return self

    def flip(self) -> VBranchTangents:
        self.data = -np.flip(self.data, axis=0)
        return self

    def split(
        self, split_position: Point, split_id: int, ctx: Dict[str, Any]
    ) -> Tuple[VBranchTangents, VBranchTangents]:
        return (VBranchTangents(self.data[:split_id]), VBranchTangents(self.data[split_id:]))


####################################################################################################
class VBranchTerminationData(VBranchGeoDataBase):
    """``VBranchTerminationData`` is a class that stores a scalar associated with a termination of a vascular branch."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        assert (
            data.ndim >= 1 and data.shape[0] == 2
        ), "Termination data must be at least a 1D array must have a length of 2."
        self.data = data

    def is_invalid(self, ctx: Dict[str, Any]) -> str:
        return ""

    def merge(self, others: List[VBranchTerminationData], ctx: Dict[str, Any]) -> VBranchTerminationData:
        self.data = np.array([self.data[0], others[-1].data[1]])
        return self

    def flip(self) -> VBranchTangents:
        self.data = np.flip(self.data, axis=0)
        return self

    def split(
        self, split_position: Point, split_id: int, ctx: Dict[str, Any]
    ) -> Tuple[VBranchTangents, VBranchTangents]:
        nan = np.empty_like(self.data[0])
        nan.fill(float("nan"))
        return (
            VBranchTangents(np.array([self.data[0], nan])),
            VBranchTangents(np.array([nan, self.data[1]])),
        )


####################################################################################################
class VBranchTerminationTangents(VBranchTerminationData):
    """``VBranchTerminationTangents`` is a class that stores the tangents associated with a termination
    of a vascular branch."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data)
        assert (
            data.ndim == 2 and data.shape[0] == 2 and data.shape[1] == 2
        ), "Termination tangents must be a 2D array of shape (2, 2)."
        self.data = data

    def flip(self) -> VBranchTangents:
        self.data = -np.flip(self.data, axis=0)
        return self


####################################################################################################
class VBranchBSpline(VBranchGeoDataBase):
    """``VBranchBSpline`` is a class that stores the B-spline representation of a branch of a vascular graph."""

    def __init__(self, data: np.ndarray | BSpline) -> None:
        super().__init__()
        if not isinstance(data, BSpline):
            data = BSpline.from_array(data)
        self.data: BSpline = data

    def is_invalid(self, ctx: Dict[str, Any]) -> str:
        curve = ctx["curve"]
        if curve is None or not len(curve):
            if len(self.data):
                return "The B-spline representation of a curve-less branch must be empty."
            else:
                return ""
        elif len(self.data) == 0:
            return "The B-spline representation of a branch must have at least one segment."
        if self.data[0].p0 != Point(*curve[0]) or self.data[-1].p1 != Point(*curve[-1]):
            return (
                f"The B-spline representation of the branch must start at the first point of the curve "
                f"and end at the last point of the curve. "
                f"(Data provided: "
                f"{self.data[0].p0} != {Point(*curve[0])} or {self.data[-1].p1} != {Point(*curve[-1])})"
            )
        return ""

    def merge(self, others: List[VBranchBSpline], ctx: Dict[str, Any]) -> VBranchBSpline:
        self.data = sum(others, start=self.data)
        return self

    def flip(self) -> VBranchBSpline:
        self.data = self.data.flip()
        return self

    def split(self, split_position: Point, split_id: int, ctx: Dict[str, Any]) -> Tuple[VBranchBSpline, VBranchBSpline]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"VBranchBSpline({self.data})"

    def transform(self, projection: FundusProjection, ctx: Dict[str, Any]) -> VBranchBSpline:
        bspline_data = [projection.transform(_.to_array()) for _ in self.data]
        return VBranchBSpline(BSpline.from_array(bspline_data))


####################################################################################################
class VBranchGeoDescriptor(NamedTuple):
    name: str
    geo_type: Optional[Type[VBranchGeoDataBase]] = None

    @staticmethod
    def parse(key: VBranchGeoDataKey, geo_type: Optional[Type[VBranchGeoDataBase]] = None) -> VBranchGeoDescriptor:
        if isinstance(key, Type) and issubclass(key, VBranchGeoDataBase):
            key = VBranchGeoField.by_type(key).value
        elif isinstance(key, VBranchGeoField):
            key = key.value
        if isinstance(key, VBranchGeoDescriptor):
            if geo_type is not None and not issubclass(key.geo_type, geo_type):
                raise ValueError(
                    f"Invalid branch geometrical attribute type: {key.geo_type.__name__} (for attribute: {key.name}). "
                    f"Expected type is: {geo_type.__name__}."
                )
            return key
        elif isinstance(key, str):
            if geo_type is not None and key in VBranchGeoField.__members__:
                key = VBranchGeoField[key].value
                if not issubclass(key.geo_type, geo_type):
                    raise ValueError(
                        f"Invalid branch geometrical attribute type: {key.geo_type.__name__} "
                        f"(for attribute: {key.name}). Expected type is: {geo_type.__name__}."
                    )
                return key
            return VBranchGeoDescriptor(key, geo_type)
        raise ValueError(f"Invalid type for geo descriptor: {type(key)}")


class VBranchGeoField(Enum):
    """``VBranchGeoFields`` is an enumeration of the fields of a branch of a vascular graph."""

    #: The tangent of the branch at each skeleton point.
    TANGENTS = VBranchGeoDescriptor("TANGENTS", VBranchTangents)

    #: The calibre of the branch at each skeleton point.
    CALIBRES = VBranchGeoDescriptor("CALIBRE", VBranchCurveData)

    #: The position of the left and right boundaries of the branch.
    BOUNDARIES = VBranchGeoDescriptor("BOUNDARIES", VBranchCurveData)

    #: The curvature of the branch at each skeleton point.
    CURVATURES = VBranchGeoDescriptor("CURVATURES", VBranchCurveData)

    #: The B-spline representation of the branch.
    BSPLINE = VBranchGeoDescriptor("BSPLINE", VBranchBSpline)

    #: The tangents at the termination of the branch.
    TERMINATION_TANGENTS = VBranchGeoDescriptor("TERMINATION_TANGENTS", VBranchTerminationTangents)

    #: The calibre at the termination of the branch.
    TERMINATION_CALIBRES = VBranchGeoDescriptor("TERMINATION_CALIBRE", VBranchTerminationData)

    #: The position of the left and right boundaries of the branch termination.
    TERMINATION_BOUNDARIES = VBranchGeoDescriptor("TERMINATION_BOUNDARIES", VBranchTerminationData)

    @classmethod
    def by_type(cls, geo_type: Type[VBranchGeoDataBase]) -> VBranchGeoField:
        if issubclass(geo_type, VBranchTangents):
            return cls.TANGENTS
        if issubclass(geo_type, VBranchBSpline):
            return cls.BSPLINE
        if issubclass(geo_type, VBranchTerminationTangents):
            return cls.TERMINATION_TANGENTS

        raise ValueError(f"Impossible to infer VBranchGeoField from geo type: {geo_type}.")


#: The type of curated dictionary of geometric data for branches.
VBranchGeoDict: TypeAlias = Dict[str, List[VBranchGeoDataBase | None]]

#: All the types which may be converted to a VBranchGeoData object.
VBranchGeoDataLike: TypeAlias = np.ndarray | BSpline | VBranchGeoDataBase

#: All the types which may be used to refer to a VBranchGeoData object.
VBranchGeoDataKey: TypeAlias = VBranchGeoDescriptor | str | Type[VBranchGeoDataBase] | VBranchGeoField

#: The type of uncurated dictionary of geometric data for branches.
VBranchGeoDictUncurated: TypeAlias = Dict[VBranchGeoDataKey, List[Optional[VBranchGeoDataLike]]]


class VBranchGeoData:
    """``VBranchGeoData`` is a utility class [...] ."""

    Type: TypeAlias = Type[VBranchGeoDataBase]
    Base = VBranchGeoDataBase
    Descriptor = VBranchGeoDescriptor
    Fields = VBranchGeoField

    BSpline = VBranchBSpline
    Curve = VBranchCurveData
    Tangents = VBranchTangents
    TerminationTangents = VBranchTerminationTangents
    TerminationData = VBranchTerminationData

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
        if typehint is not None:
            if isinstance(data, typehint):
                return data
            elif isinstance(data, VBranchGeoDataBase):
                raise ValueError(f"Invalid VBranchGeoData type: {type(data)}. Expected {typehint}.")
            return typehint(data)
        else:
            if isinstance(data, VBranchGeoDataBase):
                return data
            if isinstance(data, BSpline):
                return VBranchGeoData.BSpline(data)
            if isinstance(data, np.ndarray):
                return VBranchGeoData.Curve(data)
            raise ValueError(f"Invalid VBranchGeoData type: {type(data)}.")

    @staticmethod
    def registered_type_name(data: VBranchGeoDataBase | List[VBranchGeoDataBase | None]) -> str:
        if isinstance(data, list):
            data = next(_ for _ in data if _ is not None)
        try:
            return next(name for name, t in _registered_vbranch_geo_data_types.items() if data.__class__ is t)
        except StopIteration:
            raise ValueError(f"Unknown type: {type(data)}. Make sure corresponding type was imported.")

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
            geo_desc = VBranchGeoDescriptor.parse(geo_desc)
            geo_type = geo_desc.geo_type

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
            if geo_desc.geo_type is None:
                geo_desc = VBranchGeoDescriptor(geo_desc.name, geo_type)
            branches_geo_data[geo_desc.name] = branches_data
            branches_geo_descriptors[geo_desc.name] = geo_desc

        return branches_geo_data, branches_geo_descriptors

    @staticmethod
    def save(data: VBranchGeoDict, filename: Optional[str | Path] = None) -> NumpyDict:
        """Save a dictionary of branches geometric data to a numpy dictionary.

        Parameters
        ----------
        data :
            The dictionary of branches geometric data to save.
        filename : Optional[str  |  Path], optional
            The file to save the branches geometric data to. If not provided, the data is not saved to a file.


        Returns
        -------
        NumpyDict
            The branches geometric data as a dictionary of numpy arrays.
        """
        out = {}
        for k, v in data.items():
            type_name = VBranchGeoData.registered_type_name(v)
            out[type_name + "::" + k] = v

        if filename is not None:
            save_numpy_dict(out, filename)
        return out

    @staticmethod
    def load(filename: str | Path | NumpyDict) -> VBranchGeoDict:
        """
        Load a dictionary of geometric data from a file.

        Parameters
        ----------
        filename : str | Path | NumpyDict
            The file to load the geometric data from.

        Returns
        -------
        VBranchGeoDict
            The loaded geometric data.
        """
        if isinstance(filename, (str, Path)):
            file_data = load_numpy_dict(filename)
        else:
            file_data = filename

        data = {}
        for k, v in file_data.items():
            type_name, k = k.split("::")
            assert (
                type_name in _registered_vbranch_geo_data_types
            ), f"Unknown type: {type_name}. Make sure corresponding type was imported."
            data[k] = [VBranchGeoData.from_data(_, _registered_vbranch_geo_data_types[type_name]) for _ in v]

        return data
