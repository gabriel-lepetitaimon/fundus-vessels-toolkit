from __future__ import annotations

import itertools
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


class BranchGeoDataEditContext(NamedTuple):
    """``BranchGeoDataEditContext`` is a named tuple that stores the context in which a branch geometrical data is edited."""  # noqa: E501

    #: Attribute name
    attr_name: str

    #: The branch index.
    branch_id: int

    #: The curve of the branch.
    curve: np.ndarray

    #: The branch geometrical data (may not have been edited yet)
    geodata_attrs: Dict[str, VBranchGeoDataBase]

    #: Other information related to this edition
    info: Dict[str, Any] = {}

    def set_name(self, name: str) -> BranchGeoDataEditContext:
        return self._replace(attr_name=name)

    def set_info(self, **kwargs: Any) -> BranchGeoDataEditContext:
        return self._replace(info={**self.info, **kwargs})


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

    def is_invalid(self, ctx: BranchGeoDataEditContext) -> str:
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

    def __bool__(self) -> bool:
        return not self.is_empty()

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the attribute data is empty.

        Returns
        -------
        bool
            True if the attribute data is empty, False otherwise.
        """
        pass

    @abstractmethod
    def merge(self, others: List[Self], ctx: BranchGeoDataEditContext) -> Self:
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
    def flip(self, ctx: BranchGeoDataEditContext) -> Self:
        """Flip the parametric data.

        Returns
        -------
        VBranchParametricData
            The flipped parametric data.
        """
        pass

    @abstractmethod
    def split(self, splits_point: List[Point], splits_id: List[int], ctx: BranchGeoDataEditContext) -> List[Self]:
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

    def transform(self, projection: FundusProjection, ctx: BranchGeoDataEditContext) -> Self:
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


# def edit_after(geodata_type: Type[VBranchGeoDataBase] | str):
#     """Decorator signaling that the edition of this geodata type should be done optimally after having edited another geodata type.

#     Parameters
#     ----------
#     geodata_type : Type[VBranchGeoDataBase] | str
#         The type of the geometrical data to optimally edit before this one.

#     Returns
#     -------
#     Callable
#         A wrapper function marking the decorated function with an attribute `_execute_after` containing the geodata type.
#     """  # noqa: E501

#     geodata_type = VBranchGeoData.geodata_type(geodata_type)

#     def wrapper(f):
#         f._execute_after = geodata_type
#         return f

#     return wrapper


####################################################################################################
class VBranchCurveData(VBranchGeoDataBase):
    """``VBranchCurveData`` is a class that stores the parametric data of a vascular graph."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = data

    def is_invalid(self, ctx: BranchGeoDataEditContext) -> str:
        curve = ctx.curve
        if self.data.shape[0] != curve.shape[0]:
            return (
                f"Branch curve data must have the same length as the branch curve: {curve.shape[0]}."
                f"(Data provided has shape: {self.data.shape}.)"
            )
        return ""

    def is_empty(self) -> bool:
        return not len(self.data)

    def merge(self, others: List[VBranchCurveData], ctx: BranchGeoDataEditContext) -> VBranchCurveData:
        self.data = np.concatenate([self.data] + [_.data for _ in others], axis=0)
        return self

    def flip(self, ctx: BranchGeoDataEditContext) -> VBranchCurveData:
        self.data = np.flip(self.data, axis=0)
        return self

    def split(self, splits_point: List[Point], splits_id: List[int], ctx: BranchGeoDataEditContext) -> List[Self]:
        return [VBranchCurveData(self.data[start:end]) for start, end in itertools.pairwise(splits_id)]


####################################################################################################
class VBranchCurveIndex(VBranchGeoDataBase):
    """``VBranchCurveIndex`` stores specific indices of the branch curves."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        self.data = np.atleast_1d(data).astype(int)
        assert self.data.ndim == 1, "Branch curve index data must be a 1D array."

    def is_invalid(self, ctx: BranchGeoDataEditContext) -> str:
        self.data = np.sort(self.data.copy())
        if len(self.data) and (self.data.max() >= ctx.curve.shape[0] or self.data.min() < 0):
            return f"Branch curve index data must have values between 0 and {ctx.curve.shape[0] - 1}."
        return ""

    def is_empty(self) -> bool:
        return not len(self.data)

    def merge(self, others: List[VBranchCurveData], ctx: BranchGeoDataEditContext) -> VBranchCurveData:
        curves_start_index = np.cumsum([0] + [curve.shape[0] for curve in ctx.info["curves"][:-1]])
        self.data = np.concatenate(
            [d.data + start for d, start in zip([self] + others, curves_start_index, strict=True)], axis=0
        )
        return self

    def flip(self, ctx: BranchGeoDataEditContext) -> VBranchCurveData:
        self.data = np.flip(ctx.curve.shape[0] - self.data, axis=0)
        return self

    def split(self, splits_point: List[Point], splits_id: List[int], ctx: BranchGeoDataEditContext) -> List[Self]:
        splitted_curveId = []
        start = 0
        for id in splits_id[1:-1]:
            end = np.argmax(self.data[start:] >= id)
            splitted_curveId.append(VBranchCurveIndex(self.data[start : start + end]))
            start += end
        splitted_curveId.append(VBranchCurveIndex(self.data[start:]))

        return splitted_curveId


####################################################################################################
class VBranchTangents(VBranchGeoDataBase):
    """``VBranchTangents`` is a class that stores the tangents of a branch of a vascular graph."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        assert data.ndim == 2 and data.shape[1] == 2, "Tangents must be a 2D array with 2 columns."
        self.data = data

    def is_invalid(self, ctx: BranchGeoDataEditContext) -> str:
        curve = ctx.curve
        if self.data.shape[0] != curve.shape[0]:
            return (
                f"Branch tangents data must have the same length as the branch curve: {curve.shape[0]}."
                f"(Data provided has shape: {self.data.shape}.)"
            )
        return ""

    def is_empty(self) -> bool:
        return not len(self.data)

    def merge(self, others: List[VBranchTangents], ctx: BranchGeoDataEditContext) -> VBranchTangents:
        self.data = np.concatenate([self.data] + [_.data for _ in others], axis=0)
        return self

    def flip(self, ctx: BranchGeoDataEditContext) -> VBranchTangents:
        self.data = -np.flip(self.data, axis=0)
        return self

    def split(self, splits_point: List[Point], splits_id: List[int], ctx: BranchGeoDataEditContext) -> List[Self]:
        return [VBranchTangents(self.data[start:end]) for start, end in itertools.pairwise(splits_id)]


####################################################################################################
class VBranchTipsData(VBranchGeoDataBase):
    """``VBranchTipsData`` is a class that stores a data associated with the tips of a vascular branch."""

    def __init__(self, data: np.ndarray) -> None:
        super().__init__()
        data = np.array(data, dtype=self.dtype())
        expected_shape = (2, *self.data_shape())
        assert (
            data.shape == expected_shape
        ), f"{self.__class__.__qualname__} data shape must be {expected_shape} but {data.shape} was provided."
        self.data = data

    @classmethod
    @abstractmethod
    def data_shape(cls) -> Tuple[int, ...]: ...

    @classmethod
    def dtype(cls):
        return float

    @classmethod
    def create_empty(cls) -> Self:
        return cls([cls.empty_data()] * 2)

    @classmethod
    def empty_data(self) -> np.ndarray:
        nan = np.empty(self.data_shape(), dtype=self.dtype())
        nan.fill(float("nan"))
        return nan

    def is_empty(self) -> bool:
        return not np.all(np.isnan(self.data))

    def merge(self, others: List[VBranchTipsData], ctx: BranchGeoDataEditContext) -> VBranchTipsData:
        self.data = np.array([self.data[0], others[-1].data[1] if others[-1] is not None else self.empty_data()])
        return self

    def flip(self, ctx: BranchGeoDataEditContext) -> VBranchTangents:
        self.data = np.flip(self.data, axis=0)
        return self

    def split(self, splits_point: List[Point], splits_id: List[int], ctx: BranchGeoDataEditContext) -> List[Self]:
        nan = self.empty_data()
        cls = self.__class__
        return (
            [cls(np.array([self.data[0], nan]))]
            + [self.create_empty() for _ in range(len(splits_id) - 3)]
            + [cls(np.array([nan, self.data[1]]))]
        )


####################################################################################################
class VBranchTipsScalarData(VBranchTipsData):
    """``VBranchTipsScalarData`` is a class that stores a scalar associated with the tips of a vascular branch."""

    @classmethod
    def data_shape(cls) -> Tuple[int, ...]:
        return ()

    @classmethod
    def empty_data(cls) -> np.ndarray:
        return np.array(float("nan"), dtype=float)


####################################################################################################
class VBranchTipsDoublePointsData(VBranchTipsData):
    """``VBranchTipsScalarData`` is a class that stores a scalar associated with the tips of a vascular branch."""

    @classmethod
    def data_shape(cls) -> Tuple[int, ...]:
        return (2, 2)


####################################################################################################
class VBranchTipsTangents(VBranchTipsData):
    """``VBranchTipsTangents`` is a class that stores the tangents associated with the tips of a vascular branch."""

    @classmethod
    def data_shape(cls) -> Tuple[int, ...]:
        return (2,)

    @classmethod
    def empty_data(cls) -> np.ndarray:
        return np.zeros(2, dtype=float)

    def is_empty(self) -> bool:
        return not np.all(np.isnan(self.data) | (self.data == 0))

    def flip(self, ctx: BranchGeoDataEditContext) -> VBranchTangents:
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

    def is_invalid(self, ctx: BranchGeoDataEditContext) -> str:
        curve = ctx.curve
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

    def is_empty(self) -> bool:
        return not len(self.data)

    def merge(self, others: List[VBranchBSpline], ctx: BranchGeoDataEditContext) -> VBranchBSpline:
        for other in others:
            self.data += other.data
        return self

    def flip(self, ctx: BranchGeoDataEditContext) -> VBranchBSpline:
        self.data = self.data.flip()
        return self

    def split(self, splits_point: List[Point], splits_id: List[int], ctx: BranchGeoDataEditContext) -> List[Self]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"VBranchBSpline({self.data})"

    def transform(self, projection: FundusProjection, ctx: BranchGeoDataEditContext) -> VBranchBSpline:
        bspline_data = [projection.transform(_.to_array()) for _ in self.data]
        return VBranchBSpline(BSpline.from_array(bspline_data))


####################################################################################################
class VBranchGeoDescriptor(NamedTuple):
    name: str
    geo_type: Optional[Type[VBranchGeoDataBase]] = None

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, VBranchGeoDescriptor):
            return self.name == value.name and self.geo_type == value.geo_type
        return self.name == str(value)

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
    CALIBRES = VBranchGeoDescriptor("CALIBRES", VBranchCurveData)

    #: The position of the left and right boundaries of the branch.
    BOUNDARIES = VBranchGeoDescriptor("BOUNDARIES", VBranchCurveData)

    #: The curvature of the branch at each skeleton point.
    CURVATURES = VBranchGeoDescriptor("CURVATURES", VBranchCurveData)

    #: The curvature roots of the branch.
    CURVATURE_ROOTS = VBranchGeoDescriptor("CURVATURE_ROOTS", VBranchCurveIndex)

    #: The B-spline representation of the branch.
    BSPLINE = VBranchGeoDescriptor("BSPLINE", VBranchBSpline)

    #: The tangents at the branches tips.
    TIPS_TANGENT = VBranchGeoDescriptor("TIPS_TANGENT", VBranchTipsTangents)

    #: The calibre at the branches tips.
    TIPS_CALIBRE = VBranchGeoDescriptor("TIPS_CALIBRE", VBranchTipsScalarData)

    #: The position of the left and right boundaries of the branches tips.
    TIPS_BOUNDARIES = VBranchGeoDescriptor("TIPS_BOUNDARIES", VBranchTipsDoublePointsData)

    @classmethod
    def by_type(cls, geo_type: Type[VBranchGeoDataBase]) -> VBranchGeoField:
        if issubclass(geo_type, VBranchTangents):
            return cls.TANGENTS
        if issubclass(geo_type, VBranchBSpline):
            return cls.BSPLINE
        if issubclass(geo_type, VBranchTipsTangents):
            return cls.TIPS_TANGENT

        raise ValueError(f"Impossible to infer VBranchGeoField from geo type: {geo_type}.")

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: object) -> bool:
        return self.name == str(value)


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
    TipsData = VBranchTipsData
    TipsTangents = VBranchTipsTangents
    TipsScalar = VBranchTipsScalarData
    Tips2Points = VBranchTipsDoublePointsData

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

    @staticmethod
    def geodata_type(name: Type[VBranchGeoDataBase] | str) -> Type[VBranchGeoDataBase]:
        """Return the type of a registered branch geometrical attribute.

        Parameters
        ----------
        name : str
            The name of the branch geometrical attribute.

        Returns
        -------
        Type[VBranchGeoDataBase]
            The type of the branch geometrical attribute.
        """
        if isinstance(name, str):
            try:
                return _registered_vbranch_geo_data_types[name]
            except KeyError:
                raise ValueError(f"Unknown branch geometrical attribute type: {name}.") from None
        elif issubclass(name, VBranchGeoDataBase):
            return name
        raise ValueError(f"Invalid type for branch geometrical attribute: {name}.")
