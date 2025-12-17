from typing import List, Tuple, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

RecursiveIntList: TypeAlias = List[int] | List["RecursiveIntList"]
IntArrayLike: TypeAlias = npt.NDArray[np.int32] | int | RecursiveIntList
Int1DArrayLike: TypeAlias = npt.NDArray[np.int32] | int | List[int]
Int2DArrayLike: TypeAlias = npt.NDArray[np.int32] | List[int] | List[List[int]]
Int3DArrayLike: TypeAlias = npt.NDArray[np.int32] | List[List[int]] | List[List[List[int]]]
Int1DArray: TypeAlias = npt.NDArray[np.int32]
Int2DArray: TypeAlias = npt.NDArray[np.int32]


RecursiveBoolList: TypeAlias = List[bool] | List["RecursiveBoolList"]
BoolArrayLike: TypeAlias = npt.NDArray[np.bool_] | bool | RecursiveBoolList
Bool1DArrayLike: TypeAlias = npt.NDArray[np.bool_] | bool | List[bool]
Bool2DArrayLike: TypeAlias = npt.NDArray[np.bool_] | List[bool] | List[List[bool]]
Bool1DArray: TypeAlias = npt.NDArray[np.bool_]

RecursiveFloatList: TypeAlias = List[float] | List["RecursiveFloatList"]
FloatArrayLike: TypeAlias = npt.NDArray[np.float64] | float | RecursiveFloatList
Float1DArrayLike: TypeAlias = npt.NDArray[np.float64] | float | List[float]
Float2DArrayLike: TypeAlias = npt.NDArray[np.float64] | List[float] | List[List[float]]
Float3DArrayLike: TypeAlias = npt.NDArray[np.float64] | List[List[float]] | List[List[List[float]]]
Float1DArray: TypeAlias = npt.NDArray[np.float64]
Float2DArray: TypeAlias = npt.NDArray[np.float64]

PointLike: TypeAlias = Tuple[int, int] | List[int] | npt.NDArray[np.int32]
PointArrayLike: TypeAlias = (
    npt.NDArray[np.int32] | List[int] | List[List[int]] | Tuple[int, int] | List[Tuple[int, int]]
)

IntPairArrayLike: TypeAlias = (
    npt.NDArray[np.int32] | List[int] | List[List[int]] | Tuple[int, int] | List[Tuple[int, int]]
)
BoolPairArrayLike: TypeAlias = (
    npt.NDArray[np.bool_] | List[bool] | List[List[bool]] | Tuple[bool, bool] | List[Tuple[bool, bool]]
)

Index: TypeAlias = int
Indices: TypeAlias = npt.NDArray[np.int32] | List[int] | pd.Series
IndicesLike: TypeAlias = Index | Int1DArrayLike
