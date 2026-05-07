from __future__ import annotations

from types import EllipsisType
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

type Int1DArray = np.ndarray[tuple[int], np.dtype[np.int_]]
type Int2DArray = np.ndarray[tuple[int, int], np.dtype[np.int_]]
type Int3DArray = np.ndarray[tuple[int, int, int], np.dtype[np.int_]]
type RecursiveIntlist = list[int] | list[RecursiveIntlist]
type IntArrayLike = npt.NDArray[np.integer] | int | RecursiveIntlist
type Int1DArrayLike = npt.NDArray[np.integer] | int | list[int]
type Int2DArrayLike = npt.NDArray[np.integer] | list[int] | list[list[int]]
type Int3DArrayLike = npt.NDArray[np.integer] | list[list[int]] | list[list[list[int]]]

type Bool1DArray = np.ndarray[tuple[int], np.dtype[np.bool_]]
type Bool2DArray = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
type RecursiveBoollist = list[bool] | list[RecursiveBoollist]
type BoolArrayLike = npt.NDArray[np.bool_] | bool | RecursiveBoollist
type Bool1DArrayLike = npt.NDArray[np.bool_] | bool | list[bool]
type Bool2DArrayLike = npt.NDArray[np.bool_] | list[bool] | list[list[bool]]


type RecursiveFloatlist = list[float] | list[RecursiveFloatlist]
type FloatArrayLike = npt.NDArray[np.floating] | float | RecursiveFloatlist
type Float1DArray = np.ndarray[tuple[int], np.dtype[np.float64]]
type Float2DArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Float3DArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
type Float1DArrayLike = npt.NDArray[np.floating] | float | list[float]
type Float2DArrayLike = npt.NDArray[np.floating] | list[float] | list[list[float]]
type Float3DArrayLike = npt.NDArray[np.floating] | list[list[float]] | list[list[list[float]]]


def as_float_1d(arr: Float1DArrayLike) -> Float1DArray:
    arr = np.atleast_1d(arr).astype(np.float64)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_float_2d(arr: Float2DArrayLike) -> Float2DArray:
    arr = np.atleast_2d(arr).astype(np.float64)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_float_3d(arr: Float3DArrayLike) -> Float3DArray:
    arr = np.atleast_3d(arr).astype(np.float64)
    assert arr.ndim == 3, f"Expected a 3D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_bool_1d(arr: Bool1DArrayLike) -> Bool1DArray:
    arr = np.atleast_1d(arr).astype(np.bool_)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_bool_2d(arr: Bool2DArrayLike) -> Bool2DArray:
    arr = np.atleast_2d(arr).astype(np.bool_)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_1d(arr: Int1DArrayLike) -> Int1DArray:
    arr = np.atleast_1d(arr).astype(np.int_)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_2d(arr: Int2DArrayLike) -> Int2DArray:
    arr = np.atleast_2d(arr).astype(np.int_)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_3d(arr: Int3DArrayLike) -> Int3DArray:
    arr = np.atleast_3d(arr).astype(np.int_)
    assert arr.ndim == 3, f"Expected a 3D array, got array with shape {arr.shape}"
    return arr  # type: ignore


type FloatPair = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type FloatPairLike = tuple[float, float] | list[float] | npt.NDArray[np.floating]

type FloatPairArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
type FloatPairArrayLike = (
    npt.NDArray[np.floating]
    | list[float]
    | list[list[float]]
    | tuple[float, float]
    | Sequence[tuple[float, float]]
    | dict[int, float]
)
type FloatPairMap = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.float64]]
type FloatPairVolume = np.ndarray[tuple[int, int, int, Literal[2]], np.dtype[np.float64]]
type PointArray = FloatPairArray
type PointArrayLike = FloatPairArrayLike

type IntPairArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int_]]
type IntPairArrayLike = (
    npt.NDArray[np.integer] | list[int] | list[list[int]] | tuple[int, int] | Sequence[tuple[int, int]] | dict[int, int]
)
type IntPairMap = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.int_]]
type BoolPairArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.bool_]]
type BoolPairArrayLike = (
    npt.NDArray[np.bool_] | list[bool] | list[list[bool]] | tuple[bool, bool] | Sequence[tuple[bool, bool]]
)

type Index = int
type Indices = npt.NDArray[np.int_]
type IndicesLike = Index | npt.NDArray[np.integer] | list[int] | pd.Series


def as_float_pair(pair: FloatPairLike) -> FloatPair:
    arr = np.asarray(pair, dtype=np.float64)
    assert arr.shape == (2,), "The provided input cannot be converted to a 2D point."
    return arr  # type: ignore


def as_int_pairs(pairs: IntPairArrayLike) -> IntPairArray:
    arr = np.array(list(pairs.items()), dtype=np.int_) if isinstance(pairs, dict) else np.asarray(pairs, dtype=np.int_)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.ndim == 2 and arr.shape[1] == 2, "The provided input cannot be converted to a 2D array of shape (N, 2)."
    return arr  # type: ignore


def as_int_pairs_map(pairs: Int3DArrayLike) -> IntPairMap:
    arr = np.asarray(pairs, dtype=np.int_)
    assert arr.ndim == 3 and arr.shape[2] == 2, (
        "The provided input cannot be converted to a 3D array of shape (N, M, 2)."
    )
    return arr  # type: ignore


def as_float_pairs(pairs: FloatPairArrayLike, copy: bool | None = None) -> FloatPairArray:
    arr = (
        np.array(list(pairs.items()), dtype=np.float64)
        if isinstance(pairs, dict)
        else np.asarray(pairs, dtype=np.float64, copy=copy)
    )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.ndim == 2 and arr.shape[1] == 2, "The provided input cannot be converted to a 2D array of shape (N, 2)."
    return arr  # type: ignore


def as_float_pairs_map(pairs: Float3DArrayLike) -> FloatPairMap:
    arr = np.asarray(pairs, dtype=np.float64)
    assert arr.ndim == 3 and arr.shape[2] == 2, (
        "The provided input cannot be converted to a 3D array of shape (N, M, 2)."
    )
    return arr  # type: ignore
