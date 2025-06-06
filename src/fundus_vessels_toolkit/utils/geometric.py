from __future__ import annotations

from functools import reduce
from typing import Iterable, List, NamedTuple, Tuple, TypeGuard, overload

import numpy as np
import numpy.typing as npt


class Rect(NamedTuple):
    h: float
    w: float
    y: float = 0
    x: float = 0

    @property
    def center(self) -> Point:
        return Point(self.y + self.h // 2, self.x + self.w // 2)

    @property
    def top_left(self) -> Point:
        return Point(self.y, self.x)

    @property
    def bottom_right(self) -> Point:
        return Point(self.y + self.h, self.x + self.w)

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def shape(self) -> Point:
        return Point(y=self.h, x=self.w)

    @property
    def area(self) -> float:
        return self.h * self.w

    def to_int(self):
        return Rect(*(int(round(_)) for _ in (self.h, self.w, self.y, self.x)))

    @classmethod
    def from_tuple(
        cls,
        rect: float
        | int
        | Tuple[float | int]
        | Tuple[float | int, float | int]
        | Tuple[float | int, float | int, float | int, float | int],
    ):
        if isinstance(rect, (float, int)):
            rect = (rect, rect)
        elif isinstance(rect, tuple) and len(rect) in (2, 4) and all(isinstance(_, (float, int)) for _ in rect):
            pass
        else:
            raise TypeError("Rect can only be created from a float or a tuple of 2 or 4 floats")
        return cls(*rect)

    @classmethod
    def from_size(cls, shape: Tuple[float | int, float | int]):
        return cls(shape[0], shape[1])

    @overload
    @classmethod
    def from_points(cls, bottom_right: Tuple[float | int, float | int]) -> Rect: ...

    @overload
    @classmethod
    def from_points(cls, bottom: float | int, right: float | int) -> Rect: ...

    @overload
    @classmethod
    def from_points(
        cls,
        top_left: Tuple[float | int, float | int],
        bottom_right: Tuple[float | int, float | int],
        *,
        ensure_positive: bool,
    ) -> Rect: ...

    @overload
    @classmethod
    def from_points(
        cls, top: float | int, left: float | int, bottom: float | int, right: float | int, *, ensure_positive: bool
    ) -> Rect: ...

    @overload
    @classmethod
    def from_points(
        cls, top_left_bottom_right: Tuple[float | int, float | int, float | int, float | int], *, ensure_positive: bool
    ) -> Rect: ...

    @classmethod
    def from_points(
        cls,
        *p: float | int | Tuple[float | int, float | int] | Tuple[float | int, float | int, float | int, float | int],
        ensure_positive: bool = False,
    ) -> Rect:
        if (
            len(p) == 1
            and isinstance(p[0], tuple)
            and len(p[0]) == 2
            and all(isinstance(_, (float, int)) for _ in p[0])
        ):
            # case ((bottom, right), )
            p2 = Point(p[0], p[1])
            p1 = Point.origin()
        elif len(p) == 2 and all(isinstance(_, (float, int)) for _ in p):
            # case (bottom, right)
            p2 = Point(p[0], p[1])
            p1 = Point.origin()
        elif (
            len(p) == 2
            and all(isinstance(_, tuple) for _ in p)
            and all(isinstance(_, (float, int)) for _ in p[0] + p[1])
        ):
            # case ((top, left), (bottom, right))
            p1 = Point(p[0])
            p2 = Point(p[1])
        elif len(p) == 4 and all(isinstance(_, (float, int)) for _ in p):
            # case (top, left, bottom, right)
            p1 = Point(p[0], p[1])
            p2 = Point(p[2], p[3])
        elif (
            len(p) == 1
            and isinstance(p[0], tuple)
            and len(p[0]) == 4
            and all(isinstance(_, (float, int)) for _ in p[0])
        ):
            # case ((top, left, bottom, right), )
            p1 = Point(p[0])
            p2 = Point(p[1])
        else:
            raise TypeError("Rect can only be created from 2 or 4 floats or from 2 tuples of 2 floats")

        if not ensure_positive:
            return cls(abs(p2.y - p1.y), abs(p2.x - p1.x), min(p1.y, p2.y), min(p1.x, p2.x))
        else:
            rect = cls(p2.y - p1.y, p2.x - p1.x, p1.y, p1.x)
            return Rect.empty() if rect.h < 0 or rect.w < 0 else rect

    @classmethod
    def from_center(cls, center: Tuple[float, float], shape: float | Tuple[float, float]) -> Rect:
        if isinstance(shape, (float, int)):
            shape = (shape, shape)
        return cls(shape[0], shape[1], center[0] - shape[0] // 2, center[1] - shape[1] // 2)

    @classmethod
    def empty(cls) -> Rect:
        return cls(0, 0, 0, 0)

    def is_self_empty(self) -> bool:
        return self.w == 0 or self.h == 0

    @classmethod
    def is_empty(cls, rect: Rect | None) -> bool:
        if rect is None:
            return True
        if isinstance(rect, tuple) and len(rect) == 4:
            rect = Rect(*rect)
        return isinstance(rect, tuple) and (rect.w == 0 or rect.h == 0)

    @classmethod
    def is_rect(cls, r) -> TypeGuard[Rect]:
        return isinstance(r, Rect) or (isinstance(r, tuple) and len(r) == 4)

    def __repr__(self):
        return "Rect(y={}, x={}, h={}, w={})".format(self.y, self.x, self.h, self.w)

    def __or__(self, other) -> Rect:
        if isinstance(other, Rect):
            if self.is_self_empty():
                return other
            if other.is_self_empty():
                return self
            return Rect.from_points(
                (min(self.top, other.top), min(self.left, other.left)),
                (max(self.bottom, other.bottom), max(self.right, other.right)),
            )
        else:
            raise TypeError("Rect can only be combined only with another Rect")

    def __and__(self, other) -> Rect:
        if isinstance(other, Rect):
            return Rect.from_points(
                (max(self.top, other.top), max(self.left, other.left)),
                (min(self.bottom, other.bottom), min(self.right, other.right)),
                ensure_positive=True,
            )
        else:
            raise TypeError("Rect can only be combined only with another Rect")

    def __bool__(self) -> bool:
        return not self.is_self_empty()

    def __add__(self, other: Point | float) -> Rect:
        if isinstance(other, float):
            other = Point(other, other)
        if isinstance(other, Point):
            return self.translate(other.y, other.x)
        raise TypeError("Rect can only be translated by a Point or a float")

    def __sub__(self, other: Point | float) -> Rect:
        if isinstance(other, float):
            other = Point(other, other)
        if isinstance(other, Point):
            return self.translate(-other.y, -other.x)
        raise TypeError("Rect can only be translated by a Point or a float")

    def __mul__(self, other: float) -> Rect:
        return self.scale(other)

    def __truediv__(self, other: float) -> Rect:
        return self.scale(1 / other)

    def __contains__(self, other: Point | Rect) -> bool:
        if isinstance(other, Point):
            return self.y <= other.y <= self.y + self.h and self.x <= other.x <= self.x + self.w
        elif isinstance(other, Rect):
            return not Rect.is_empty(self & other)
        else:
            raise TypeError("Rect can only be compared with a Point or a Rect")

    def translate(self, y: float, x: float) -> Rect:
        return Rect(self.h, self.w, self.y + y, self.x + x)

    def scale(self, fy: float, fx: float | None = None) -> Rect:
        if fx is None:
            fx = fy
        return Rect(self.h * fy, self.w * fx, self.y * fy, self.x * fx)

    def clip(self, rect: float | Tuple[float, float] | Tuple[float, float, float, float]) -> Rect:
        rect = Rect.from_tuple(rect)
        return Rect.from_points(
            (max(self.top, rect.top), max(self.left, rect.left)),
            (min(self.bottom, rect.bottom), min(self.right, rect.right)),
            ensure_positive=True,
        )

    @overload
    def pad(self, pad: float | Tuple[float, float]) -> Rect: ...

    @overload
    def pad(self, vertical: float, horizontal: float) -> Rect: ...

    @overload
    def pad(self, top: float, right: float, bottom: float, left: float) -> Rect: ...

    def pad(self, *pad: float | Tuple[float, float]) -> Rect:
        if len(pad) == 1 and isinstance(pad[0], (int, float)):
            pad = (pad[0],) * 4
        elif (
            len(pad) == 1
            and isinstance(pad[0], tuple)
            and len(pad[0]) == 2
            and all(isinstance(_, (int, float)) for _ in pad[0])
        ):
            # case ((vertical, horizontal), )
            pad = (pad[0][0], pad[0][1]) * 2
        elif len(pad) == 2 and all(isinstance(_, (int, float)) for _ in pad):
            # case (vertical, horizontal)
            pad = pad * 2
        elif len(pad) == 4 and all(isinstance(_, (int, float)) for _ in pad):
            # case (top, right, bottom, left)
            pass
        else:
            raise TypeError("Rect.pad() only accept 1, 2 or 4 floats as arguments")

        return Rect(self.h + pad[0] + pad[2], self.w + pad[1] + pad[3], self.y - pad[0], self.x - pad[3])

    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom

    def slice(self) -> tuple[slice, slice]:
        r = self.to_int()
        return slice(r.y, r.y + r.h), slice(r.x, r.x + r.w)

    @staticmethod
    def union(*rects: Tuple[Iterable[Rect] | Rect, ...]) -> Rect:
        rects = sum(((r,) if isinstance(r, Rect) else tuple(r) for r in rects), ())
        return reduce(lambda a, b: a | b, rects)

    @staticmethod
    def intersection(*rects: Tuple[Iterable[Rect] | Rect, ...]) -> Rect:
        rects = sum(((r,) if isinstance(r, Rect) else tuple(r) for r in rects), ())
        return reduce(lambda a, b: a & b, rects)


class Point(NamedTuple):
    y: float
    x: float

    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    def to_int(self) -> Point:
        return Point(int(round(self.y)), int(round(self.x)))

    def clip(self, rect: float | Tuple[float, float] | Tuple[float, float, float, float]) -> Point:
        rect = Rect.from_tuple(rect)
        return Point(
            min(max(self.y, rect.top), rect.bottom),
            min(max(self.x, rect.left), rect.right),
        )

    def __add__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y + other, self.x + other)
        y, x = other
        return Point(self.y + y, self.x + x)

    def __radd__(self, other: Tuple[float, float] | float):
        return self + other

    def __sub__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y - other, self.x - other)
        y, x = other
        return Point(self.y - y, self.x - x)

    def __rsub__(self, other: Tuple[float, float] | float):
        return -(self - other)

    def __mul__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y * other, self.x * other)
        y, x = other
        return Point(self.y * y, self.x * x)

    def __rmul__(self, other: Tuple[float, float] | float):
        return self * other

    def __truediv__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(self.y / other, self.x / other)
        y, x = other
        return Point(self.y / y, self.x / x)

    def __rtruediv__(self, other: Tuple[float, float] | float):
        if isinstance(other, (float, int)):
            return Point(other / self.y, other / self.x)
        y, x = other
        return Point(y / self.y, x / self.x)

    def __neg__(self):
        return Point(-self.y, -self.x)

    @classmethod
    def origin(cls):
        return cls(0, 0)

    @classmethod
    def from_tuple(cls, point: float | int | Tuple[float | int] | Tuple[float | int, float | int]):
        if isinstance(point, (float, int)):
            return cls(point, point)
        if len(point) == 1:
            if isinstance(point[0], (float, int)):
                return cls(point[0], point[0])
            if isinstance(point[0], tuple):
                return cls(*point[0])
        if len(point) == 2:
            return cls(*point)
        raise TypeError("Point can only be created from a float or a tuple of 2 floats")

    @overload
    def distance(self, other: Point) -> float: ...

    @overload
    def distance(self, other: List[Point]) -> List[float]: ...

    @overload
    def distance(self, other: npt.NDArray[np.float]) -> npt.NDArray[np.float]: ...

    def distance(self, other: Point | Iterable[Point]) -> float | Iterable[float]:
        import numpy as np

        if isinstance(other, np.ndarray):
            return np.linalg.norm(other - self, axis=-1)
        elif isinstance(other, list):
            return [self.distance(p) for p in other]
        return ((self.y - other.y) ** 2 + (self.x - other.x) ** 2) ** 0.5


def distance_matrix(points_coord: np.ndarray):
    """
    Compute the distance matrix between a set of points.

    Parameters
    ----------
    nodes_coord: A 2D array of shape (nb_nodes, 2) containing the coordinates of the nodes.

    Returns
    -------
    A 2D array of shape (nb_nodes, nb_nodes) containing the distance between each pair of nodes.
    """
    return np.linalg.norm(points_coord[:, None, :] - points_coord[None, :, :], axis=2)
