import abc
from typing import Dict, List, Mapping, Optional, Self, Tuple, Type

import numpy as np

from .geometric import Rect


def _np_short_str(arr: np.ndarray) -> str:
    if arr.ndim == 2:
        return "[" + "| ".join([" ".join(f"{v:.2f}" for v in row) for row in arr]) + "]"
    elif arr.ndim == 1:
        return "[" + " ".join(f"{v:.2f}" for v in arr) + "]"
    return str(arr)


class FundusProjection(abc.ABC):
    @classmethod
    def identity(cls) -> Self:
        """
        Returns the identity projection model.

        Returns
        -------
        projection : Self
            The identity projection model.
        """
        return IdentityProjection()

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray) -> Tuple[Self, float]:
        """
        Fits a projection model to map points from ``src`` to ``dst``.

        Parameters
        ----------
        src : np.ndarray
            The source points coordinates (N x 2) where N is the number of points.

        dst : np.ndarray
            The destination points coordinates (N x 2).

        Returns
        -------
        projection : Self
            The fitted projection model.

        error : float
            The mean square error of the fitted model.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement the 'fit' method")

    @staticmethod
    def fit_to_projection(
        src: np.ndarray, dst: np.ndarray, projection: Type[Self] | Dict[int, Type[Self]]
    ) -> Tuple[Self, float]:
        """Fits a given projection model to map points from ``src`` to ``dst``.

        Parameters
        ----------
        src : np.ndarray
            The source points coordinates (N, 2) where N is the number of points.

        dst : np.ndarray
            The destination points coordinates (N, 2).

        projection : Type[Self] | Dict[int, Type[Self]]
            The projection model to fit. If a dictionary is provided, the key is the minimum number of inliers required to use the corresponding projection.

        Returns
        -------
        Tuple[Self, float]
            The fitted projection model and the mean square error of the fitted model.

        """  # noqa: E501

        if isinstance(projection, Mapping):
            projection = {k: projection[k] for k in sorted(projection.keys(), reverse=True)}

            n_inliers = src.shape[0]
            proj = None
            for k, p in projection.items():
                proj = p
                if k <= n_inliers:
                    break
            else:
                raise ValueError("No projection model matches the number of inliers")
            return proj.fit(src, dst)
        elif issubclass(projection, FundusProjection):
            return projection.fit(src, dst)

        raise ValueError("projection must be a projection model or a dictionary of projection models")

    def compose(self, T1: Self) -> Self:
        """
        Composes this projection model with another one.

        Parameters
        ----------
        T1 : Self
            The other projection model to compose with.

        Returns
        -------
        T : Self
            The composed projection model: T = self @ T1.
        """
        return ProjectionComposition.simplify_composition(self, T1)

    def invert(self) -> Self:
        """
        Inverts this projection model.

        Returns
        -------
        T : Self
            The inverted projection model: T
        """
        return ProjectionInverse(self)

    @property
    def is_exact(self) -> bool:
        """
        Whether this projection model is exact (it doesn't provide an approximation for example using Newton algorithm).
        """
        return True

    @property
    def is_inverse_exact(self) -> bool:
        """
        Whether the inverse of this projection model is exact.
        (I.e. it doesn't provide an approximation for example using Newton algorithm).
        """
        return True

    @abc.abstractmethod
    def transform(self, src: np.ndarray) -> np.ndarray:
        """
        Transforms a set of points with this projection model.

        Parameters
        ----------
        src : np.ndarray
            The source points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        dst : np.ndarray
            The transformed points coordinates (N x 2).
        """
        pass

    def transform_inverse(self, dst: np.ndarray) -> np.ndarray:
        """
        Transforms a set of points with the inverse of this projection model.

        Parameters
        ----------
        dst : np.ndarray
            The destination points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        src : np.ndarray
            The source points coordinates (N x 2).
        """
        invert_t = self.invert()
        if isinstance(invert_t, ProjectionInverse):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement the 'transform_inverse' nor the 'invert' methods"
            )
        return invert_t.transform(dst)

    def transform_domain(self, moving_domain: Rect) -> Rect:
        """
        Transforms a domain with this projection model.

        Parameters
        ----------
        src_domain : Rect
            The source domain to transform

        Returns
        -------
        Rect
            The transformed domain.
        """
        corners = self.transform(moving_domain.corners())
        return Rect.from_points(tuple(np.amin(corners, axis=0)), tuple(np.amax(corners, axis=0))).to_int()

    def quadratic_error(self, src: np.ndarray, dst: np.ndarray, mean: bool = False) -> np.ndarray | float:
        """
        Calculates the quadratic error of the projection model when mapping points from ``src`` to ``dst``.

        Parameters
        ----------
        src : np.ndarray
            The source points coordinates (N x 2) where N is the number of points.

        dst : np.ndarray
            The destination points coordinates (N x 2).

        mean : bool, optional
            Whether to return the mean error. By default False.

        Returns
        -------
        error : np.ndarray | float
            The quadratic error of each point or the mean error if ``mean`` is True.
        """
        errors = np.sum((dst - self.transform(src)) ** 2, axis=1)
        return np.mean(errors) if mean else errors

    def warp(
        self, src_img: np.ndarray, src_domain: Optional[Rect] = None, warped_domain: Optional[Rect] = None
    ) -> Tuple[np.ndarray, Rect]:
        """
        Warps an image using this projection model.

        Parameters
        ----------
        src_img : np.ndarray
            The source image to warp. The image must be cv2 compatible: shape=(H x W [x C]) and dtype=np.uint8|np.float32.

        src_domain : Rect
            The domain of the source image. If None, the domain is inferred from the image shape.

        warped_domain : Optional[Rect], optional
            The domain of the destination image. If None, the destination domain is computed by transforming ``src_domain``.

        Returns
        -------
        dst_img : np.ndarray
            The warped image.

        warped_domain : Rect
            The domain of the warped image.
        """  # noqa: E501
        from ..utils.safe_import import import_cv2

        cv2 = import_cv2()

        if src_domain is None:
            src_domain = Rect.from_size(src_img.shape[:2])
        if warped_domain is None:
            warped_domain = self.transform_domain(src_domain)

        yy, xx = np.mgrid[warped_domain.slice()]
        dst_yx = np.column_stack((yy.ravel(), xx.ravel()))
        dst_yx -= src_domain.top_left.numpy()[None, :]
        src_map = self.transform_inverse(dst_yx).reshape(warped_domain.shape + (2,))
        src_map = src_map.astype(np.float32)[..., ::-1]

        def warp(img):
            return cv2.remap(img, src_map, None, cv2.INTER_LINEAR)

        return warp(src_img) if isinstance(src_img, np.ndarray) else [warp(_) for _ in src_img], warped_domain


class ProjectionComposition(FundusProjection):
    def __init__(self, *Ts: Tuple[FundusProjection]) -> None:
        self.Ts = Ts
        super().__init__()

    def __repr__(self) -> str:
        return f"ProjectionComposition({', '.join(repr(T) for T in self.Ts)})"

    def __str__(self) -> str:
        return " @ ".join(str(T) for T in self.Ts)

    @staticmethod
    def simplify_composition(*Ts: FundusProjection) -> FundusProjection:
        expanded_transforms = []
        for T in Ts:
            if isinstance(T, ProjectionComposition):
                expanded_transforms.extend(T.Ts)
            elif not isinstance(T, IdentityProjection):
                expanded_transforms.append(T)
        Ts = expanded_transforms

        simplified = True
        while simplified:
            simplified = False
            i = 0
            while i < len(Ts) - 1:
                T1, T2 = Ts[i], Ts[i + 1]
                if (isinstance(T1, ProjectionInverse) and T1.T is T2) or (
                    isinstance(T2, ProjectionInverse) and T2.T is T1
                ):
                    simplified = True
                    del Ts[i + 1]
                    del Ts[i]
                else:
                    i += 1

        if not Ts:
            return IdentityProjection()
        if len(Ts) == 1:
            return Ts[0]
        return ProjectionComposition(*Ts)

    def compose(self, T1: FundusProjection) -> Self:
        if isinstance(T1, ProjectionComposition):
            return ProjectionComposition.simplify_composition(*self.T, *T1.Ts)
        return ProjectionComposition.simplify_composition(*self.T, T1)

    @property
    def is_exact(self) -> bool:
        return all(T.is_exact for T in self.Ts)

    @property
    def is_inverse_exact(self) -> bool:
        return all(T.is_inverse_exact for T in self.Ts)

    def invert(self) -> Self:
        return ProjectionComposition(*(T.invert() for T in reversed(self.Ts)))

    def transform(self, src: np.ndarray) -> np.ndarray:
        for T in self.Ts:
            src = T.transform(src)
        return src

    def transform_inverse(self, dst: np.ndarray) -> np.ndarray:
        for T in reversed(self.Ts):
            dst = T.transform_inverse(dst)
        return dst


class ProjectionInverse(FundusProjection):
    def __init__(self, T: FundusProjection) -> None:
        self.T = T
        super().__init__()

    def __repr__(self) -> str:
        return f"ProjectionInverse({self.T})"

    def __str__(self) -> str:
        return f"Inv[{self.T}]"

    def invert(self) -> Self:
        return self.T

    @property
    def is_exact(self) -> bool:
        return self.T.is_inverse_exact

    @property
    def is_inverse_exact(self) -> bool:
        return self.T.is_exact

    def transform(self, src: np.ndarray) -> np.ndarray:
        return self.T.transform_inverse(src)

    def transform_inverse(self, dst: np.ndarray) -> np.ndarray:
        return self.T.transform(dst)


class IdentityProjection(FundusProjection):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "IdentityProjection()"

    def __str__(self) -> str:
        return "I"

    def invert(self) -> Self:
        return IdentityProjection()

    def compose(self, T1: Self) -> Self:
        return T1

    def transform(self, src: np.ndarray) -> np.ndarray:
        return src

    def transform_inverse(self, dst: np.ndarray) -> np.ndarray:
        return dst


class AffineProjection(FundusProjection):
    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        assert R.shape == (2, 2) and t.shape == (2,), "R must be a 2x2 matrix and t must be a 2D vector"
        self.R = R
        self.t = t
        super().__init__()

    def __repr__(self) -> str:
        return f"AffineProjection(R={self.R}, t={self.t})"

    def __str__(self) -> str:
        return f"Affine(R={_np_short_str(self.R)}, t={_np_short_str(self.t)})"

    @classmethod
    def rotate(cls, theta: float, center: np.ndarray = (0, 0)) -> Self:
        """Create an affine transformation that rotates by theta and translates by t.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. Positive values rotate clockwise.
        t : np.ndarray
            Translation vector.

        Returns
        -------
        AffineProjection
            The corresponding affine transformation.
        """
        theta = np.deg2rad(theta)
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        center = np.asarray(center)
        t = center - R @ center
        return cls(R, t)

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray) -> Tuple[Self, float]:
        src, dst = np.asarray(src), np.asarray(dst)
        assert src.ndim == 2 and src.shape[1] == 2, "src must be a 2D array of 2D coordinates"
        assert src.shape == dst.shape, "src and dst must have the same shape"
        src = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
        X, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        R, t = X[:2].T, X[2]
        error = np.sum((dst - src @ X) ** 2, axis=1)
        return cls(R, t), np.mean(error)

    def compose(self, T1: FundusProjection) -> FundusProjection:
        if isinstance(T1, AffineProjection):
            return self.__class__(self.R @ T1.R, self.R @ T1.t + self.t)
        return super().compose(T1)

    def invert(self) -> Self:
        X = np.concatenate((self.R, self.t[:, None]), axis=1)
        X = np.concatenate((X, [[0, 0, 1]]), axis=0)
        X_inv = np.linalg.inv(X)
        R_inv, t_inv = X_inv[:2, :2], X_inv[:2, 2]
        return self.__class__(R_inv, t_inv)

    def transform(self, src: np.ndarray) -> np.ndarray:
        src = np.asarray(src)
        return (self.R @ src.T + self.t[:, None]).T

    def warp(
        self,
        src_img: np.ndarray | List[np.ndarray],
        src_domain: Optional[Rect] = None,
        warped_domain: Optional[Rect] = None,
    ) -> Tuple[np.ndarray, Rect]:
        from ..utils.safe_import import import_cv2

        cv2 = import_cv2()

        if src_domain is None:
            src_domain = Rect.from_size(src_img.shape[:2])
        if warped_domain is None:
            warped_domain = self.transform_domain(src_domain)

        t = self.t - warped_domain.top_left + src_domain.top_left
        M = np.concatenate((self.R[::-1, ::-1], t[::-1, None]), axis=1)

        def warp(img):
            return cv2.warpAffine(img, M, warped_domain.size.xy, flags=cv2.INTER_LINEAR)

        return warp(src_img) if isinstance(src_img, np.ndarray) else [warp(_) for _ in src_img], warped_domain


class QuadraticProjection(FundusProjection):
    """
    A quadratic projection model that maps points from a source to a destination using a quadratic transformation.

    The transformation is defined as:
        dst = [Q, R, t] @ [src.y², src.x², src.x*src.y, src.y, src.x, 1].T

    """

    def __init__(self, Q: np.ndarray, R: np.ndarray, t: np.ndarray) -> None:
        Q, R, t = np.asarray(Q), np.asarray(R), np.asarray(t)
        assert (
            Q.shape == (2, 3) and R.shape == (2, 2) and t.shape == (2,)
        ), "Q must be a 2x3 matrix, R must be a 2x2 matrix and t must be a 2D vector"
        self.Q = Q
        self.R = R
        self.t = t
        self._inverse_transform: bool | None | QuadraticProjection = False
        super().__init__()

    def __repr__(self) -> str:
        return f"QuadraticProjection(Q={self.Q}, R={self.R}, t={self.t})"

    def __str__(self) -> str:
        return f"Quadratic(Q={_np_short_str(self.Q)}, R={_np_short_str(self.R)}, t={_np_short_str(self.t)})"

    @property
    def is_inverse_exact(self) -> bool:
        return False

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray) -> Tuple[Self, float]:
        src_y = src[:, 0]
        src_x = src[:, 1]
        src_ = np.stack((src_y**2, src_x**2, src_x * src_y, src_y, src_x, np.ones((src.shape[0],))), axis=1)
        X, _, _, _ = np.linalg.lstsq(src_, dst, rcond=None)
        Q, R, t = X[:3].T, X[3:5].T, X[5]

        # if np.any(abs(Q) > 1e-5):
        error = np.sum((dst - src_ @ X) ** 2, axis=1)
        return cls(Q, R, t), np.mean(error)
        # else:
        #    T = AffineProjection(R.T, t)
        #    return T, np.mean(T.quadratic_error(src, dst))

    def transform(self, src: np.ndarray) -> np.ndarray:
        src = np.asarray(src)
        src_y, src_x = src[:, 0], src[:, 1]
        src_yy_xx_yx = np.stack((src_y**2, src_x**2, src_x * src_y), axis=1)
        return (self.Q @ src_yy_xx_yx.T + self.R @ src.T + self.t[:, None]).T

    def jacobian(self, src: np.ndarray) -> np.ndarray:
        src = np.asarray(src)
        return self.R[None, :, :] + (self.Q[None, :, 2, None] + 2 * self.Q[None, :, :2]) * src[:, None, :]

    def _eval_inverse_transform(self) -> Self | None:
        # Sample points to estimate the inverse transformation
        src = np.mgrid[0:1000:100, 0:1000:100].reshape(2, -1).T
        dst = self.transform(src)
        invT, error = QuadraticProjection.fit(dst, src)
        invT._inverse_transform = self
        return None if error > 1 else invT

    def transform_inverse_newton(self, dst: np.ndarray) -> np.ndarray:
        # initial guess using only the affine part
        x = AffineProjection(self.R, self.t).transform_inverse(dst)

        NITERS = 20
        TOL = 1

        # Newton's method
        for _ in range(NITERS):
            p = self.transform(x)
            t = dst - p
            if np.all(np.linalg.norm(t, axis=-1) < TOL):
                break

            J = self.jacobian(x)
            dx = np.linalg.solve(J, t)
            x += dx
            if np.all(np.linalg.norm(dx, axis=-1) < TOL):
                break

        return x

    def transform_inverse(self, dst: np.ndarray) -> np.ndarray:
        if self._inverse_transform is False:
            self._inverse_transform = self._eval_inverse_transform()
        if self._inverse_transform is None:
            return self.transform_inverse_newton(dst)
        return self._inverse_transform.transform(dst)


def ransac_fit_projection(
    fix: np.ndarray,
    moving: np.ndarray,
    sampling_probability: Optional[np.ndarray] = None,
    initial_projection: Type[FundusProjection] = AffineProjection,
    final_projection: Optional[Type[FundusProjection] | Dict[int, Type[FundusProjection]]] = None,
    *,
    n: int = 4,
    initial_inliers_tolerance: float = 5,
    min_initial_inliers: int | float = 0.5,
    final_inliers_tolerance: Optional[float] = None,
    max_iterations: int = 300,
    early_stop_mean_error: float = 1,
    early_stop_min_inliers: int = 0.5,
    rng: np.random.Generator = None,
) -> Tuple[FundusProjection, float, np.ndarray]:
    """
    Estimates a 2D transformation matrix that maps points from ``src`` to ``dst`` using the RANSAC algorithm.

    Parameters
    ----------
        fix: np.ndarray
            Coordinates of the fix points (N x 2) where N is the number of points.

        moving: np.ndarray
            Coordinates of the moving points (N x 2) where N is the number of points.

        sampling_probability:
            Probability of sampling each point. If None, all points are sampled with the same probability.

        initial_projection: Type[FundusProjection], optional
            The type of projection to use for the initial estimation.

        final_projection: Type[FundusProjection] | Dict[int, Type[FundusProjection]], optional
            The type of projection to use for the final estimation.

            - If a dictionary is provided, the key is the minimum number of inliers required to use the corresponding projection.
            - If None (by default), the initial projection is used for the final estimation.

        n: int, optional
            Number of points to sample for each iteration.

        initial_inliers_tolerance: float, optional
            Maximum distance between the transformed points and the destination points to consider them as inliers.

        min_initial_inliers: int | float, optional
            Minimum number of inliers required to consider the transformation as valid. If a float, it is interpreted as a ratio of the total number of points.

        final_inliers_tolerance: float, optional
            Maximum distance between the transformed points and the destination points to consider them as inliers in the final estimation. The returned transformation is the one with the most of such inliers.

        max_iterations: int, optional
            Maximum number of iterations.

        early_stop_mean_error: float, optional
            Mean distance under which the algorithm should stop early.

        early_stop_min_inliers: int | float, optional
            Minimum number of inliers required to stop the algorithm early. If a float, it is interpreted as a ratio of the total number of points.

        rng: np.random.Generator, optional
            Random number generator.

    Returns
    -------
        T: FundusProjection
            The best transformation of type ``final_projection`` found to map the moving points to the fix points.

        error: float
            Mean distance of the best transformation.

        inliers: np.ndarray
            Indices of the points that are considered inliers.

    Raises
    ------
        ValueError
            If no transformation matches the criteria
    """  # noqa: E501
    assert moving.ndim == 2 and moving.shape[1] == 2, "moving must be a 2D array of 2D coordinates"
    assert moving.shape == fix.shape, "moving and fix must have the same shape"
    N = moving.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(final_projection, Mapping):
        final_projection[-1] = initial_projection
    elif final_projection is None:
        final_projection = initial_projection

    initial_inliers_tolerance = initial_inliers_tolerance**2
    final_inliers_tolerance = (
        initial_inliers_tolerance if final_inliers_tolerance is None else final_inliers_tolerance**2
    )
    early_stop_mean_error = early_stop_mean_error**2

    if min_initial_inliers < 1:
        min_initial_inliers = int(min_initial_inliers * (N - n))
    if early_stop_min_inliers < 1:
        early_stop_min_inliers = int(early_stop_min_inliers * (N - n))

    best_T = None
    best_mean_error = np.inf
    best_inliers = []

    for _ in range(max_iterations):
        # Sample n points
        if sampling_probability is not None:
            idx = rng.choice(N, n, replace=False, p=sampling_probability)
            idx = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)])
        else:
            idx = np.arange(N)
            rng.shuffle(idx)

        # Estimate initial transformation
        iniT, _ = initial_projection.fit(moving[idx[:n]], fix[idx[:n]])

        # Apply initial transformation to all other points and calculate error
        errors = iniT.quadratic_error(moving[idx[n:]], fix[idx[n:]])

        # Check if initial transformation m is valid
        n_inliers = np.sum(errors < initial_inliers_tolerance) + n
        if n_inliers < max(min_initial_inliers + n, len(best_inliers)):
            continue

        # Optimize transformation using all inliers and the final projection
        inliers = np.concatenate([idx[:n], idx[n:][errors < initial_inliers_tolerance]])
        T, mean_error = FundusProjection.fit_to_projection(moving[inliers], fix[inliers], final_projection)
        errors = np.sum((fix - T.transform(moving)) ** 2, axis=1)

        inliers = np.where(errors < final_inliers_tolerance)[0]

        # Save transformation if it is better
        if len(inliers) > len(best_inliers) or mean_error < best_mean_error:
            best_T = T
            best_inliers = inliers
            best_mean_error = mean_error

            # Early stop if the error is below the tolerance
            if best_mean_error < early_stop_mean_error and len(best_inliers) >= early_stop_min_inliers:
                break
    else:
        # If the loop completes without early stopping, recompute the transformation using all best inliers
        T, mean_error = FundusProjection.fit_to_projection(moving[best_inliers], fix[best_inliers], final_projection)
        if mean_error < best_mean_error:
            best_T = T
            best_mean_error = mean_error

    if best_T is None:
        raise ValueError("RANSAC algorithm failed: no transformation matched the criteria")

    return best_T, np.sqrt(best_mean_error), best_inliers


########################################################################################################################
def fit_mse_affine_tranform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Calculates the least-squares best-fit translation and rotation that maps corresponding points ``source`` to ``dest``.

    Return the transformation matrix X which solve: dst.T = T @ [src, 1].T

    Parameters
    ----------
    src : np.ndarray
        Source points (N x m) where N is the number of points and m is the number of dimensions.

    dst : np.ndarray
        Destination points (N x m).

    Returns
    -------
    T : np.ndarray
        Homogeneous transformation matrix (m x m+1). E.g. for 2D points, the matrix is:
            [[cos(theta), -sin(theta), ty],
             [sin(theta),  cos(theta), tx]]

    """  # noqa: E501

    assert src.ndim == 2 and dst.ndim == 2, "src and dst must be 2D arrays"
    assert src.shape == dst.shape, "src and dst must have the same shape"
    N, m = src.shape

    src = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
    # dst = np.concatenate((dst, np.ones((dst.shape[0], 1))), axis=1)
    return np.linalg.lstsq(src, dst, rcond=None)[0].T


def apply_affine_transform(src: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Applies a 2D affine transformation matrix to a set of points following the formula:
        ``dst.T = T @ [src, 1].T``.

    Parameters
    ----------
    src : np.ndarray
        Source points (N x m) where N is the number of points and m is the number of dimensions.

    T : np.ndarray
        Affine transformation matrix (m x m+1).

    Returns
    -------
    dst : np.ndarray
        Transformed points (N x m).
    """
    src = np.asarray(src)
    assert src.ndim == 2, "src must be a 2D array"
    _, m = src.shape
    T_square = T[:, :m]
    dst = T_square @ src.T + T[:, m, None]
    return dst.T


def compose_affine_transforms(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    Composes two 2D affine transformation matrices.

    Parameters
    ----------
    T1 : np.ndarray
        First affine transformation matrix (m x m+1).

    T2 : np.ndarray
        Second affine transformation matrix (m x m+1).

    Returns
    -------
    T : np.ndarray
        Composed affine transformation matrix (m x m+1).
    """
    assert T1.ndim == T2.ndim == 2, "T1 and T2 must be 2D arrays"
    assert T1.shape == T2.shape, "T1 and T2 must have the same shape"

    R1, t1 = T1[:, :-1], T1[:, -1]
    R2, t2 = T2[:, :-1], T2[:, -1]

    return np.concatenate((R1 @ R2, R1 @ t2[:, None] + t1[:, None]), axis=1)
