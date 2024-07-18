import abc
from typing import Optional, Self, Tuple, Type

import numpy as np

from .geometric import Rect


class FundusProjection(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def identity(cls) -> Self:
        """
        Returns the identity projection model.

        Returns
        -------
        projection : Self
            The identity projection model.
        """
        pass

    @classmethod
    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def invert(self) -> Self:
        """
        Inverts this projection model.

        Returns
        -------
        T : Self
            The inverted projection model: T
        """
        pass

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

    def transform_domain(self, src_domain: Rect) -> Rect:
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
        corners = self.transform(src_domain.corners())
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

    @abc.abstractmethod
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
        pass


class AffineProjection(FundusProjection):
    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        assert R.shape == (2, 2) and t.shape == (2,), "R must be a 2x2 matrix and t must be a 2D vector"
        self.R = R
        self.t = t
        super().__init__()

    @classmethod
    def identity(cls) -> Self:
        return cls(np.eye(2), np.zeros(2))

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray) -> Tuple[Self, float]:
        src = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
        X, error, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        R, t = X[:2].T, X[2]
        return cls(R, t), np.mean(error)

    def compose(self, T1: Self) -> Self:
        return self.__class__(self.R @ T1.R, self.R @ T1.t + self.t)

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
        self, src_img: np.ndarray, src_domain: Optional[Rect] = None, warped_domain: Optional[Rect] = None
    ) -> Tuple[np.ndarray, Rect]:
        from ..utils.safe_import import import_cv2

        cv2 = import_cv2()

        if src_domain is None:
            src_domain = Rect.from_size(src_img.shape[:2])
        if warped_domain is None:
            warped_domain = self.transform_domain(src_domain)

        t = self.t - warped_domain.top_left + src_domain.top_left
        M = np.concatenate((self.R[::-1, ::-1], t[::-1, None]), axis=1)
        dst_img = cv2.warpAffine(src_img, M, warped_domain.shape.xy, flags=cv2.INTER_LINEAR)

        return dst_img, warped_domain


def ransac_fit_projection(
    src: np.ndarray,
    dst: np.ndarray,
    initial_projection: Type[FundusProjection] = AffineProjection,
    final_projection: Type[FundusProjection] = AffineProjection,
    *,
    n: int = 3,
    inliers_tolerance=5,
    min_inliers: int = 0.5,
    max_iterations: int = 100,
    early_stop_tolerance: float = 1,
    early_stop_min_inliers: int = 0.5,
) -> Tuple[FundusProjection, float, np.ndarray]:
    """
    Estimates a 2D transformation matrix that maps points from ``src`` to ``dst`` using the RANSAC algorithm.

    Parameters
    ----------
        src: np.ndarray
            Source points coordinates (N x 2) where N is the number of points.

        dst: np.ndarray
            Destination points coordinates (N x 2).

        initial_projection: Type[FundusProjection], optional
            The type of projection to use for the initial estimation.

        final_projection: Type[FundusProjection], optional
            The type of projection to use for the final estimation.

        n: int, optional
            Number of points to sample for each iteration.

        inliers_tolerance: float, optional
            Maximum distance between points to consider them as inliers.

        min_matching_points: float, optional
            Minimum number of points that must be inliers to consider the transformation as valid.

        max_iterations: int, optional
            Maximum number of iterations.

        early_stop_tolerance: float, optional
            Mean distance under which the algorithm should stop early.

    Returns
    -------
        T: FundusProjection
            The best transformation of type ``final_projection`` found to map points from ``src`` to ``dst``.

        error: float
            Mean distance of the best transformation.

        inliers: np.ndarray
            Indices of the points that are considered inliers.

    Raises
    ------
        ValueError
            If no transformation matches the criteria
    """
    assert src.ndim == 2 and src.shape[1] == 2, "src must be a 2D array of 2D coordinates"
    assert src.shape == dst.shape, "src and dst must have the same shape"
    N = src.shape[0]

    inliers_tolerance = inliers_tolerance**2
    early_stop_tolerance = early_stop_tolerance**2

    if min_inliers < 1:
        min_inliers = int(min_inliers * (N - n))
    if early_stop_min_inliers < 1:
        early_stop_min_inliers = int(early_stop_min_inliers * (N - n))

    best_T = None
    best_mean_error = np.inf
    best_inliers = []

    for _ in range(max_iterations):
        # Sample n points
        idx = np.arange(N)
        np.random.shuffle(idx)

        # Estimate transformation
        iniT, _ = initial_projection.fit(src[idx[:n]], dst[idx[:n]])

        # Apply transformation to all other points and calculate error
        errors = iniT.quadratic_error(src[idx[n:]], dst[idx[n:]])

        # Check if transformation m is valid
        n_inliers = np.sum(errors < inliers_tolerance) + n
        if n_inliers < max(min_inliers + n, len(best_inliers)):
            continue

        # Optimize transformation using all inliers
        inliers = np.concatenate([idx[:n], idx[n:][errors < inliers_tolerance]])
        T, mean_error = final_projection.fit(src[inliers], dst[inliers])

        # Save transformation if it is better
        if n_inliers > len(best_inliers) or mean_error < best_mean_error:
            best_mean_error = mean_error
            best_T = T
            best_inliers = inliers

            # Early stop if the error is below the tolerance
            if best_mean_error < early_stop_tolerance and n_inliers >= early_stop_min_inliers + n:
                break

    if best_T is None:
        raise ValueError("RAMSAC algorithm failed: no transformation matched the criteria")

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
