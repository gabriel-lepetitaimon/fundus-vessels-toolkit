from typing import Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt

from fundus_vessels_toolkit.utils.fundus_projections import FundusProjection

from .geometric import Rect


def stitch_images(
    images: Iterable[npt.NDArray[np.float32]],
    transforms: Iterable[FundusProjection],
    masks: Optional[Iterable[npt.NDArray[np.float32]] | None] = None,
    return_warped_domain: bool = False,
) -> npt.NDArray[np.float32] | Tuple[npt.NDArray[np.float32], Rect]:
    """
    Stitch images together using the provided transformations.

    Parameters
    ----------
    images : Iterable[np.ndarray]
        The images to stitch.

    transforms : Iterable[np.ndarray]
        The transformations to apply to each image.

    masks : Optional[Iterable[np.ndarray]], optional
        The masks to apply to weight each pixel of each image in the final stitched image. By default None.

    return_warped_domain : bool, optional
        Whether to return the domain of the stitched image. By default False.

    Returns
    -------
    np.ndarray
        The stitched image.

    Rect
        The domain of the stitched image if ``return_warped_domain`` is True.
    """

    if masks is None:
        masks = [None for img in images]

    # Get the size of the stitched image
    warped_rects = []

    for img, T in zip(images, transforms, strict=True):
        src_domain = Rect.from_size(img.shape[:2])
        warped_rects.append(T.transform_domain(src_domain))

    stitched_domain = Rect.union(*warped_rects).to_int()
    warped_rects = [_ - stitched_domain.top_left for _ in warped_rects]

    # Create the stitched image
    stitched = np.zeros(tuple(stitched_domain.shape) + img.shape[2:], dtype=images[0].dtype)
    stitched_w = np.zeros(stitched_domain.shape, dtype=np.float32)

    for img, T, mask, warped_rect in zip(images, transforms, masks, warped_rects, strict=True):
        src_domain = Rect.from_size(img.shape[:2]) - stitched_domain.top_left
        # Warp the image
        img_warped, _ = T.warp(img, src_domain, warped_rect)

        # Warp the mask
        if mask is None:
            mask = np.ones(img.shape[:2], dtype=np.float32) if mask is None else mask
        else:
            mask = np.asarray(mask, dtype=np.float32)
        mask_warped, _ = T.warp(mask, src_domain, warped_rect)

        # Apply the mask to the image
        img_warped = img_warped * mask_warped[..., None]

        # Add the warped image to the stitched image
        stitched[warped_rect.slice()] += img_warped
        stitched_w[warped_rect.slice()] += 1 if mask is None else mask_warped

    # Normalize the stitched image
    non_zero = stitched_w > 0
    if stitched.ndim == 2:
        stitched[non_zero] = stitched[non_zero] / stitched_w[non_zero]
    else:
        for c in range(stitched.shape[2]):
            stitched[non_zero, c] = stitched[non_zero, c] / stitched_w[non_zero]

    return stitched if not return_warped_domain else stitched, stitched_domain


def fundus_circular_fuse_mask(fundus_or_mask: npt.NDArray[np.float32 | np.bool_]):
    fundus_or_mask = np.asarray(fundus_or_mask)
    if fundus_or_mask.dtype != bool:
        from .fundus import fundus_ROI

        mask = fundus_ROI(fundus_or_mask)
    else:
        mask = fundus_or_mask

    # Evaluate the mask radius and center
    mask_left = np.min(np.argmax(mask, axis=1))
    mask_right = mask.shape[1] - np.min(np.argmax(mask[:, ::-1], axis=1))
    mask_top = np.min(np.argmax(mask, axis=0))
    mask_bottom = mask.shape[0] - np.min(np.argmax(mask[::-1, :], axis=0))

    mask_center = np.array([(mask_top + mask_bottom) / 2, (mask_left + mask_right) / 2])
    mask_radius = max(mask_right - mask_left, mask_bottom - mask_top) / 2

    # Compute a gaussian centered on the mask center
    y, x = np.indices(mask.shape)
    std = mask_radius / 3
    weights = np.exp(-((y - mask_center[0]) ** 2 + (x - mask_center[1]) ** 2) / (2 * std**2))

    return weights * mask
