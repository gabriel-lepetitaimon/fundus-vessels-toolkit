import warnings

import numpy as np
import numpy.typing as npt


def fundus_ROI(
    fundus: np.ndarray,
    blur_radius=5,
    red_threshold=30,
    morphological_clean=False,
    smoothing_radius=0,
    final_erosion=5,
    check=True,
) -> npt.NDArray[np.bool_]:
    """Compute the region of interest (ROI) of a fundus image by thresholding its red channel.

    Parameters:
    -----------
    fundus:
        The fundus image. Expect dimensions: (H, W, rgb).

    blur_radius:
        The radius of the median blur filter.

        By default: 5.

    red_threshold:
        The threshold value for the red channel.

        By default: 30.

    morphological_clean:
        Whether to perform morphological cleaning. (small objects removal and filling of the holes not on the border)

        By default: False.

    smoothing_radius:
        The radius of the Gaussian blur filter.

        By default: 0.

    final_erosion:
        The radius of the disk used for the final erosion.

        By default: 4.

    Returns:
        The ROI mask.

    """
    import skimage.measure as skmeasure
    import skimage.morphology as skmorph

    from .safe_import import import_cv2 as cv2

    padding = blur_radius + smoothing_radius
    fundus = np.pad(fundus[..., 0], ((padding, padding), (padding, padding)), mode="constant")
    if fundus.dtype != np.uint8:
        if fundus.max() <= 1:
            fundus *= 255
        fundus = fundus.astype(np.uint8)
    fundus = cv2().medianBlur(fundus, blur_radius * 2 - 1)
    mask = fundus >= red_threshold

    if morphological_clean:
        mask = skmorph.remove_small_objects(mask, 5000)

        # Remove holes that are not on the border
        MASK_BORDER = np.zeros_like(mask)
        MASK_BORDER[0, :] = 1
        MASK_BORDER[-1, :] = 1
        MASK_BORDER[:, 0] = 1
        MASK_BORDER[:, -1] = 1
        labelled_holes = skmeasure.label(mask == 0)
        for i in range(1, labelled_holes.max() + 1):
            hole_mask = labelled_holes == i
            if not np.any(MASK_BORDER & hole_mask):
                mask[hole_mask] = 1

    # Take the largest connected component
    _, labels, stats, _ = cv2().connectedComponentsWithStats(mask.astype(np.uint8), 4, cv2().CV_32S)
    if stats.shape[0] >= 1:
        mask = labels == (np.argmax(stats[1:, cv2().CC_STAT_AREA]) + 1)

    if smoothing_radius > 0:
        mask = (
            cv2.GaussianBlur(
                mask.astype(np.uint8) * 255,
                (smoothing_radius * 6 + 1, smoothing_radius * 6 + 1),
                smoothing_radius,
                borderType=cv2.BORDER_CONSTANT,
            )
            > 125
        )

    if final_erosion > 0:
        skmorph.binary_erosion(mask, skmorph.disk(final_erosion), out=mask)

    if check:
        if mask.sum() < mask.size * 0.6:
            warnings.warn(
                "The computed ROI mask is smaller than 60% of the image size and might be invalid.",
                RuntimeWarning,
                stacklevel=2,
            )

    return mask[padding:-padding, padding:-padding]
