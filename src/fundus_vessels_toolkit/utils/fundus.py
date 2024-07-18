import cv2
import numpy as np
import skimage.measure as skmeasure
import skimage.morphology as skmorph


def compute_ROI_mask(img: np.ndarray, median_blur_size: int | None = None, threshold: int = 5):
    """
    Compute the fundus region of interest by performing a threshold on blurred red channel.
    Only the largest connected component is returned.

    Args:
        img: The raw fundus image (numpy array: shape=(height, width, rgb_channels), dtype=np.uint8).
        median_blur_size: Size of the kernel used to perform the median blur. By default width/50.
        threshold: Threshold used on the red channel to differenciate the background from the ROI. By default: 5.
    Return:
        The mask of the Region of Interest.
        (numpy array: shape=(height, width), dtype=np.uint8)
    """
    import math

    import cv2
    from skimage import measure

    if img.ndim == 4:
        return np.stack([compute_ROI_mask(i) for i in img])
    elif img.ndim != 3:
        raise ValueError("Invalid image shape. Expected 3 or 4 dimensions.")

    if median_blur_size is None:
        median_blur_size = int(math.ceil(img.shape[1] / 50))
        median_blur_size += median_blur_size % 2 == 0

    if img.dtype != np.uint8:
        if img.max() <= 1:
            img *= 255
        img = img.astype(np.uint8)
    img = cv2.medianBlur(img[:, :, 2], median_blur_size) > threshold

    components = measure.label(img, connectivity=1)
    components_size = np.bincount(components.flatten())
    max_size_component_id = np.argmax(components_size[1:])
    return components == max_size_component_id + 1


def fundus_ROI(
    fundus: np.ndarray, blur_radius=5, morphological_clean=False, smoothing_radius=0, final_erosion=4
) -> np.ndarray:
    """Compute the region of interest (ROI) of a fundus image.

    Parameters:
    -----------
    fundus:
        The fundus image.

    blur_radius:
        The radius of the median blur filter.

        By default: 5.

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
    padding = blur_radius + smoothing_radius
    fundus = np.pad(fundus[..., 1], ((padding, padding), (padding, padding)), mode="constant")
    if fundus.dtype != np.uint8:
        fundus = (fundus * 255).astype(np.uint8)
    fundus = cv2.medianBlur(fundus, blur_radius * 2 - 1)
    mask = fundus > 10

    if morphological_clean:
        # Remove small objects
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

    return mask[padding:-padding, padding:-padding]
