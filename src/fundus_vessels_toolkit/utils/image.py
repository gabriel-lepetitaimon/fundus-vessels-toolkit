from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import cv2.typing as cvt


def crop_pad_center(img: npt.NDArray, shape: Tuple[int, ...]) -> npt.NDArray:
    H, W = img.shape[:2]
    h, w = shape[:2]

    out = np.zeros(shape=shape + img.shape[2:], dtype=img.dtype)

    if H < h:
        x0 = (h - H) // 2
        x1 = 0
        h0 = H
    else:
        x0 = 0
        x1 = (H - h) // 2
        h0 = h

    if W < w:
        y0 = (w - W) // 2
        y1 = 0
        w0 = W
    else:
        y0 = 0
        y1 = (W - w) // 2
        w0 = w

    out[x0 : x0 + h0, y0 : y0 + w0] = img[x1 : x1 + h0, y1 : y1 + w0]
    return out


def rotate(image: cvt.MatLike, angle: float, interpolation: Optional[bool] = None) -> cvt.MatLike:
    """Rotate an image by a given angle.

    Parameters
    ----------
    img : npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The image to rotate.
    angle : float
        The angle by which to rotate the image in degrees.

    Returns
    -------
        npt.NDArray: Rotated image.
    """
    from .safe_import import import_cv2

    cv2 = import_cv2()

    if interpolation is None:
        interpolation = not (image.dtype == np.uint8 and image.max() < 10)
    interpol_mode = cv2.INTER_LINEAR if interpolation else cv2.INTER_NEAREST

    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
        image_dtype = "float"
    elif image.dtype == np.bool_:
        image = (image * 255).astype(np.uint8)
        image_dtype = "bool"
    elif image.dtype == np.uint8:
        image_dtype = "uint8"
    else:
        raise ValueError(f"Unsupported image type: {image.dtype}")

    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=interpol_mode)

    match image_dtype:
        case "float":
            image = image.astype(np.float_) / 255
        case "bool":
            image = image > 127  # type: ignore[no-redef]

    return image


def resize(image: cvt.MatLike, size: Tuple[int, int], interpolation: Optional[bool] = None) -> cvt.MatLike:
    """Resize an image to a given size.

    Parameters
    ----------
    image : npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The image to resize.
    size : Tuple[int, int]
        The size to which the image should be resized as (height, width).
    interpolation : Optional[bool], optional
        Wether to use interpolation or not:
        - if False, the image is resized using the nearest neighbor interpolation;
        - if True, the image is resized using the linear interpolation for upscaling and the area interpolation for down-sampling;
        - if None, the interpolation is automatically selected based on the image type.
        The default is None.

    Returns
    -------
    npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The resized image.
    """  # noqa: E501
    from .safe_import import import_cv2

    cv2 = import_cv2()

    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255).astype(np.uint8)
        image_dtype = "float"
    elif image.dtype == np.bool_:
        image = (image * 255).astype(np.uint8)
        image_dtype = "bool"
    elif image.dtype == np.uint8:
        image_dtype = "uint8"
    else:
        raise ValueError(f"Unsupported image type: {image.dtype}")

    if interpolation is None:
        interpolation = not (image.dtype == np.uint8 and image.max() < 10)
    interpol_mode = cv2.INTER_NEAREST
    if interpolation:
        interpol_mode = cv2.INTER_LINEAR if size[0] > image.shape[0] or size[1] > image.shape[1] else cv2.INTER_AREA

    image = cv2.resize(image, (size[1], size[0]), interpolation=interpol_mode)

    match image_dtype:
        case "float":
            image = image.astype(np.float_) / 255
        case "bool":
            image = image > 127  # type: ignore[no-redef]

    return image


def smooth_labels(image: npt.NDArray[np.uint8], std: float) -> npt.NDArray[np.uint8]:
    """Smooth the labels of an image using a Gaussian filter.

    Parameters
    ----------
    image : npt.NDArray[np.uint8]
        The image to smooth.
    std : float
        The standard deviation of the Gaussian filter.

    Returns
    -------
    npt.NDArray[np.uint8]
        The smoothed image.
    """
    from .safe_import import import_cv2

    cv2 = import_cv2()
    labels = np.unique(image)
    if len(labels) == 1:
        return image

    ksize = int(2 * std + 1)
    if ksize % 2 == 0:
        ksize += 1

    # -- Smooth labels --
    if labels[0] == 0:
        labels = labels[1:]  # Ignore background
    labels_lookup = np.zeros(256, dtype=np.uint8)
    for i, label in enumerate(labels):
        labels_lookup[i + 1] = label

    image_one_hots = {label: (image == label).astype(np.uint8) * 255 for label in labels}

    for label, one_hot in image_one_hots.items():
        image_one_hots[label] = cv2.GaussianBlur(one_hot, (ksize, ksize), std)
    all_one_hots = np.stack(list(image_one_hots.values()), axis=0)
    smoothed_image = np.argmax(all_one_hots, axis=0).astype(np.uint8) + 1
    smoothed_image = labels_lookup[smoothed_image]

    # -- Smooth background --
    mask = (image != 0).astype(np.uint8) * 255
    smoothed_mask = cv2.GaussianBlur(mask, (ksize, ksize), std)
    smoothed_image[smoothed_mask < 100] = 0

    return smoothed_image
