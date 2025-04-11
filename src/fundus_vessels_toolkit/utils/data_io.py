from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt

NumpyDict: TypeAlias = Mapping[str, Union[npt.NDArray[Any], "NumpyDict"]] | List[npt.NDArray[Any]] | List["NumpyDict"]

SEP = "/"

if TYPE_CHECKING:
    import cv2.typing as cvt


def save_numpy_dict(data_dict: NumpyDict, file_path: str | Path, compress=False):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    def recursive_flatten_dict(d, parent_key=""):
        items = []
        if isinstance(d, list):
            d = {"[" + str(i) + "]": v for i, v in enumerate(d)}
        for k, v in d.items():
            assert SEP not in k, f"Separator '{SEP}' is not allowed in keys, but found in '{k}'"
            assert k[0] != "[" or k[-1] != "]", f"Key '{k}' cannot start with '['and end with']'."
            new_key = parent_key + SEP + k if parent_key else k
            if isinstance(v, dict):
                items.extend(recursive_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    if not compress:
        np.savez(file_path, **recursive_flatten_dict(data_dict))
    else:
        np.savez_compressed(file_path, **recursive_flatten_dict(data_dict))


def load_numpy_dict(file_path: str | Path) -> NumpyDict:
    file_path = Path(file_path)
    data = np.load(file_path, allow_pickle=True)

    data_dict = {}
    for k, v in data.items():
        keys = k.split("/")
        d = data_dict
        is_keys_list = [key[0] == "[" and key[-1] == "]" for key in keys]
        keys = [int(key[1:-1]) if is_list else key for key, is_list in zip(keys, is_keys_list, strict=True)]
        for key, is_list, next_is_list in zip(keys[:-1], is_keys_list[:-1], is_keys_list[1:], strict=True):
            if is_list:
                while len(d) <= key:
                    d.append([] if next_is_list else {})
                d = d[key]
            else:
                d = d.setdefault(key, [] if next_is_list else {})
        if is_keys_list[-1]:
            while len(d) <= keys[-1]:
                d.append(None)
        d[keys[-1]] = v

    return data_dict


def pandas_to_numpy_dict(data_frame) -> NumpyDict:
    return {col: data_frame[col].to_numpy() for col in data_frame.columns}


def load_av(file, av_inverted=False, pad=None):
    from .safe_import import import_cv2

    cv2 = import_cv2()

    av_color = cv2.imread(str(file))
    if av_color is None:
        raise ValueError(f"Could not load image from {file}")

    av = np.zeros(av_color.shape[:2], dtype=np.uint8)  # Unknown
    v = av_color.mean(axis=2) > 10
    a = av_color[:, :, 2] > av_color[:, :, 0]
    if av_inverted:
        a = ~a
    av[v & a] = 1  # Artery
    av[v & ~a] = 2  # Vein

    if pad is not None:
        av = np.pad(av, pad, mode="constant", constant_values=0)
    return av


def load_image(path: str | Path, binarize=False, resize=None, pad=None, cast_to_float=True) -> np.ndarray:
    from .safe_import import import_cv2

    cv2 = import_cv2()

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image from {path}")

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]  # BGR to RGB

    if img is None:
        raise ValueError(f"Could not load image from {path}")

    if resize is not None:
        img = resize_image(img, resize)

    if binarize:
        img = img.mean(axis=2) > 127 if img.ndim == 3 else img > 127
    elif cast_to_float:
        img = img.astype(float) / 255

    if pad is not None:
        if isinstance(pad, int):
            pad = ((pad, pad), (pad, pad))
        elif isinstance(pad, tuple) and all(isinstance(_, int) for _ in pad) and len(pad) == 2:
            pad = (pad, pad)
        img = np.pad(img, pad, mode="constant", constant_values=0)

    return img


def resize_image(image: cvt.MatLike, size: Tuple[int, int], interpolation: Optional[bool] = None) -> cvt.MatLike:
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
        # - if True, the image is resized using the linear interpolation for upscaling and the area interpolation for down-sampling;
        - if None, the interpolation is automatically selected based on the image type.
        The default is None.

    Returns
    -------
    npt.NDArray[np.float  |  np.uint8  |  np.bool_]
        The resized image.
    """  # noqa: E501
    from .safe_import import import_cv2

    cv2 = import_cv2()

    is_image_bool = image.dtype == np.bool_
    in_img = image.astype(np.uint8) * 255 if is_image_bool else image

    if interpolation is None:
        interpolation = image.dtype == np.float_ or (image.dtype == np.uint8 and image.max() > 10)
    interpol_mode = cv2.INTER_NEAREST
    if interpolation:
        interpol_mode = cv2.INTER_LINEAR if size[0] > image.shape[0] or size[1] > image.shape[1] else cv2.INTER_AREA

    image = cv2.resize(in_img, (size[1], size[0]), interpolation=interpol_mode)

    if is_image_bool:
        image = image > 127

    return image
