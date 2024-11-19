from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, TypeAlias, Union

import numpy as np
import numpy.typing as npt

NumpyDict: TypeAlias = Mapping[str, Union[npt.NDArray[Any], "NumpyDict"]] | List[npt.NDArray[Any]] | List["NumpyDict"]

SEP = "/"


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
        img = cv2.resize(img, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)

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
