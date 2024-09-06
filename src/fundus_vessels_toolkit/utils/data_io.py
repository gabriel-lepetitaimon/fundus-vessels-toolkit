from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypeAlias, Union

import numpy as np
import numpy.typing as npt

NumpyDict: TypeAlias = Dict[str, Union[npt.NDArray, "NumpyDict"]] | List[Union[npt.NDArray, "NumpyDict"]]

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
