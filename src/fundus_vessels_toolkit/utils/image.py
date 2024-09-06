from pathlib import Path

import cv2


def load_img(path: str | Path, binarize=False, resize=None):
    img = cv2.imread(str(path))
    img = img.astype(float) / 255
    if resize is not None:
        img = cv2.resize(img, resize)
    if binarize:
        img = img.mean(axis=2) > 0.5
    return img
