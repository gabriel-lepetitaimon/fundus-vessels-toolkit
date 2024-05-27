from pathlib import Path

import cv2


def load_img(path: str | Path, binarize=False):
    img = cv2.imread(str(path))
    img = img.astype(float) / 255
    return img.mean(axis=2) > 0.5 if binarize else img
