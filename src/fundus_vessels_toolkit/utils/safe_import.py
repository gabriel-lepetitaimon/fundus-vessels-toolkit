_cv2 = None


def import_cv2():
    global _cv2
    if _cv2 is None:
        try:
            import cv2
        except ImportError:
            raise ImportError(  # noqa: B904
                "cv2 is not available in the current python environment.\n"
                "\t This package is required by fundus-vessels-toolkit but "
                "we can't add it to the dependencies to prevent versions conflict.\n"
                "\t Please install it by yourself using `pip install opencv-python-headless`."
            )
        _cv2 = cv2
    return _cv2


def is_torch_tensor(x) -> bool:
    """Check if the input is a torch.Tensor.

    If the type of the input not named "Tensor", it will return False without importing torch.
    """
    if type(x).__qualname__ == "Tensor":
        import torch

        return isinstance(x, torch.Tensor)
    return False
