def import_cv2():
    try:
        import cv2
    except ImportError:
        raise ImportError(  # noqa: B904
            "cv2 is not available in the current python environment.\n"
            "\t This package is required by fundus-vessels-toolkit but "
            "we can't add it to the dependencies to prevent versions conflict.\n"
            "\t Please install it by yourself using `pip install opencv-python-headless`."
        )
    return cv2
