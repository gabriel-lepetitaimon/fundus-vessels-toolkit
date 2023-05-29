import numpy as np

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
    
    import cv2
    from skimage import measure
    import math
    
    if median_blur_size is None:
        median_blur_size = int(math.ceil(img.shape[1]/50))
        median_blur_size += median_blur_size % 2 == 0
    
    if img.dtype != np.uint8:
        if img.max() <= 1:
            img *= 255
        img = img.astype(np.uint8)
    img = cv2.medianBlur(img[:, :, 2], median_blur_size) > threshold
    
    components = measure.label(img, connectivity=1)
    components_size = np.bincount(components.flatten())
    max_size_component_id = np.argmax(components_size[1:])
    return components == max_size_component_id+1