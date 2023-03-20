
def iter_index(shape):
    import numpy as np
    stride = np.concatenate(([1], np.cumprod(shape[::-1])))[::-1]
    for i in range(int(np.prod(shape))):
        yield tuple((i % stride[j])//stride[j+1] for j in range(len(shape)))
