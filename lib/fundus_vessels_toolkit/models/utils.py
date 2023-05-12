

def img_to_torch(x, device='cuda'):
    import torch
    import numpy as np

    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f'Unknown type: {type(x)}.\n Expected numpy.ndarray or torch.Tensor.')

    match x.shape:
        case s if len(s) == 3:
            if s[2] == 3:
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        case s if len(s) == 4:
            assert s[1] == 3, f'Expected 3 channels, got {s[1]}'

    return x.float().to(device=device)


def ensure_superior_multiple(x, m=32):
    """
    Return y such that y >= x and y is a multiple of m.
    """
    return m - (x-1) % m + x - 1
