import numpy as np
import torch


def img_to_torch(x, device="cuda"):
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f"Unknown type: {type(x)}.\n Expected numpy.ndarray or torch.Tensor.")

    match x.shape:
        case s if len(s) == 3:
            if s[2] == 3:
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        case s if len(s) == 4:
            assert s[1] == 3, f"Expected 3 channels, got {s[1]}"

    return x.float().to(device=device)


def recursive_numpy2torch(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        return x.to(device) if device is not None else x
    if isinstance(x, dict):
        return {k: recursive_numpy2torch(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_numpy2torch(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple([recursive_numpy2torch(v, device) for v in x])
    return x


def recursive_torch2numpy(x):
    if isinstance(x, torch.Tensor):
        r = x.cpu().numpy()
        return r
    if isinstance(x, dict):
        return {k: recursive_torch2numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_torch2numpy(v) for v in x]
    if isinstance(x, tuple):
        return tuple(recursive_torch2numpy(v) for v in x)
    return x


def torch_apply(func, *args, device=None, **kwargs):
    from_numpy = None
    new_args = []
    for arg in args:
        if from_numpy is None:
            if isinstance(arg, torch.Tensor):
                from_numpy = False
            if isinstance(arg, np.ndarray):
                from_numpy = True
        new_args.append(recursive_numpy2torch(arg, device))
    for key, value in kwargs.items():
        if from_numpy is None:
            if isinstance(value, torch.Tensor):
                from_numpy = False
            if isinstance(value, np.ndarray):
                from_numpy = True
        kwargs[key] = recursive_numpy2torch(value, device)

    r = func(*new_args, **kwargs)

    return recursive_torch2numpy(r) if from_numpy else r


def autocast_torch(f):
    def wrapper(*args, **kwargs):
        return torch_apply(f, *args, **kwargs)

    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__

    return wrapper
