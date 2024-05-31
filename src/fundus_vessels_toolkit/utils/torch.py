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


def torch_apply(func, *args, device="cuda", **kwargs):
    from_numpy = None
    for i, arg in enumerate(args):
        if from_numpy is None:
            if isinstance(arg, torch.Tensor):
                from_numpy = False
            if isinstance(arg, np.ndarray):
                from_numpy = True
        if isinstance(arg, torch.Tensor):
            args[i] = arg.to(device=device)
        if isinstance(arg, np.ndarray):
            args[i] = torch.from_numpy(arg).to(device=device)
    for key, value in kwargs.items():
        if from_numpy is None:
            if isinstance(arg, torch.Tensor):
                from_numpy = False
            if isinstance(arg, np.ndarray):
                from_numpy = True
        if isinstance(value, torch.Tensor):
            kwargs[key] = value.to(device=device)
        if isinstance(value, np.ndarray):
            kwargs[key] = torch.from_numpy(value).to(device=device)

    r = func(*args**kwargs)
    if from_numpy:
        return r.cpu().numpy()
