def normalize_vector(vector: Union[Tuple[torch.Tensor], torch.Tensor], epsilon: int = 1e-8):
    """
    Normalize a vector field to unitary norm.
    Args:
        vector:
        epsilon:

    Shape:
        vector: [2, ...]

    Returns: The vector of unitary norm,
             the norm maxtrix.

    """
    if isinstance(vector, (list, tuple)):
        vectors = []
        norms = []
        for v in vector:
            v, norm = normalize_vector(v, epsilon)
            vectors += [v]
            norms += [norm]
        return vectors, norms
    d = torch_norm2d(vector)
    return vector / (d+epsilon), d


_norm2d = [None]
def torch_norm2d(xy):
    if _norm2d[0] is None:
        import torch
        def linalg_norm(xy):
            return torch.linalg.norm(xy, dim=0)
        def legacy_norm(xy):
            return torch.norm(xy, dim=0)
        try:
            d = linalg_norm(xy)
            _norm2d[0] = linalg_norm
        except AttributeError:
            d = legacy_norm(xy)
            _norm2d[0] = legacy_norm
        return d
    else:
        return _norm2d[0](xy)