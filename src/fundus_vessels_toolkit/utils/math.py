def ensure_superior_multiple(x, m=32):
    """
    Return y such that y >= x and y is a multiple of m.
    """
    return m - (x - 1) % m + x - 1
