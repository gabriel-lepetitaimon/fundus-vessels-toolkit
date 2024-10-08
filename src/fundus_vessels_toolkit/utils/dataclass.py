class UpdateableDataclass(object):
    def update(self, d=None, /, **kwargs):
        """
        Update the dataclass with the given dictionary and keyword arguments.
        """
        if isinstance(d, dict):
            d.update(kwargs)
            kwargs = d

        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)
        return self
