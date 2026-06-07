from time import perf_counter


_profilers = {}


class Profiler:
    def __init__(self, name=""):
        self.name = name
        self.elapsed_time = []
        self.t0 = 0
        self.sub_profiler = {}

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time.append(perf_counter() - self.t0)

    def sub(self, name=""):
        return self.sub_profiler.setdefault(name, Profiler(name))

    def _largest_name_length(self):
        largest_name_length = len(self.name)
        for sub_profiler in self.sub_profiler.values():
            largest_name_length = max(largest_name_length, sub_profiler._largest_name_length())
        return largest_name_length

    def _max_depth(self):
        max_depth = 1
        for sub_profiler in self.sub_profiler.values():
            max_depth = max(max_depth, 1 + sub_profiler._max_depth())
        return max_depth

    def print(self, largest_name_length=None):
        if largest_name_length is None:
            largest_name_length = self._largest_name_length() + 4 * self._max_depth()
        res = f"{self.name} ".ljust(largest_name_length)
        res += f"{time2str(sum(self.elapsed_time))} (runs={len(self.elapsed_time)}"
        if len(self.elapsed_time) > 1:
            res += f", avg={time2str(sum(self.elapsed_time) / len(self.elapsed_time))}"
        res += ")"
        for i, sub_profiler in enumerate(self.sub_profiler.values()):
            sub_res = sub_profiler.print(largest_name_length - 4)
            if i == len(self.sub_profiler) - 1:
                res += "\n└── " + sub_res.replace("\n", "\n    ")
            else:
                res += "\n├── " + sub_res.replace("\n", "\n│   ")
        return res

    @classmethod
    def get(cls, name):
        return _profilers.setdefault(name, Profiler(name))

    @classmethod
    def reset(cls, name=None):
        if name is None:
            _profilers.clear()
        else:
            _profilers.pop(name, None)


def time2str(float, length=8):
    if float < 1e-3:
        return f"{float * 1e6:.1f}µs".rjust(length)
    elif float < 1:
        return f"{float * 1e3:.2f}ms".rjust(length)
    else:
        return f"{float:.3f}s".rjust(length)
