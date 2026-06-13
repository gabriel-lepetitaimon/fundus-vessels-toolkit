from __future__ import annotations

import sys
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Optional

from rich.console import Console


class ProfilerWatch:
    def __init__(self, name, profiler, stack_level=0):
        t0 = perf_counter()
        stack = sys._getframe(stack_level + 1)
        self._filename = stack.f_code.co_filename + ":" + str(stack.f_lineno)
        self.name = name
        self.profiler = profiler
        self.t0 = 0
        self.sub_watches = {}
        self._profiler_token: Optional[Token[ProfilerWatch | None]] = None
        self.dt = perf_counter() - t0
        self.total_time = 0
        self.runs = 0

    def __enter__(self):
        self._profiler_token = self.profiler._current_watch.set(self)
        self.t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.total_time += perf_counter() - self.t0
        if self._profiler_token is not None:
            self.profiler._current_watch.reset(self._profiler_token)
        self.runs += 1

    def sub(self, name="", stack_level=0):
        if name in self.sub_watches:
            return self.sub_watches[name]
        profiler = ProfilerWatch(name, self.profiler, stack_level=stack_level + 1)
        self.sub_watches[name] = profiler
        return profiler

    def _max_depth(self):
        max_depth = 0
        for sub_profiler in self.sub_watches.values():
            max_depth = max(max_depth, 1 + sub_profiler._max_depth())
        return max_depth

    def _total_dt(self):
        total_dt = self.dt
        for sub_profiler in self.sub_watches.values():
            total_dt += sub_profiler._total_dt()
        return total_dt

    def _name_length(self, depth=0):
        name_length = len(self.name) + 4 * depth
        for sub_profiler in self.sub_watches.values():
            name_length = max(name_length, sub_profiler._name_length(depth + 1))
        return name_length

    def pretty_print(self, name_length=None, parent_t: list[float] | None = None, depth: int | None = None):
        if depth is None:
            depth = self._max_depth()
            print(depth)
        t = self.total_time - self._total_dt()
        if name_length is None:
            name_length = self._name_length()

        def bold(s):
            return f"[bold]{s}[/bold]"

        def grey(s):
            return f"[bright_black]{s}[/bright_black]"

        res = f"[link=file://{self._filename}]" + self.name + "[/link]"
        if self._max_depth() > 1:
            res = bold(res)
        res += " " * (max(0, name_length - len(self.name))) + " "

        if parent_t is None:
            parent_t = []
        times = ""
        for T in parent_t[:-1]:
            times += grey(format_percentage(t / T, 7)) + "  "
        if len(parent_t) > 0:
            times += grey("  ↳" + format_percentage(t / parent_t[-1], 4)) + "  "
        times += bold(time2str(self.total_time, length=7)) + "  "
        times += " " * (9 * depth)

        res += f"{times}(runs={self.runs}"
        if self.runs > 1:
            res += f", avg={time2str(self.total_time / self.runs)}"
        res += ")"
        for i, sub_watch in enumerate(self.sub_watches.values()):
            sub_res = sub_watch.pretty_print(name_length - 4, parent_t=parent_t + [t], depth=depth - 1)
            if i == len(self.sub_watches) - 1:
                res += "\n└── " + sub_res.replace("\n", "\n    ")
            else:
                res += "\n├── " + sub_res.replace("\n", "\n│   ")
        return res


class DummyWatch(ProfilerWatch):
    def __init__(self): ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def sub(self, name="", stack_level=0):
        return self

    def __getattribute__(self, item):
        if item in ("__class__", "sub", "__enter__", "__exit__"):
            return super().__getattribute__(item)
        raise AttributeError(f"DummyWatch has no attribute '{item}'")


def time2str(float, length=8):
    if float < 1e-3:
        return f"{float * 1e6:.1f}µs".rjust(length)
    elif float < 1:
        return f"{float * 1e3:.1f}ms".rjust(length)
    else:
        return f"{float:.1f}s".rjust(length)


def format_percentage(value, length=4, max_precision=2):
    value *= 100
    if value < 0.1:
        return f"{value * 10:.{min(length - 3, max_precision)}f}‰".rjust(length)
    if value >= 10:
        if length >= 5:
            return f"{value:.0f}%".rjust(length)
        else:
            return f"{value:.{min(length - 4, max_precision)}f}%".rjust(length)
    return f"{value:.{min(length - 3, max_precision)}f}%".rjust(length)


def watch(name, sub: bool = True) -> ProfilerWatch:
    profiler = _current_profiler.get()
    if profiler is None:
        return DUMMY_WATCH
    return profiler.get(name, sub=sub, stack_level=1)


class Profiler:
    def __init__(self, reset=True, print=True):
        self.print = print
        self.reset = reset
        self._profiler_token = None
        self.watches: dict[str, ProfilerWatch] = {}
        self._current_watch: ContextVar[ProfilerWatch | None] = ContextVar("current_watch", default=None)

    def __enter__(self):
        if self.reset:
            self.reset_watches()
        self._profiler_token = _current_profiler.set(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._profiler_token is not None:
            _current_profiler.reset(self._profiler_token)
        if self.print:
            self.print_all()

    def reset_watches(self, name=None):
        if name is None:
            self.watches.clear()
        elif isinstance(name, str):
            self.watches.pop(name, None)
        else:
            for n in name:
                self.watches.pop(n, None)

    def get(self, name, sub: bool = True, stack_level=0) -> ProfilerWatch:
        if sub and (current_watch := self._current_watch.get()) is not None:
            return current_watch.sub(name, stack_level=stack_level + 1)
        if name in self.watches:
            return self.watches[name]
        watch = ProfilerWatch(name, self, stack_level=stack_level + 1)
        self.watches[name] = watch
        return watch

    def current_watch(self) -> Optional[ProfilerWatch]:
        return self._current_watch.get()

    def print_all(self, width=120):
        console = Console(highlight=False, width=width)
        for watch in self.watches.values():
            console.print(watch.pretty_print())


_current_profiler: ContextVar[Profiler | None] = ContextVar("current_profiler", default=None)

DUMMY_WATCH = DummyWatch()
