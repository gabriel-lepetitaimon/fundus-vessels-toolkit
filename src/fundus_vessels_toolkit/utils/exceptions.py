import warnings
from typing import Literal, Self


class GeometryParserWarning(Warning):
    pass


class CheckReport:
    def __init__(self, on_error: Literal["raise", "warn", "report"] = "report", stacklevel: int = 2):
        self._errors: dict[str, list[str]] = {}
        self._on_error = on_error
        self.stack_level = stacklevel

    def __str__(self) -> str:
        msg = ""
        for section, errors in self._errors.items():
            msg += f"--- {section} ---\n" + "\n - ".join(errors) + "\n"
        return msg

    def log_error(self, section: str, error: str, detail: str | None = None, stack_level: int | None = None):
        error = error if detail is None else f"{error}\n\t{detail.replace('\n', '\n\t')}"
        if self._on_error == "raise":
            raise ValueError(f"{section}: {error}")
        elif self._on_error == "warn":
            if stack_level is None:
                stack_level = self.stack_level
            warnings.warn(error, stacklevel=stack_level + 1)
        self._errors.setdefault(section, []).append(error)

    def __bool__(self) -> bool:
        return bool(self._errors)

    def extend(self, other: Self):
        for section, errors in other._errors.items():
            self._errors.setdefault(section, []).extend(errors)
