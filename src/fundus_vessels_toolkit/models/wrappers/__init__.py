import importlib
from typing import TYPE_CHECKING

__all__ = ["automorph_segment_av", "vascx_segment_av"]  # type: ignore

# LAZY IMPORT OF SUBMODULES
_LAZY_IMPORT_MAPPING = {
    "automorph_segment_av": (".automorph", "automorph_segment_av"),
    "vascx_segment_av": (".vascx", "vascx_segment_av"),
}


def __getattr__(name):
    if name in _LAZY_IMPORT_MAPPING:
        module_path, target_name = _LAZY_IMPORT_MAPPING[name]
        module = importlib.import_module(module_path, __package__)
        value = getattr(module, target_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .automorph import automorph_segment_av
    from .vascx import vascx_segment_av
