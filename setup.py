from pathlib import Path

from setuptools import setup
from torch.__config__ import parallel_info
from torch.utils import cpp_extension

DEBUG = True

WORKSPACE_FOLDER = Path(__file__).parent


def get_torch_extensions():
    if DEBUG:
        extra_compile_args = {"cxx": ["-g3", "-ggdb", "-O0"]}
        extra_link_args = ["-g"]
    else:
        extra_compile_args = {"cxx": ["-O3"]}
        extra_link_args = ["-s"]

    info = parallel_info()
    if not DEBUG and "backend: OpenMP" in info and "OpenMP not found" not in info:
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        extra_compile_args["cxx"] += ["-fopenmp"]
        extra_link_args += ["-lgomp"]
    else:
        print("Compiling without OpenMP...")

    extra_compile_args["cxx"] += ["-fdiagnostics-color=always"]  # Colorize the output
    extra_compile_args["cxx"] += ["-Wno-dangling-reference"]  # Remove dangling reference warning from torch
    extra_link_args += ["-lsupc++"]  # Fix import error: "undefined symbol: __cxa_call_terminate"

    CPP_FOLDER = Path("src/fundus_vessels_toolkit/utils/cpp_extensions")

    return [
        cpp_extension.CppExtension(
            f"fundus_vessels_toolkit.utils.cpp_extensions.{src.stem}_cpp",
            sources=[str(src.resolve().relative_to(WORKSPACE_FOLDER))],
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args,
        )
        for src in CPP_FOLDER.glob("*.cpp")
        if src.stem != "common"
    ] + [
        cpp_extension.CppExtension(
            f"fundus_vessels_toolkit.utils.cpp_extensions.{src_folder.stem}_cpp",
            sources=[
                str(src.resolve().relative_to(WORKSPACE_FOLDER))
                for src in sum([list(src_folder.rglob(f"*.{ext}")) for ext in ["cpp", "c"]], [])
            ],
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args,
        )
        for src_folder in CPP_FOLDER.iterdir()
        if src_folder.is_dir()
    ]


setup(
    ext_modules=get_torch_extensions(),
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
