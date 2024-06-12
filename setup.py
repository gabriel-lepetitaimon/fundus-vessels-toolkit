import os
from pathlib import Path

import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup
from torch.__config__ import parallel_info
from torch.utils import cpp_extension

# These are optional
Options.docstrings = True
Options.annotate = False

DEBUG = False
DEBUG_LEVEL = 0


# Modules to be compiled and include_dirs when necessary
def get_cython_extensions():
    include_dirs = [numpy.get_include()]
    define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    extensions = [
        Extension(
            "fundus_vessels_toolkit.vgraph.matching.edit_distance_cy",
            ["src/fundus_vessels_toolkit/vgraph/matching/edit_distance_cy.pyx"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
        Extension(
            "fundus_vessels_toolkit.utils.graph.graph_cy",
            ["src/fundus_vessels_toolkit/utils/graph/graph_cy.pyx"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
    ]

    return cythonize(
        extensions,
        compiler_directives={"language_level": 3, "profile": False},
        nthreads=os.cpu_count(),
        force=True,
    )


def get_torch_extensions():
    if DEBUG:
        extra_compile_args = {"cxx": ["-g3", "-O0", "-DDEBUG=%s" % DEBUG_LEVEL, "-UNDEBUG"]}
    else:
        extra_compile_args = {"cxx": ["-O3", "-fdiagnostics-color=always"]}
    extra_link_args = ["-s"]
    info = parallel_info()

    if "backend: OpenMP" in info and "OpenMP not found" not in info:
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        extra_compile_args["cxx"] += ["-Wno-dangling-reference"]  # Remove dangling reference warning from torch
        extra_compile_args["cxx"] += ["-fopenmp"]
        extra_link_args += ["-lgomp"]
    else:
        print("Compiling without OpenMP...")

    CPP_FOLDER = Path("src/fundus_vessels_toolkit/utils/cpp_extensions")
    COMMON_CPP = ["src/fundus_vessels_toolkit/utils/cpp_extensions/common.cpp"]

    return [
        cpp_extension.CppExtension(
            f"fundus_vessels_toolkit.utils.cpp_extensions.{src.stem}_cpp",
            sources=[str(src)] + COMMON_CPP,
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args,
        )
        for src in CPP_FOLDER.glob("*.cpp")
        if src.stem != "common"
    ] + [
        cpp_extension.CppExtension(
            f"fundus_vessels_toolkit.utils.cpp_extensions.{src_folder.stem}_cpp",
            sources=[
                str(src.absolute()) for src in sum([list(src_folder.rglob(f"*.{ext}")) for ext in ["cpp", "c"]], [])
            ]
            + COMMON_CPP,
            extra_compile_args=extra_compile_args["cxx"],
            extra_link_args=extra_link_args,
        )
        for src_folder in CPP_FOLDER.iterdir()
        if src_folder.is_dir()
    ]


setup(
    ext_modules=get_cython_extensions() + get_torch_extensions(),
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
