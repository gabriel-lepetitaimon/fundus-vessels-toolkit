import os

import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

CYTHON_BUILD_DIR = os.path.join(os.path.dirname(__file__), "cython_build")

# These are optional
Options.docstrings = True
Options.annotate = False

# Modules to be compiled and include_dirs when necessary
extensions = [
    Extension(
        "fundus_vessels_toolkit.vgraph.edit_distance_cy",
        ["src/fundus_vessels_toolkit/vgraph/edit_distance_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        "fundus_vessels_toolkit.seg2graph.graph_utilities_cy",
        ["src/fundus_vessels_toolkit/seg2graph/graph_utilities_cy.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),

]


setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={"language_level": 3, "profile": False},
                          nthreads=os.cpu_count(),
                          build_dir=CYTHON_BUILD_DIR,
                          force=True,
                            annotate=True,
                          ),
)
