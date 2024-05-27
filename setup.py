import os

import numpy
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# These are optional
Options.docstrings = True
Options.annotate = False


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


setup(ext_modules=get_cython_extensions())
