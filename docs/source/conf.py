# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fundus-vessels-toolkit"
copyright = "2024, Gabriel LEPETIT-AIMON"
author = "Gabriel LEPETIT-AIMON"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "sphinx_copybutton",
]


templates_path = ["_templates"]
exclude_patterns = []

# -- AutoDoc configuration ---------------------------------------------------
autodoc_typehints = "description"
autodoc_class_signature = "separated"
add_module_names = False


# -- Options for Napoleon ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-napoleon
napoleon_numpy_docstring = True
# napoleon_preprocess_types = True
# napoleon_use_rtype = False

# -- intersphinx configuration ------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/icon.svg"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}

# -- Sphinx Prolog -----------------------------------------------------------
rst_prolog = """
.. # define a hard line break for HTML
.. |br| raw:: html

   <br />

.. # GLOBAL SUBSTITUTIONS
.. |MAPLES-DR| replace:: :abbr:`MAPLES-DR (MESSIDOR Anatomical and Pathological Labels for Explainable Screening of Diabetic Retinopathy)`
.. |DR| replace:: :abbr:`DR (Diabetic Retinopathy)`
.. |ME| replace:: :abbr:`ME (Macular Edema)`
.. |CWS| replace:: :abbr:`CWS (Cotton Wool Spots)`
"""

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_prolog = """
.. raw:: html

    <style>
        .document p,
        .document h2,
        .document h3,
        .document h4,
        .document h5,
        .document h6 {
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .document .rendered_html pre{
            font-size: 12px;
        }
    </style>
"""

# -- Options for sphinxcontrib-bibtex -----------------------------------------
bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "unsrt"
