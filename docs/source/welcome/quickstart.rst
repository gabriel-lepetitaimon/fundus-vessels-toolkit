************
Quick Start
************

The `fundus-vessels-toolkit` package is a Python package that provides a set of tools to work with fundus images. It is designed to be easy to use and to provide a set of tools that can be used to build more complex applications.

This page will guide you through the installation process and basic usage of the package.

Installation
============

The `fundus-vessels-toolkit` package is available on PyPI and can be installed using pip:

.. code-block:: bash
    
    git clone https://github.com/gabriel-lepetitaimon/fundus-vessels-toolkit.git
    pip install -e fundus-vessels-toolkit


This package requires cv2, you can install it with:

.. code-block:: bash

    pip install opencv-python-headless


If you plan to use the commmon use-case example provided as Jupyter Notebooks in the `samples/` folder, you should
also install their dependencies:

.. code-block:: bash
    
    pip install -e "fundus-vessels-toolkit[samples]"

