# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath(".."))
# root_source_folder = os.path.abspath(
#    os.path.join(os.path.dirname(__file__),
#                 os.pardir)
# )
# sys.path.insert(0, root_source_folder)


# -- Project information -----------------------------------------------------

project = "Convolutional Omics Kernel Network"
copyright = "2022, Jonas C. Ditz"
author = "Jonas C. Ditz"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # for automatic docstring inclusion
    "sphinx.ext.intersphinx",  # link to other projects
    "sphinx.ext.todo",  # support TODOs
    "sphinx.ext.ifconfig",  # include stuff based on configuration
    "sphinx.ext.viewcode",  # add source code
    "sphinx.ext.napoleon",  # to understand numpy and google docstring format
    "sphinx.ext.autosummary",  # to generate function/method/attribute summary lists
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
# autoclass_content = "both"         # Add __init__ doc (ie. params) to class summaries
# html_show_sourcelink = False       # Remove 'view source code' from top of page (for html, not python)
# autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
# add_module_names = False           # Remove namespaces from class/method signatures


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# Tell auodoc to mock some external libraries
autodoc_mock_imports = ["Bio"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
