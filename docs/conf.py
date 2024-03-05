# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = 'DICE Embeddings'
copyright = '2023, Caglar Demir'
author = 'Caglar Demir'
release = '0.1.3.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension",
              "sphinx.ext.githubpages",
              "sphinx.ext.todo",
              "sphinx.ext.viewcode",
              "sphinx.ext.autodoc"]

# autoapi for dicee.
autoapi_dirs = ['../dicee']

# by default all are included but had to reinitialize this to remove private members from showing
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'special-members',
                   'imported-members']

# this is set to false, so we can add it manually in index.rst together with the other .md files of the documentation.
autoapi_add_toctree_entry = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------

latex_engine = 'xelatex'
latex_show_urls = 'footnote'
latex_theme = 'howto'

latex_elements = {
    'preamble': r'''
\renewcommand{\pysiglinewithargsret}[3]{%
  \item[{%
      \parbox[t]{\linewidth}{\setlength{\hangindent}{12ex}%
        \raggedright#1\sphinxcode{(}\linebreak[0]{\renewcommand{\emph}[1]{\mbox{\textit{##1}}}#2}\sphinxcode{)}\linebreak[0]\mbox{#3}}}]}
''',
    'printindex': '\\def\\twocolumn[#1]{#1}\\footnotesize\\raggedright\\printindex',
}


def setup(app):
    # -- Options for HTML output ---------------------------------------------
    app.add_css_file('theme_tweak.css')
