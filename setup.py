import re
from setuptools import setup, find_packages

# Installation options:
# - Default install (CPU-only recommended): pip install dicee --extra-index-url https://download.pytorch.org/whl/cpu
# - GPU/CUDA install: pip install dicee
# - Development: pip install dicee[dev] --extra-index-url https://download.pytorch.org/whl/cpu
# - Documentation: pip install dicee[docs]
# - All extras: pip install dicee[all] --extra-index-url https://download.pytorch.org/whl/cpu
#
# NOTE: To avoid ~2GB NVIDIA CUDA dependencies, always use --extra-index-url https://download.pytorch.org/whl/cpu

# Core dependencies (includes torch and lightning)
# Use --extra-index-url https://download.pytorch.org/whl/cpu for CPU-only installation
_core_deps = [
    "numpy==1.26.4",
    "torch>=2.5.1",
    "lightning>=2.5.0.post0",
    "pandas<=2.3.3",
    "requests>=2.32.3",
    "polars>=0.16.14",
    "pytorch_lightning>=2.5.1",
    "tiktoken>=0.5.1",
]

# Optional dependencies for various features
_optional_deps = [
    "pyarrow>=11.0.0",
    "rdflib>=7.0.0",
    "tiktoken>=0.5.1",
    "pykeen>=1.10.2",
    "psutil>=5.9.4",
    "matplotlib>=3.8.2",
    "zstandard>=0.21.0",
    "requests>=2.32.3",
    "scikit-learn>=1.2.2",
]

# Development dependencies
_dev_deps = [
    "pytest>=7.2.2",
    "ruff>=0.0.284",
    "scikit-learn>=1.2.2",
]

# Documentation dependencies
_docs_deps = [
    "sphinx>=7.2.6",
    "sphinx-autoapi>=3.0.0",
    "myst-parser>=2.0.0",
    "sphinx_rtd_theme>=2.0.0",
    "sphinx-theme>=1.0",
    "sphinxcontrib-plantuml>=0.27",
    "plantuml-local-client>=1.2022.6",
]

# Combine all dependencies for regex parsing
_all_deps = _core_deps + _optional_deps + _dev_deps + _docs_deps

# Parse dependencies into a dictionary
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _all_deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


# Define extras
extras = dict()

# Minimal installation - only core dependencies
extras["min"] = _core_deps

# Development extras
extras["dev"] = _core_deps + _optional_deps + _dev_deps

# Documentation extras
extras["docs"] = _docs_deps

# All extras
extras["all"] = _core_deps + _optional_deps + _dev_deps + _docs_deps

# Base installation includes core dependencies (torch and lightning included)
# Use --extra-index-url https://download.pytorch.org/whl/cpu to avoid NVIDIA CUDA dependencies
install_requires = _core_deps

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name="dicee",
    description="Dice embedding is an hardware-agnostic framework for large-scale knowledge graph embedding applications",
    version="0.3.1",
    packages=find_packages(exclude=["tests", "tests.*"]),
    extras_require=extras,
    install_requires=list(install_requires),
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/dice-group/dice-embeddings',
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License"],
    python_requires='>=3.11',
    entry_points={"console_scripts":
                      ["dicee=dicee.scripts.run:main",
                       "dicee_vector_db=dicee.scripts.index_serve:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
