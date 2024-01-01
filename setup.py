import re
from setuptools import setup, find_packages

#  pip3 install "dicee" .
#  pip3 install "dicee[dev]" .
#  pip3 install "dicee[min]" .
_deps = [
    "torch>=2.0.0",
    "pandas>=2.1.0",
    "polars>=0.16.14",
    "scikit-learn>=1.2.2",
    "pyarrow>=11.0.0",
    "pytorch-lightning==1.6.4",
    "pykeen==1.10.1",
    "zstandard>=0.21.0",
    "pytest>=7.2.2",
    "psutil>=5.9.4",
    "ruff>=0.0.284",
    "gradio>=3.23.0",
    "rdflib>=7.0.0",
    "tiktoken>=0.5.1",
    "beautifulsoup4>=4.12.2",
]

# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]

extras = dict()
extras["dev"] = deps_list("torch", "pytorch-lightning", "pykeen", "tiktoken", "pandas", "polars", "psutil", "rdflib", "ruff", "pytest")
extras["min"] = deps_list("torch", "pandas")
extras["test"] = (extras["dev"] + deps_list("ruff", "pytest"))

install_requires = [
    extras["dev"],  # filesystem locks, e.g., to prevent parallel downloads
    # deps["pandas"],
    # deps["polars"],  # can be optional
    # deps["rdflib"],  # can be optional
    # deps["tiktoken"],  # can be optional
    # deps["gradio"],  # must be optinal
    # deps["beautifulsoup4"],  # Not quire sure where we use it
    # deps["scikit-learn"],  # # can be optional
    # deps["pyarrow"],  # not quire sure whether we are still using it
    # deps["pykeen"],  # can be optional
    # deps["zstandard"],  # not quire sure whether we are still using it
    # deps["pytest"],  # can be optinal
    # deps["ruff"],  # should be only in testing mode
]

with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name="dicee",
    description="Dice embedding is an hardware-agnostic framework for large-scale knowledge graph embedding applications",
    version="0.1.3",
    packages=find_packages(),
    extras_require=extras,
    install_requires=list(install_requires),
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    url='https://github.com/dice-group/dice-embeddings',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"],
    python_requires='>=3.9.18',
    entry_points={"console_scripts": ["dicee = dicee.run:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
