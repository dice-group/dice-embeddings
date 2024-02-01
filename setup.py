import re
from setuptools import setup, find_packages

#  pip3 install "dicee" .
#  pip3 install "dicee[dev]" .
#  pip3 install "dicee[min]" .
_deps = [
    "torch>=2.0.0",
    "lightning>=2.1.3",
    "pandas>=2.1.0",
    "polars>=0.16.14",  # this can be lazy imported
    "scikit-learn>=1.2.2",  # this can be lazy imported
    "pyarrow>=11.0.0",  # this can be lazy imported
    # "pykeen==1.10.1", # Temporarly removed due to
    "zstandard>=0.21.0",  # this can be lazy imported
    "pytest>=7.2.2",  # if testing required
    "psutil>=5.9.4",
    "ruff>=0.0.284",  # if testing required
    "gradio>=3.23.0",  # if deployment required
    "rdflib>=7.0.0",
    "tiktoken>=0.5.1",
    "matplotlib>=3.8.2",
    "beautifulsoup4>=4.12.2"
]

# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = dict()
extras["dev"] = deps_list(
    "pandas", "polars", "rdflib",  # Loading KG
    "torch", "lightning",  # Training KGE
    "tiktoken",
    # "pyarrow",
    "psutil",  # Memory tracking: (do we need it
    "matplotlib")
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
    entry_points={"console_scripts":
                      ["dicee=dicee.scripts.run:main",
                       "diceeindex=dicee.scripts.index:main",
                       "diceeserve=dicee.scripts.serve:main"]},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
