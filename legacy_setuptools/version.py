# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 2  # use '' for first of series, number for 1 and above
_version_extra = "dev"
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "ecephys: Python tools for extracellular electrophysiology"
# Long description will go up on the pypi page
long_description = """
ecephys: Python tools for extracellular electrophysiology
"""

NAME = "ecephys"
MAINTAINER = "Graham Findlay"
MAINTAINER_EMAIL = "gfindlay@wisc.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/CSC-UW/ecephys"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Graham Findlay"
AUTHOR_EMAIL = "gfindlay@wisc.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
EXTRAS_REQUIRE = {
    "dev": ["altair"],
}
INSTALL_REQUIRES = [
    "black",
    "scipy",
    "numpy>=1.18",
    "pandas",
    "xarray",
    "scikit-learn",
    "statsmodels",
    "mat73",
    "matplotlib==3.5.1",
    "pyyaml",
    "seaborn",
    "colorcet",
    "ipykernel>=6.0.0",
    "ipywidgets",
    "ipympl",
    "tdt",
    "bg-atlasapi",
    "ripple_detection",
    "spikeinterface",
    "tqdm",
    "netcdf4",
    "docopt",
    "kcsd @ git+https://github.com/Neuroinflab/kCSD-python@master#egg=kcsd",
]

PYTHON_REQUIRES = ">=3.8, <3.11"