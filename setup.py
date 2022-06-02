#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

about = {}
with open("./emg_from_lfp/__about__.py") as f:
    exec(f.read(), about)

install_requires = ["numpy>=1.17.2", "tqdm", "scipy", "tdt", "xarray", "netcdf4"]

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    license=about["__license__"],
    install_requires=install_requires,
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
