#!/usr/bin/env python

import os
import sys
from setuptools import setup, find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# # Load requirements from txt file
# with open("requirements.txt") as f:
#     requirements = f.read().splitlines()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

readme = open("README.md").read()
doclink = """
Documentation
-------------

The full documentation is at https://devcompsy.github.io/cpm/."""
history = open("CHANGELOG.md").read().replace(".. :changelog:", "")

setup(
    name="cpm",
    version="0.23.3",
    description="cpm",
    long_description=readme + "\n\n" + doclink + "\n\n" + history,
    author="Lenard Dome, Frank Hezemans, Andrew Webb, Marc Carrera, Tobias Hauser",
    author_email="lenarddome@gmail.com",
    url="https://github.com/DevComPsy/cpm",
    packages=find_packages(),
    include_package_data=True,
    package_data={"cpm": ["**/*.csv"]},
    python_requires=">3.11.0",
    install_requires=[
        "numpy>=2.0.0",  # Numerical functions
        "SciPy>=1.11.4",  # Scientific functions
        "pandas>=2.1.4",  # Data structures & analysis
        "multiprocess>=0.70.16",  # Multiprocessing
        "ipyparallel>=8.8.0",  # IPython parallel
        "numdifftools>=0.9.41",  # Numerical differentiation
        "pybads>=1.0.4",  # Bayesian Directed Search
        "dill>=0.3.8",  # Serialization
    ],
    license="AGPLv3",
    zip_safe=False,
    keywords="cpm",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: AGPLv3",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
