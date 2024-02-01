#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Load requirements from txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

readme = open("README.rst").read()
doclink = """
Documentation
-------------

The full documentation is at http://cpm.rtfd.org."""
history = open("HISTORY.rst").read().replace(".. :changelog:", "")

setup(
    name="cpm",
    version="0.1.0",
    description="cpm",
    long_description=readme + "\n\n" + doclink + "\n\n" + history,
    author="Lenard Dome",
    author_email="lenarddome@gmail.com",
    url="https://github.com/DevComPsy/cpm",
    packages=[
        "cpm",
    ],
    package_dir={"cpm": "cpm"},
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "copy", "pickle"],
    license="AGPLv3",
    zip_safe=False,
    keywords="cpm",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: AGPLv3",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
