# cpm

![the toolbox logo](https://raw.githubusercontent.com/DevComPsy/cpm/main/docs/_static/logo/cpm-logo-light.png)

![PyPI - Version](https://img.shields.io/pypi/v/cpm-toolbox)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/cpm-toolbox?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/cpm-toolbox)


cpm is a fundamental package for Computational Psychiatry. It is designed to provide a set of tools for researchers and clinicians to analyze and model data in the field of Computational Psychiatry.

## Installation and Usage

To install the package, run the following command:

```bash
pip install cpm-toolbox
```

Once the package is installed, you can import it in your Python code:

```python
import cpm
```

## Install from GitHub for Nightly Builds

In order to install the package from GitHub, run the following command:

```bash
pip install git+https://github.com/DevComPsy/cpm.git
```

## Documentation

The documentation can be viewed here: [link](https://devcompsy.github.io/cpm/).

## Citing _cpm_

If you are using _cpm_, we recommend citing the titular software paper [(Dome et al., 2026)](https://doi.org/10.1371/journal.pcbi.1014481):

```
@article{dome2026cpm,
  title = {cpm: {{A}} Python Library for Theory-Driven Modelling in Computational Psychiatry},
  author = {Dome, Lenard and Hezemans, Frank H. and Kadri, Kenza and Wagner, Ben J. and Webb, Andrew and Hauser, Tobias U.},
  year = 2026,
  journal = {PLOS Computational Biology},
  volume = {22},
  number = {7},
  pages = {1--31},
  publisher = {Public Library of Science},
  doi = {10.1371/journal.pcbi.1014481}
}
```

### Development of documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) from the docstrings in `cpm/` and the pages in `docs/`.
To build it locally, run the following commands in the root directory:

```bash
pip install -e ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html` in a browser.
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to write documentation and tutorials, and for a live preview while you write.

The documentation is published to GitHub Pages by the `docs` GitHub Actions workflow whenever `main` changes.
