# Contributing to `cpm`

Thank you for contributing to **cpm**.
This guide explains how to set up your environment, make changes, and submit a pull request.

## Prerequisites

- Python `>=3.12`
- `pip`
- `git`

## Local setup

1. Fork the repository and clone your fork.
2. Create and activate a virtual environment.
3. Install the package in editable mode:

```bash
pip install -e .
```

4. (Optional, for docs work) Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Development workflow (with direct access to repository)

1. Create a branch from `main`.
2. Make focused changes with clear commit history. See commit message conventions below.
3. Run tests locally before opening a pull request.
4. Update documentation and changelog when relevant.
5. Open a [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) on GitHub.

## Development workflow (without direct access)

1. [Fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) the repository.
1. (optional) Create a branch from `main`.
2. Make focused changes with clear commit history. See commit message conventions below.
3. Run tests locally before opening a pull request.
4. Update documentation and changelog when relevant.
5. Open a [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) on GitHub.

### Branch naming convention

Start your branch name with one of:

- `feature/` for adding, refactoring, or removing a feature
- `bugfix/` for fixing a bug
- `hotfix/` for urgent, temporary, or process-exception changes
- `test/` for experiments and test-related work
- `docs/` for adding documentation or examples.

Example:

```text
feature/add-hierarchical-export
```

## Code style and quality

- Format Python code with **Black**.
- Keep code readable and focused; avoid unrelated refactors in the same PR.
- Use **NumPy-style docstrings** where applicable.
- Keep public API changes backward-compatible where possible.

## Testing

Run the test suite from the repository root:

```bash
pytest
```

If you changed specific functionality, also run targeted tests first (for example, under `test/models/` or `test/optimisation/`).

## Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and the [PyData theme](https://pydata-sphinx-theme.readthedocs.io/).
It has two sources: the docstrings in `cpm/`, which make up the API reference, and the pages under `docs/`.

### Building the documentation

```bash
pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
While you write, `sphinx-autobuild` rebuilds the pages and reloads the browser on every change:

```bash
sphinx-autobuild docs docs/_build/html --watch cpm --ignore "*/api/generated/*" --ignore "*/_static/thumbnails/*"
```

The build must finish without warnings: CI runs it with `-W`.

### Where things live

| Folder | Contents |
| --- | --- |
| `docs/get-started/` | installation, quickstart, core concepts, troubleshooting |
| `docs/tutorials/` | the numbered learning path, as Jupyter notebooks |
| `docs/examples/` | research examples, as Jupyter notebooks, grouped by domain in `examples/index.md` |
| `docs/how-to/` | short guides for specific tasks |
| `docs/api/` | the API reference pages; add new public objects to the `autosummary` list of their page |
| `docs/about/` | citing, changelog, roadmap, contributing, license |
| `docs/scripts/` | checks run in CI, and tools for notebooks |

### Docstrings

Docstrings use the [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) and reStructuredText, not Markdown:

- math: `` :math:`lpha` `` inline, and a `.. math::` block for equations. Docstrings with a backslash must be raw strings (`r"""..."""`).
- links to cpm objects: `` :class:`~cpm.generators.Wrapper` ``, `` :func:`~cpm.datasets.load_bandit_data` ``; in a "See Also" section, just the name: `cpm.generators.Wrapper : description`.
- other links: `` `text <https://example.org>`__ ``.
- `` `single backticks` `` show code.

`python docs/scripts/md2rst_docstrings.py --check cpm` finds Markdown left in docstrings, and `python docs/scripts/check_api_coverage.py` finds public objects missing from the API reference.
To link an API page to the tutorials that use the object, add it to `API_TUTORIAL_LINKS` in `docs/conf.py`.

### Notebooks

Tutorials and examples are Jupyter notebooks, rendered with their stored outputs; they are not run when the documentation is built.
Each notebook follows the same structure:

1. a short `#` title, followed by a line with the level, the time it takes, the prerequisites, the cpm components used, and the data,
2. a "What you'll learn" list,
3. one setup cell that imports cpm, prints `cpm.__version__`, and sets a random seed,
4. the content, with headings that do not skip levels,
5. a `## Summary and next steps` section, and a `## References` section.

Link to the API with `` [`Wrapper`](#cpm.generators.Wrapper) ``, and to other pages with relative links to their files, such as `[Tutorial 1](../tutorials/first-model.ipynb)`, so that the links also work when the notebook is opened in Jupyter.
In examples, tag the cell whose figure should be the gallery thumbnail with `thumbnail`.
Notebooks must not write files into `docs/`.

Before committing a notebook, run it from top to bottom in a fresh kernel, then clean and check it:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/tutorials/first-model.ipynb
python docs/scripts/clean_notebooks.py docs/tutorials/first-model.ipynb
python docs/scripts/check_notebooks.py
```

A new example also needs a card in `docs/examples/index.md`, and a new tutorial one in `docs/tutorials/index.md`.

## Commit messages

Please follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary).

Examples:

- `feat(models): add drift-bias parameter`
- `fix(optimisation): handle empty bounds`
- `docs(api): clarify hierarchical fitting inputs`

## Pull requests

When opening a PR:

- Keep the PR scope small and focused.
- Include a clear description of **what** changed and **why**.
- Link related issues.
- Add/update tests for behavior changes.
- Update docs when user-facing behavior changes.

Before requesting review, verify:

- Tests pass locally.
- The documentation builds without warnings if you changed docstrings or pages.
- No unrelated files are included.

## Troubleshooting and getting help

If you run into issues at any point, please reach out.
We are happy to help with anything.

For faster support, please include:

- What you were trying to do
- The exact error message or traceback
- Your Python version and operating system
- Steps to reproduce the issue

Best ways to get help:

- Open a GitHub issue for bugs, setup problems, or unclear behavior
- Open a draft PR if you want feedback while you are still working
- Ask your question directly in the PR conversation if it is related to your change

## Build and release notes

If you are preparing a release artifact:

```bash
python setup.py sdist bdist_wheel
```

Publishing to PyPI is maintainer-only and typically done with:

```bash
twine upload dist/*
```
