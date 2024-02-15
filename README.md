# modelling-toolbox

## Install from GitHub

In order to install the package from GitHub, run the following command:

```bash
pip install git+https://github.com/DevComPsy/modelling-toolbox.git
```

## Documentation

First, install all requirements:

```bash
pip install -r docs/requirements.txt
```

In the root directory, run the following commands:

```bash
mkdocs build
mkdocs serve
```

Then open a browser and go to http://127.0.0.1:8000/

# Development process

To work on the toolbox, create a new branch from the `main` branch. Then, create a pull request to merge the new feature into the `main` branch. Once the pull request is approved, merge the new feature into the `main` branch.

## Branch naming convention

A git branch should start with a category. Pick one of these: feature, bugfix, hotfix, or test.

* `feature` is for adding, refactoring or removing a feature
* `bugfix` is for fixing a bug
* `hotfix` is for changing code with a temporary solution and/or without following the usual process (usually because of an emergency)
* `test` is for experimenting outside of an issue/ticket

See this [link](https://dev.to/couchcamote/git-branching-name-convention-cch) for some great description of the naming convention.

## Commit message conventions

Please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) guidelines for commit messages.
Feel free to use gitmoji for commit messages, but insert them at the end of the problem description.
See this [link](https://gitmoji.dev/) for more information.

## Pull request conventions
    
When creating a pull request, make sure to follow these conventions: [link](https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/)

## Compiling the package

To compile the package, run the following command in the root directory:

```bash
python setup.py sdist bdist_wheel
```

## Uploading the package to PyPi

To upload the package to PyPi, run the following command in the root directory:

```bash
twine upload dist/*
```

## Development tools we use

* Black linter for python code formatting
* Numpy-style docstrings for documentation
* mkdocs for documentation generation
* pytest for testing