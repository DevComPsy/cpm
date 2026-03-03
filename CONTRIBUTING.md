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

4. (Optional, for docs work) Install docs dependencies:

```bash
pip install -r docs/requirements.txt
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

Documentation lives in docstrings and Markdown files under `docs/`.

For local docs preview:

```bash
mkdocs build
mkdocs serve
```

Then open `http://127.0.0.1:8000/`.

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
- Docs build if docs were changed.
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

