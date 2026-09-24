"""Check that the documentation notebooks follow the notebook conventions.

See the "Writing documentation" section of CONTRIBUTING.md. Exits with an error
if any notebook breaks a rule.

Usage::

    python docs/scripts/check_notebooks.py
"""

import re
import sys
from pathlib import Path

import nbformat

DOCS = Path(__file__).resolve().parents[1]
FOLDERS = ["get-started", "tutorials", "examples"]
MAX_NOTEBOOK_BYTES = 2_500_000
MAX_OUTPUT_BYTES = 1_000_000
BANNED_MIME = {
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.microsoft.datawrangler.viewer.v0+json",
}
BANNED_CODE = [
    (re.compile(r"__version__\s*[<>]"), "compares version strings"),
    (re.compile(r"\.to_csv\("), "writes files into the docs tree"),
    (re.compile(r"prettyformatter|from packaging"), "uses an undeclared dependency"),
]


def check(path):
    problems = []
    nb = nbformat.read(path, as_version=4)
    if nb.metadata.get("kernelspec", {}).get("name") != "python3":
        problems.append("kernelspec is not python3")
    if path.stat().st_size > MAX_NOTEBOOK_BYTES:
        problems.append(f"notebook is larger than {MAX_NOTEBOOK_BYTES // 1_000_000} MB")

    first = next((c for c in nb.cells if c.cell_type == "markdown"), None)
    if first is None or not first.source.lstrip().startswith("# "):
        problems.append("the first markdown cell must start with an H1 title")
    elif "(cpm." in first.source.splitlines()[0]:
        problems.append("the title must not list cpm modules")

    headings = [
        line
        for c in nb.cells
        if c.cell_type == "markdown"
        for line in c.source.splitlines()
        if line.startswith("## ")
    ]
    if not any("Summary" in h for h in headings):
        problems.append('no "## Summary" section')
    if path.parent.name != "get-started" and not any("References" in h for h in headings):
        problems.append('no "## References" section')

    counts = [c.execution_count for c in nb.cells if c.cell_type == "code"]
    if counts and counts != list(range(1, len(counts) + 1)):
        problems.append("cells were not run top to bottom in a fresh kernel")

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for pattern, message in BANNED_CODE:
            if pattern.search(cell.source):
                problems.append(f"cell {i} {message}")
        for output in cell.outputs:
            if output.output_type == "error":
                problems.append(f"cell {i} has an error output")
            if BANNED_MIME & set(output.get("data", {})):
                problems.append(f"cell {i} has widget or Data Wrangler output")
            if len(str(output)) > MAX_OUTPUT_BYTES:
                problems.append(f"cell {i} has an output larger than 1 MB")

    if path.parent.name == "examples":
        tags = [t for c in nb.cells for t in c.metadata.get("tags", [])]
        if "thumbnail" not in tags:
            problems.append('no cell is tagged "thumbnail"')
    return problems


def main():
    failed = False
    for folder in FOLDERS:
        for path in sorted((DOCS / folder).glob("*.ipynb")):
            for problem in check(path):
                print(f"{path.relative_to(DOCS)}: {problem}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
