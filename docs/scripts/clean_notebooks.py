"""Clean the stored outputs and metadata of the documentation notebooks.

- removes ipywidget outputs, VS Code Data Wrangler outputs and text progress bars,
- removes widget state and other editor metadata,
- sets a standard ``python3`` kernelspec.

Usage::

    python docs/scripts/clean_notebooks.py docs/tutorials/first-model.ipynb
    python docs/scripts/clean_notebooks.py docs   # every notebook under docs/
"""

import re
import sys
from pathlib import Path

import nbformat

DROP_MIME = {
    "application/vnd.jupyter.widget-view+json",
    "application/vnd.microsoft.datawrangler.viewer.v0+json",
}
KERNELSPEC = {"name": "python3", "display_name": "Python 3", "language": "python"}
KEEP_METADATA = {"kernelspec", "language_info"}
# tqdm-style progress bars, such as the one ipyparallel prints when it starts a cluster
PROGRESS_BAR = re.compile(r"\d+%\|")


def clean_output(output):
    """Return the output without unwanted mime types, or None to drop it."""
    if output.get("output_type") == "stream":
        text = "".join(output.get("text", ""))
        if PROGRESS_BAR.search(text) or not text.strip():
            return None
        return output
    data = output.get("data")
    if data is None:
        return output
    for mime in DROP_MIME:
        data.pop(mime, None)
    if not data or set(data) == {"text/plain"} and _is_widget_repr(data["text/plain"]):
        return None
    return output


def _is_widget_repr(text):
    text = "".join(text) if isinstance(text, list) else text
    return text.startswith(("HBox(", "VBox(", "IntProgress(", "FloatProgress("))


def clean(path):
    nb = nbformat.read(path, as_version=4)
    nb.metadata = {k: v for k, v in nb.metadata.items() if k in KEEP_METADATA}
    nb.metadata["kernelspec"] = KERNELSPEC
    for cell in nb.cells:
        cell.metadata.pop("vscode", None)
        cell.metadata.pop("widgets", None)
        if cell.cell_type == "code":
            outputs = [clean_output(o) for o in cell.outputs]
            cell.outputs = [o for o in outputs if o is not None]
    nbformat.write(nb, path)


def main(paths):
    for arg in paths:
        path = Path(arg)
        notebooks = [path] if path.is_file() else sorted(path.rglob("*.ipynb"))
        for notebook in notebooks:
            if "_build" in notebook.parts or ".ipynb_checkpoints" in notebook.parts:
                continue
            clean(notebook)
            print(f"cleaned {notebook}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["docs"])
