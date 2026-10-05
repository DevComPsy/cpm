"""Local Sphinx extension for the cpm documentation.

It extracts gallery thumbnails from notebooks: the first PNG output of the
code cell tagged ``thumbnail`` is written to ``_static/thumbnails/<name>.png``.
Notebooks without a tagged cell get the logo instead.
"""

import base64
import json
from pathlib import Path

GALLERY_DIRS = ("examples", "tutorials")
PLACEHOLDER = Path("_static") / "logo" / "cpm-mark.png"


def extract_thumbnails(app):
    srcdir = Path(app.srcdir)
    outdir = srcdir / "_static" / "thumbnails"
    outdir.mkdir(parents=True, exist_ok=True)
    for folder in GALLERY_DIRS:
        for notebook in sorted((srcdir / folder).glob("*.ipynb")):
            png = _thumbnail_png(notebook)
            if png is None:
                png = (srcdir / PLACEHOLDER).read_bytes()
            target = outdir / f"{notebook.stem}.png"
            if not target.exists() or target.read_bytes() != png:
                target.write_bytes(png)


def _thumbnail_png(notebook):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        if "thumbnail" not in cell.get("metadata", {}).get("tags", []):
            continue
        for output in cell.get("outputs", []):
            data = output.get("data", {}).get("image/png")
            if data:
                if isinstance(data, list):
                    data = "".join(data)
                return base64.b64decode(data)
    return None


def setup(app):
    app.connect("builder-inited", extract_thumbnails)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
