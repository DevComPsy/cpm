"""Check that every page of the old MkDocs site still resolves.

The old site served each page as a folder (``examples/bandit-task/``), so each
legacy URL must exist as ``<path>/index.html`` in the built site and redirect
to a page that exists.

Usage::

    python docs/scripts/check_redirects.py docs/_build/html
"""

import re
import sys
from pathlib import Path

LEGACY_PAGES = [
    "",
    "installation",
    "troubleshooting",
    "roadmap",
    "contributing",
    "examples/associative-learning",
    "examples/bandit-task",
    "examples/example6",
    "examples/fitting-hierarchical",
    "examples/fitting-hierarchical-estimation",
    "examples/fitting-hierarchical-run",
    "examples/functions/__init__",
    "examples/functions/model_anti",
    "examples/functions/model_delta",
    "examples/functions/model_kernel",
    "examples/hpc-example",
    "examples/metacognition",
    "examples/model-parameter-recovery",
    "examples/model-recovery",
    "examples/model-recovery-run",
    "api/applications",
    "api/datasets",
    "api/generators",
    "api/hierarchical",
    "api/models",
    "api/optimisation",
    "api/utils",
]
TARGET = re.compile(r'var target = "([^"]+)"')


def main(outdir):
    outdir = Path(outdir)
    failed = False
    for page in sorted(set(LEGACY_PAGES)):
        index = outdir / page / "index.html"
        if not index.exists():
            print(f"missing: /{page}/")
            failed = True
            continue
        match = TARGET.search(index.read_text(encoding="utf-8"))
        if match is None:
            continue  # a real page, such as the landing page
        target = (index.parent / match.group(1)).resolve()
        if not target.exists():
            print(f"/{page}/ redirects to a missing page: {match.group(1)}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html"))
