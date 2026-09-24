"""Check that every public object of cpm has a page in the API reference.

An object is public if it is listed in the ``__all__`` of a subpackage, or
defined without a leading underscore in one of the modules listed below.

Usage::

    python docs/scripts/check_api_coverage.py
"""

import importlib
import inspect
import re
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1] / "api"

MODULES = [
    "cpm.generators",
    "cpm.hierarchical",
    "cpm.models.learning",
    "cpm.models.decision",
    "cpm.models.activation",
    "cpm.models.attention",
    "cpm.models.utils",
    "cpm.optimisation",
    "cpm.optimisation.minimise",
    "cpm.optimisation.compare",
    "cpm.applications.reinforcement_learning",
    "cpm.applications.decision_making",
    "cpm.applications.signal_detection",
    "cpm.datasets",
    "cpm.utils.data",
    "cpm.utils.metad",
]
# Helpers that are public by name but internal by intent.
IGNORE = {
    "cpm.optimisation.minimise.check_nan_and_bounds_in_input",
    "cpm.optimisation.minimise.check_nan_bounds_in_log",
}


def public_objects():
    for name in MODULES:
        module = importlib.import_module(name)
        names = getattr(module, "__all__", None)
        if names is None:
            names = [
                n
                for n, obj in vars(module).items()
                if not n.startswith("_")
                and (inspect.isclass(obj) or inspect.isfunction(obj))
                and obj.__module__ == name
            ]
        for n in names:
            obj = getattr(module, n)
            if inspect.ismodule(obj):
                continue
            yield f"{name}.{n}"


def documented():
    found = set()
    for page in API.glob("*.rst"):
        text = page.read_text(encoding="utf-8")
        current = re.search(r"^\.\. currentmodule:: (\S+)", text, re.M)
        prefix = current.group(1) if current else ""
        for block in re.findall(r"\.\. autosummary::\n((?:   .*\n|\n)+)", text):
            for line in block.splitlines():
                item = line.strip()
                if item and not item.startswith(":"):
                    found.add(f"{prefix}.{item}" if prefix else item)
    return found


def main():
    missing = sorted(set(public_objects()) - documented() - IGNORE)
    for name in missing:
        print(f"not in the API reference: {name}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
