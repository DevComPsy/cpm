"""
Run the test suite in each local configuration, then the benchmark suite.

    python scripts/local_tests.py [--python PATH ...] [--no-benchmark]

For each Python interpreter (by default the one running this script), the
suite runs three times:

- with numba, if it is installed in that environment;
- with ``CPM_DISABLE_JIT=1``, as if numba were not installed, which runs the
  built-in applications as plain Python;
- with ``NUMBA_DISABLE_JIT=1``, numba's own switch, for the kernel and application
  tests only.

To test several environments, create one virtual environment per Python
version, install cpm into each (``pip install -e .[numba] pytest``, and one
with ``pip install -e . pytest`` for an install without numba), and pass their
interpreters with ``--python``. The benchmark runs with the first interpreter
and writes benchmarks/results/local.csv.

numba's cache of compiled code is deleted first: numba only checks whether a
function's own file changed, so the compiled loops of cpm.applications would
otherwise keep using kernels of cpm.models.kernels that have since changed.

The script exits with a non-zero status if any run fails.
"""

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAST = ["test/models/test_kernels.py", "test/applications/test_backends.py"]


def run(python, environment, tests, label):
    env = {**os.environ, **environment}
    command = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    print(f"\n=== {label}: {' '.join(f'{k}={v}' for k, v in environment.items()) or 'numba on'}", flush=True)
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", nargs="*", default=[sys.executable])
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args()

    for path in ROOT.glob("cpm/**/__pycache__/*.nb[ci]"):
        path.unlink()

    failures = []
    for python in args.python:
        version = subprocess.run(
            [python, "-c", "import sys, platform; print(platform.python_version(), sys.platform)"],
            capture_output=True, text=True,
        ).stdout.strip()
        for environment, tests in (
            ({}, []),
            ({"CPM_DISABLE_JIT": "1"}, []),
            ({"NUMBA_DISABLE_JIT": "1"}, FAST),
        ):
            label = f"{python} (Python {version})"
            if run(python, environment, tests, label) != 0:
                failures.append((label, environment))

    if not args.no_benchmark:
        print("\n=== benchmark", flush=True)
        code = subprocess.run(
            [args.python[0], "benchmarks/run.py", "--label", "local"], cwd=ROOT
        ).returncode
        if code != 0:
            failures.append(("benchmark", {}))

    if failures:
        print("\nFailed:")
        for label, environment in failures:
            print(f"  {label} {environment}")
        sys.exit(1)
    print("\nAll local runs passed.")


if __name__ == "__main__":
    main()
