"""Build the documentation and serve it locally.

Run from anywhere, with the environment in which cpm and the documentation
dependencies are installed (``pip install -e ".[docs]"``) activated::

    python docs/scripts/serve.py            # build once, then serve at http://localhost:8000
    python docs/scripts/serve.py --live     # rebuild and reload the browser on every change
    python docs/scripts/serve.py --clean    # delete the previous build first
    python docs/scripts/serve.py --port 8080
"""

import argparse
import functools
import http.server
import shutil
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
BUILD = DOCS / "_build" / "html"
GENERATED = [DOCS / "_build", DOCS / "api" / "generated", DOCS / "jupyter_execute"]


def check_environment():
    """Stop early if the docs would be built with the wrong cpm, or without Sphinx."""
    try:
        import cpm
    except ImportError:
        sys.exit('cpm is not installed. Run: pip install -e ".[docs]"')
    if ROOT not in Path(cpm.__file__).resolve().parents:
        sys.exit(
            f"This Python imports cpm from {Path(cpm.__file__).parent},\n"
            "not from this repository, so the documentation would describe the wrong version.\n"
            'Activate the environment of this repository, or run: pip install -e ".[docs]"'
        )
    for module in ("sphinx", "pydata_sphinx_theme", "myst_nb"):
        try:
            __import__(module)
        except ImportError:
            sys.exit(f'{module} is not installed. Run: pip install -e ".[docs]"')


def build():
    command = [sys.executable, "-m", "sphinx", "-b", "html", str(DOCS), str(BUILD)]
    if subprocess.run(command, cwd=ROOT).returncode != 0:
        sys.exit("The build failed; see the errors above.")


def serve(port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(BUILD))
    with http.server.ThreadingHTTPServer(("localhost", port), handler) as server:
        print(f"\nServing the documentation at http://localhost:{port} (press Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def live(port):
    if shutil.which("sphinx-autobuild") is None:
        sys.exit('sphinx-autobuild is not installed. Run: pip install -e ".[docs]"')
    command = [
        sys.executable, "-m", "sphinx_autobuild", str(DOCS), str(BUILD),
        "--watch", str(ROOT / "cpm"),
        # these folders are written by the build itself; watching them would rebuild forever
        "--ignore", "*/api/generated/*",
        "--ignore", "*/_static/thumbnails/*",
        "--ignore", "*/jupyter_execute/*",
        "--port", str(port),
        "--open-browser",
    ]
    try:
        subprocess.run(command, cwd=ROOT)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Build the cpm documentation and serve it locally.")
    parser.add_argument("--live", action="store_true", help="rebuild and reload on every change")
    parser.add_argument("--clean", action="store_true", help="delete the previous build first")
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (default: 8000)")
    args = parser.parse_args()

    check_environment()
    if args.clean:
        for folder in GENERATED:
            shutil.rmtree(folder, ignore_errors=True)
    if args.live:
        live(args.port)
    else:
        build()
        serve(args.port)


if __name__ == "__main__":
    main()
