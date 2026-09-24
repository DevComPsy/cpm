"""
Where does the time of one evaluation of the objective go, for every built-in
application of cpm and every backend?

    python benchmarks/run.py [--repeats 200] [--label NAME]

Prints a table and writes benchmarks/results/<label>.csv. Times are medians of
wall-clock time per call. `backend` is "trial" for the original per-trial
applications and "python" or "numba" for their session versions.
"""

import argparse
import pathlib
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import cpm  # noqa: E402
from cpm.core.optimisers import objective  # noqa: E402
from cpm.optimisation.minimise import LogLikelihood  # noqa: E402

from cases import BACKENDS, CASES, build  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"


def clock(function, repeats):
    """Median wall-clock seconds of one call, after one warm-up call."""
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def layers(model, observed, x, repeats):
    """Seconds per call of the objective and of its parts."""
    loss = LogLikelihood.bernoulli
    out = {"objective": clock(lambda: objective(x, model, observed, loss, True), repeats)}

    def reset():
        model.__run__ = True
        model.reset(parameters=x)

    out["reset"] = clock(reset, repeats)
    out["prior"] = clock(lambda: model.parameters.PDF(log=True), repeats)
    model.reset(parameters=x)

    def run():
        model.__run__ = False
        model.run()

    out["run"] = clock(run, max(repeats // 4, 5))
    predicted = model.dependent.copy()
    out["loss"] = clock(lambda: loss(predicted=predicted, observed=observed), repeats)
    return out, -objective(x, model, observed, loss, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--label", default="current")
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    rows = []
    for case in args.cases:
        for backend in BACKENDS:
            built = build(case, backend)
            if built is None:
                continue
            model, observed, x = built
            times, log_posterior = layers(model, observed, x, args.repeats)
            rows.append(
                {
                    "case": case,
                    "backend": backend or "trial",
                    "trials": len(observed),
                    "parameters": len(x),
                    **{f"{k}_ms": v * 1e3 for k, v in times.items()},
                    "log_posterior": log_posterior,
                }
            )
            print(f"{case:>10} {backend or 'trial':>6}: {times['objective'] * 1e3:8.3f} ms", flush=True)

    table = pd.DataFrame(rows)
    trial = table[table.backend == "trial"].set_index("case")["objective_ms"]
    table["speed_up"] = trial.reindex(table.case).to_numpy() / table["objective_ms"]
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"{args.label}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    try:
        import numba

        numba_version = numba.__version__
    except ImportError:
        numba_version = "not installed"
    print(
        f"\ncpm {cpm.__version__} ({pathlib.Path(cpm.__file__).parent}), numpy {np.__version__}, "
        f"numba {numba_version}, Python {platform.python_version()}, {platform.system()}"
    )


if __name__ == "__main__":
    main()
