---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Faster fitting in cpm 0.26

*28 September 2026 · cpm 0.26.0, in development*

The built-in models of cpm now compute all trials of a participant in one call instead of one call per trial, and, when [numba](https://numba.pydata.org/) is installed, that loop over trials is compiled.
Fitting them is 90 to 230 times faster than in cpm 0.25, and more than 1,000 times faster for {py:class}`~cpm.applications.decision_making.PTSM2025`, with the same results.
Your code does not change.

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RESULTS = "../../benchmarks/results"
main = pd.read_csv(f"{RESULTS}/report-main.csv")
branch = pd.read_csv(f"{RESULTS}/report-branch.csv")
hierarchical = pd.read_csv(f"{RESULTS}/hierarchical-report.csv")

MODELS = ["RLRW", "HybridMBMF", "PTSM", "PTSM1992", "PTSM2025"]
## one colour per variant, the same in every figure (checked for colour-vision deficiencies)
SERIES = [
    ("main", "cpm 0.25", "#6b6a65", True),
    ("per_trial", "Per trial", "#1baf7a", False),
    ("python", "Without numba", "#eb6834", False),
    ("numba", "With numba", "#2a78d6", False),
]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

evaluation = pd.DataFrame(
    {
        "model": MODELS,
        "trials": [int(main.set_index("case").loc[m, "trials"]) for m in MODELS],
        "main": [main.set_index("case").loc[m, "objective_ms"] for m in MODELS],
        **{
            key: [branch[(branch.case == m) & (branch.backend == backend)].objective_ms.iloc[0] for m in MODELS]
            for key, backend in (("per_trial", "trial"), ("python", "python"), ("numba", "numba"))
        },
    }
)


def dot_plot(rows, labels, unit, ticks, tick_labels, series=SERIES):
    """One row per item, one dot per variant, on a log scale; the speed-up of numba over cpm 0.25 above its dot."""
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(rows) + 1.1), dpi=144)
    fig.patch.set_facecolor("white")
    for i, (_, row) in enumerate(rows.iterrows()):
        y = len(rows) - 1 - i
        values = [row[key] for key, *_ in series]
        ax.plot([min(values), max(values)], [y, y], color=AXIS, linewidth=1, zorder=1)
        for key, _, color, hollow in series:
            if hollow:
                ax.scatter(row[key], y, s=58, facecolor="white", edgecolor=color, linewidth=2, zorder=3)
            else:
                ax.scatter(row[key], y, s=72, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{row['main'] / row['numba']:,.0f}×", (row["numba"], y), xytext=(0, 9),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xticks(ticks, tick_labels)
    ax.minorticks_off()
    ax.set_yticks(range(len(rows)), labels[::-1])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=10, length=0)
    ax.set_xlabel(unit, color=INK_2, fontsize=9.5)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=7.5, markerfacecolor="white" if hollow else color,
               markeredgecolor=color, markeredgewidth=2 if hollow else 0, label=label)
        for _, label, color, hollow in series
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False,
              fontsize=9, handletextpad=0.3, columnspacing=1.4, labelcolor=INK_2)
    fig.tight_layout()
    return fig
```

## What changes for you

- **Install numba** with cpm: `pip install "cpm-toolbox[numba]"`. That is all it takes. {py:class}`~cpm.applications.reinforcement_learning.RLRW`, {py:class}`~cpm.applications.reinforcement_learning.HybridMBMF`, {py:class}`~cpm.applications.decision_making.PTSM`, {py:class}`~cpm.applications.decision_making.PTSM1992` and {py:class}`~cpm.applications.decision_making.PTSM2025` keep their arguments, parameters, priors and outputs, and give the same results.
- **Without numba**, the same models run as plain Python, 13 to 25 times faster than in cpm 0.25 (about 250 times for `PTSM2025`). numba is optional because it supports a new NumPy release only some time after it comes out, and does not support PyPy.
- **Your own models are faster too.** Every per-trial {py:class}`~cpm.generators.Wrapper` model runs about twice as fast, and you can compile your own models with the new {py:class}`~cpm.generators.SessionWrapper`: see {doc}`/how-to/fast-session-models`.
- **Hierarchical results change.** {py:class}`~cpm.hierarchical.EmpiricalBayes` and {py:class}`~cpm.hierarchical.VariationalBayes` used to discard the population priors they estimated after the first evaluation of each fit. They now use them throughout, so their results differ from cpm 0.25 (see [below](#same-results-and-one-fix)).

## How much faster

Fitting a model evaluates its objective function (the negative log posterior) at one parameter set after another, thousands of times per participant.
The figure shows how long one evaluation takes for each built-in model, fitted to one participant.

```{code-cell} ipython3
:tags: [remove-input]

fig = dot_plot(
    evaluation,
    [f"{m}  ({t} trials)" for m, t in zip(evaluation.model, evaluation.trials)],
    "milliseconds per evaluation of the objective function (log scale)",
    [0.1, 1, 10, 100], ["0.1", "1", "10", "100"],
)
plt.show()
```

**Per trial** is a model's own per-trial function, `model(parameters, trial)`, in a plain `Wrapper`, which is how you extend a built-in model trial by trial; it runs without numba.
**Without numba** and **With numba** are the built-in model computing all trials at once.

```{code-cell} ipython3
:tags: [remove-input]

table = evaluation.set_index("model")[["trials", "main", "per_trial", "python", "numba"]].copy()
table["speed-up"] = (table["main"] / table["numba"]).map(lambda x: f"{x:,.0f}×")
table.columns = ["Trials", "cpm 0.25, ms", "Per trial, ms", "Without numba, ms", "With numba, ms", "Speed-up with numba"]
table.index.name = None
table.style.format({c: "{:.3g}" for c in table.columns[1:5]})
```

With numba, running the model now takes about as long as computing the loss, and resetting the parameters and evaluating the priors, which took about 1 ms per evaluation in cpm 0.25, take a few microseconds.

## Hierarchical estimation

The two hierarchical tutorials, {doc}`/tutorials/hierarchical-empirical-bayes` and {doc}`/tutorials/hierarchical-variational-bayes`, simulate 50 participants and fit them with two starts each, in 2 chains of up to 6 iterations; each iteration refits all 50 participants.
Their model is written out per trial, and is the built-in `RLRW` model, which gives the same estimates.

In cpm 0.25, the fitters stopped after fewer iterations, because discarding the updated priors made each iteration refit almost the same model.
The figure therefore compares the same amount of work: the separate maximum-likelihood fits, and one iteration of each method.

```{code-cell} ipython3
:tags: [remove-input]

variant = {"main, tutorial model": "main", "branch, tutorial model": "per_trial",
           "branch, RLRW without numba": "python", "branch, RLRW with numba": "numba"}
h = hierarchical.assign(key=hierarchical.variant.map(variant))
eb = h[h.analysis == "empirical Bayes"].set_index("key")
vb = h[h.analysis == "variational Bayes"].set_index("key")
work = pd.DataFrame(
    [
        {"item": "Separate fits", **eb.separate_fits_s.to_dict()},
        {"item": "Empirical Bayes, one iteration", **eb.seconds_per_iteration.to_dict()},
        {"item": "Variational Bayes, one iteration", **vb.seconds_per_iteration.to_dict()},
    ]
)
HSERIES = [
    ("main", "cpm 0.25 (tutorial model)", "#6b6a65", True),
    ("per_trial", "Tutorial model", "#1baf7a", False),
    ("python", "RLRW without numba", "#eb6834", False),
    ("numba", "RLRW with numba", "#2a78d6", False),
]
fig = dot_plot(work, list(work["item"]), "seconds (log scale)", [1, 10, 60], ["1 s", "10 s", "1 min"], HSERIES)
plt.show()
```

The whole analyses, fitting only, with the number of iterations they ran:

```{code-cell} ipython3
:tags: [remove-input]

def cell(row):
    return f"{row.fitting_total_s:,.0f} s ({row.iterations})" if row.fitting_total_s >= 10 else f"{row.fitting_total_s:.1f} s ({row.iterations})"


totals = pd.DataFrame(
    {name: [cell(frame.loc[k]) for k in ("main", "per_trial", "python", "numba")] for name, frame in (("Empirical Bayes", eb), ("Variational Bayes", vb))},
    index=["cpm 0.25, tutorial model", "Tutorial model", "RLRW without numba", "RLRW with numba"],
).T
totals
```

The tutorials themselves keep their per-trial model, which shows how a model is built from cpm's components; they run about twice as fast as before.

## Same results, and one fix

- The objective function of the built-in models is bit-identical to cpm 0.25, and so are all their outputs: exports, dependent variables, states and seeded simulations, with and without numba.
- The classes in {py:mod}`cpm.models` give bit-identical results, and the quickstart, the first tutorial and the two-step example reproduce their stored outputs.
- Hierarchical estimation changes, as intended. The table compares the group estimates of the first chain with the group distribution the data were simulated from.

```{code-cell} ipython3
:tags: [remove-input]

accuracy = pd.DataFrame(
    [
        [0.60, 0.20, 2.00, 1.00],
        [0.547, 0.297, 2.688, 2.369],
        [0.570, 0.202, 2.163, 0.916],
        [0.536, 0.304, 2.635, 2.375],
        [0.553, 0.229, 2.131, 0.981],
    ],
    index=["True group distribution", "Empirical Bayes, cpm 0.25", "Empirical Bayes, cpm 0.26",
           "Variational Bayes, cpm 0.25", "Variational Bayes, cpm 0.26"],
    columns=["α mean", "α SD", "β mean", "β SD"],
)
accuracy.style.format("{:.2f}")
```

In cpm 0.25, every participant was fitted with the broad starting priors, so the estimated spread of the inverse temperature β was more than twice its true value.

## How we measured

- **Machine:** Intel Xeon w5-2455X (12 cores), 32 GB, Windows 11; Python 3.12.4, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, numba 0.67.0.
- **Code:** cpm 0.25 is `main` at commit `32629af`; cpm 0.26 is the development version at commit `830d479`.
- **One evaluation:** `python benchmarks/run.py --repeats 300`, the median of 300 calls after one warm-up call.
- **Hierarchical:** `python benchmarks/hierarchical.py eb|vb tutorial|rlrw OUT`, single runs.

All runs used one thread for NumPy's linear algebra, ran one at a time, and loaded numba's compiled code from its cache.
The first run of a model with numba compiles it once, which takes a few seconds (up to about 15 s for the prospect-theory models); after that, cpm loads the compiled code from disk.
The results on other machines, operating systems and Python versions will differ, but the ratios should be similar.
The data behind the figures are in `benchmarks/results/` in the [cpm repository](https://github.com/DevComPsy/cpm).

For every change in this version, see the {doc}`changelog </about/changelog>`.
