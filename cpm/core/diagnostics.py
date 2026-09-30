import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parameter_bounds(parameters):
    """
    Collect the lower and upper bounds of the freely-varying parameters.

    Parameters
    ----------
    parameters : cpm.generators.Parameters
        The parameters of the model.

    Returns
    -------
    dict
        A dictionary mapping each freely-varying parameter with scalar bounds to a `(lower, upper)` tuple.
    """
    bounds = {}
    for name in parameters.free():
        lower, upper = parameters[name].lower, parameters[name].upper
        if np.ndim(lower) == 0 and np.ndim(upper) == 0:
            bounds[name] = (lower, upper)
    return bounds


def convergence_diagnostics_plots(
    hyperparameters, show=True, save=False, path=None, bounds=None
):
    # hyperparameters have columns: parameter, iteration, chain, lme, mean, and sd
    ## Plot the convergence diagnostics
    """
    This function plots the convergence diagnostics for the hyperparameters of the model.
    The hyperparameters should have the following columns: parameter, iteration, chain, lme, mean, and sd.

    Parameters
    ----------
    hyperparameters : pandas.DataFrame
        The hyperparameters of the model.
    bounds : dict, optional
        A dictionary mapping parameter names to `(lower, upper)` tuples, used as the y-axis limits of the trace plots.
        Parameters without finite bounds are scaled to the range of the data.
    """

    parameters = hyperparameters.parameter.unique()
    bounds = bounds if bounds is not None else {}

    mosaic = [["lme", "lme", "lme"]]

    for _, names in enumerate(parameters):
        mosaic.append([names + " mean", names + " sd", names + " traces"])

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(12, 8))

    for chain in hyperparameters.chain.unique():
        lme = hyperparameters.loc[
            (hyperparameters.chain == chain)
            & (hyperparameters.parameter == parameters[0]),
            "lme",
        ]
        iteration = np.arange(len(lme))
        axs["lme"].plot(iteration, lme)

    axs["lme"].set_title("Log Model Evidence")
    axs["lme"].set_xlabel("Iteration")
    axs["lme"].set_ylabel("LME")

    for names in parameters:
        for chain in hyperparameters.chain.unique():
            ## extract values here for readability of code
            means = hyperparameters.loc[
                (hyperparameters.chain == chain) & (hyperparameters.parameter == names),
                "mean",
            ]
            sd = hyperparameters.loc[
                (hyperparameters.chain == chain) & (hyperparameters.parameter == names),
                "sd",
            ]
            iteration = np.arange(len(means))

            ## plot the mean
            axs[names + " mean"].hist(
                means.to_numpy(),
                alpha=0.5,
            )
            axs[names + " mean"].set_title(rf"$\mu_{{{names}}}$")

            ## plot the standard deviation
            axs[names + " sd"].hist(
                sd.to_numpy(),
                alpha=0.5,
            )
            axs[names + " sd"].set_title(rf"$\sigma_{{{names}}}$")

            ## plot the traces
            axs[names + " traces"].plot(iteration, means)
            axs[names + " traces"].fill_between(
                iteration,
                means - sd,
                means + sd,
                alpha=0.5,
            )

            ## set x-axis ticks to be integers for readability
            axs[names + " traces"].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[names + " traces"].set_xlabel("Iteration")
            axs[names + " traces"].set_title(rf"$traces_{{{names}}}$")

        ## limit the traces to the parameter bounds, where they are finite
        lower, upper = bounds.get(names, (-np.inf, np.inf))
        if np.isfinite(lower) and np.isfinite(upper):
            axs[names + " traces"].set_ylim(lower, upper)

    ## add legend to figure
    fig.legend(
        labels=hyperparameters.chain.unique(),
        loc="center left",
        ncol=1,
        title="Chain",
    )

    fig.tight_layout()

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        return fig


def gelman_rubin(hyperparameters):
    """
    This function calculates the Gelman-Rubin statistic for the hyperparameters of the model.
    The hyperparameters should have the following columns: parameter, iteration, chain, lme, mean, and sd.

    Returns
    -------
    pandas.DataFrame
        A table with one row per parameter and hyperparameter (mean and sd), and the Gelman-Rubin statistic in the `rhat` column.

    Notes
    -----
    Chains can stop at different iterations, so they are compared over the iterations they have in common.
    At least two chains with at least two iterations each are required.
    """
    chains = hyperparameters.chain.unique()
    if len(chains) < 2:
        raise ValueError("The Gelman-Rubin statistic requires at least two chains.")

    records = []
    for names in hyperparameters.parameter.unique():
        subset = hyperparameters.loc[hyperparameters.parameter == names]
        for variables in ["mean", "sd"]:
            traces = [
                subset.loc[subset.chain == chain]
                .sort_values("iteration")[variables]
                .to_numpy(dtype=float)
                for chain in chains
            ]
            n = min(len(trace) for trace in traces)
            if n < 2:
                raise ValueError(
                    "The Gelman-Rubin statistic requires at least two iterations per chain."
                )
            traces = np.stack([trace[:n] for trace in traces])

            ## within-chain (W) and between-chain (B) variances
            W = traces.var(axis=1, ddof=1).mean()
            B = n * traces.mean(axis=1).var(ddof=1)
            V = (n - 1) / n * W + B / n
            if W > 0:
                rhat = np.sqrt(V / W)
            else:
                # constant chains agree only if they sit at the same value
                rhat = 1.0 if B == 0 else np.inf

            records.append(
                {"parameter": names, "hyperparameters": variables, "rhat": rhat}
            )

    return pd.DataFrame(records, columns=["parameter", "hyperparameters", "rhat"])


def psrf(hyperparameters):
    """
    This function calculates the potential scale reduction factor for the hyperparameters of the model.
    The hyperparameters should have the following columns: parameter, iteration, chain, lme, mean, and sd.

    Returns
    -------
    pandas.DataFrame
        A table with one row per parameter and hyperparameter (mean and sd), and the potential scale reduction factor in the `psrf` column.
    """
    rhat = gelman_rubin(hyperparameters)
    psrf = (
        rhat.groupby(["parameter", "hyperparameters"], sort=False)["rhat"]
        .apply(lambda values: np.sqrt(np.mean(values**2)))
        .reset_index(name="psrf")
    )
    return psrf
