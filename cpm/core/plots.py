import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize


def convergence_diagnostics(hyperparameters, show=True, save=False, path=None):
    # hyperparameters have columns: parameter, iteration, chain, lme, mean, and sd
    ## Plot the convergence diagnostics
    """
    This function plots the convergence diagnostics for the hyperparameters of the model.
    The hyperparameters should have the following columns: parameter, iteration, chain, lme, mean, and sd.

    """

    parameters = hyperparameters.parameter.unique()

    # Set up the figure (parameters + 1 for the LME)
    fig, axes = plt.subplots(len(parameters) + 1, 1, figsize=(10, 10 * len(parameters)))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle("Convergence Diagnostics")

    # Plot the LME
    axes[0].set_title("Log Marginal Likelihood")
    for chain in hyperparameters.chain.unique():
        lme = hyperparameters.loc[hyperparameters.chain == chain, "lme"]
        axes[0].plot(lme, label=f"Chain {chain}")
    axes[0].legend()
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("LME")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot the parameters
    for i, parameter in enumerate(parameters):
        axes[i + 1].set_title(parameter)
        for chain in hyperparameters.chain.unique():
            axes[i + 1].plot(
                hyperparameters.loc[
                    (hyperparameters.chain == chain)
                    & (hyperparameters.parameter == parameter),
                    "mean",
                ],
                label=f"Chain {chain}",
            )
        axes[i + 1].legend()
        axes[i + 1].set_xlabel("Iteration")
        axes[i + 1].set_ylabel("Mean")
        axes[i + 1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        return fig
