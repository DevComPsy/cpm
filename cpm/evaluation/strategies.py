import numpy as np
from numpy.random import default_rng as rng

def grid(template=None, bounds=None, population=None, **kwargs):
    """
    Create a grid of parameters using a uniform distribution.

    Parameters
    ----------
    template : dict
        The parameters.
    bounds : list
        The parameter bounds.
    population : int, optional
        The number of parameter sets to create. If not provided, default value is 1000.
    **kwargs : dict, optional
        Additional keyword arguments.

    Returns
    -------
    grid: numpy.ndarray
        The grid of parameters.
    """
    counts = len(template)
    if population is None:
        population = 1000
    grid = []
    for i in range(population):
        row = {}
        for j, k in zip(range(counts), template.keys()):
            generation = rng().uniform(bounds[j][0], bounds[j][1])
            row[k] = generation
        grid.append(row)
    return grid

def normal(params, bounds, iterations=None):
    """
    Create a grid of parameters using a normal distribution.

    Parameters
    ----------
    params : dict
        The parameters.
    bounds : list
        The upper bounds for each parameter.
    iterations : int, optional
        The number of parameter sets to create. If not provided, default value is 1000.

    Returns
    -------
    grid: numpy.ndarray
        The grid of parameters.
    """
    
    counts = len(params)
    if iterations is None:
        iterations = 1000
    else:
        iterations = sets
    generation = abs(rng().normal(0, 0.5, size=(iterations, counts)))
    grid = []
    for i in range(iterations):
        row = {}
        for j, k in zip(range(counts), params.keys()):
            row[k] = generation[i][j] * bounds[j][1]
        grid.append(row)
    return np.array(grid)

# FIXME: This is not a prior, it's a grid - incorporate prior
def priors(params, priors=None, bounds=None, generator=None, iterations=None):
    """
    Create a grid of parameters using a prior.

    Parameters
    ----------
    params : dict
        The parameters.
    priors : array-like, optional
        The priors.
    bounds : array-like, optional
        The parameter bounds.
    generator : str, optional
        The random number generator from numpy.random.Generator.
    iterations : int, optional
        The number of parameter sets to create.

    Returns
    -------
    list: numpy.ndarray
        The grid of parameters.
    """
    
    generator = getattr(rng(), generator)
    counts = len(params)
    if iterations is None:
        iterations = 1000
    else:
        iterations = sets
    generation = abs(generator(0, 0.5, size=(iterations, counts)))
    grid = []
    for i in range(iterations):
        row = {}
        for j, k in zip(range(counts), params.keys()):
            row[k] = generation[i][j] * bounds[j][1]
        grid.append(row)
    return np.array(grid)