import numpy as np
from numpy.random import default_rng as rng

def grid(params, bounds, iterations=None):
    """
    Create a grid of parameters.

    Parameters:
    params (dict): The parameters.
    row (int): The number of parameter sets to create.

    Returns:
    list: The grid of parameters.
    """
    counts = len(params)
    if iterations is None:
        iterations = 1000
    else:
        iterations = sets
    grid = []
    for i in range(iterations):
        row = {}
        for j, k in zip(range(counts), params.keys()):
            generation = rng().uniform(bounds[j][0], bounds[j][1])
            row[k] = generation
        grid.append(row)
    return grid

def normal(params, bounds, iterations=None):
    """
    Create a grid of parameters.

    Parameters:
    params (dict): The parameters.
    row (int): The number of parameter sets to create.

    Returns:
    list: The grid of parameters.
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
    return grid

def priors(params, priors = None, bounds = None, generator = None, iterations=None):
    """
    Create a grid of parameters.

    Parameters:
    params (dict): The parameters.
    priors (array-like): The priors.
    bounds (array-like): The parameter bounds.
    generator (str): The random number generator from numpy.random.Generator.
    iterations (int): The number of parameter sets to create.

    Returns:
    list: The grid of parameters.
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
    return grid