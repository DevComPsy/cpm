"""
This module provides a simple benchmarking set-up to test the performance
of parallel versions of the CPM optimisation code.

Specifically, it uses timeit to run the cpm.optimisation.Fmin optimiser
both sequentially, and parallely with different numbers of cores.

In a virtual environment with cpm installed, one can run
`python benchmark.py` 
to run. There are a number of options available:
    - `-rs` will change the number of repeats that timeit averages across
    - `-ppts` allows you to set the number of participants to simulate
    - `--fixed` will use predefined trial information; otherwise, trial
        information will be randomised
    - `--complex` will increase the amount of information per trial
    - `-v` wiil print out the setup and command code that is being passed
        to timeit to run the benchmark
"""

import timeit
import argparse

# import components we want to use
from cpm.models.utils import Nominal
from cpm.models.learning import DeltaRule
from cpm.models.decision import Sigmoid

# import some useful stuff
import numpy as np

from cpm.optimisation import minimise, Fmin
import multiprocess as mp

# This defines the code that will be run before running the timeit
# benchmark for the sequential case
SEQUENTIAL_TEMPLATE = '''
from cpm.generators import Parameters, Wrapper
import numpy as np
from __main__ import fit_and_optimise, create_participants, model
parameters = Parameters(
    alpha=0.1, temperature=1, values=np.array([[0.25, 0.25, -0.25, -0.25]])
)
participants = create_participants({}, {}, {})
wrap = Wrapper(model=model, parameters=parameters, data=participants[0])
'''

# This defines the code that will be run before running the timeit
# benchmark for each parallel case
PARALLEL_TEMPLATE = '''
from cpm.generators import Parameters, Wrapper
import numpy as np
from __main__ import fit_and_optimise, create_participants, model
parameters = Parameters(
    alpha=0.1, temperature=1, values=np.array([[0.25, 0.25, -0.25, -0.25]])
)
participants = create_participants({}, {}, {})
wrap = Wrapper(model=model, parameters=parameters, data=participants[0])
cores = {}
'''


# define a quick model
def model(parameters, trial):
    # import parameters
    alpha = parameters.alpha
    temperature = parameters.temperature
    # import variables
    weights = parameters.values

    # import trial
    stimulus = trial.get("trials")
    stimulus = Nominal(target=stimulus, bits=4)
    feedback = trial.get("feedback")

    # specify the computations for a given trial
    # multiply weights with stimulis vector
    active = stimulus * weights
    # this is a mock-up policy
    policy = Sigmoid(temperature=temperature,
                     beta=0.5,
                     activations=active).compute()
    # learning
    error = DeltaRule(
        weights=active, feedback=feedback, alpha=alpha, input=stimulus
    ).compute()
    weights += error
    output = {
        "policy": np.array([policy]),
        "values": weights,
        "dependent": np.array([policy]),
    }
    return output


def fit_and_optimise(model, data, parallel=False, cores=None):
    """
    `fit_and_optimise` creates a cpm.optimise.Fmin object for a particular
    model and dataset, and then attempts to optimise it either sequentially
    or in parallel across a specified number of cores.

    Parameters
    ----------
    model
        The specific model to optimise
    data
        Participant trial data
    parallel
        Whether to run the optimiser sequentially or in parallel
    cores
        If running in parallel, the number of cores to use
    """
    Fit = Fmin(
        model=model,
        data=data,
        initial_guess=[0.32, 0.788],
        minimisation=minimise.LogLikelihood.bernoulli,
        parallel=parallel,  # parallel True will use all available cores
        cl=cores,
        disp=False,  # kwargs passed to scipy.fmin - this will suppress output
    )
    Fit.optimise()


def create_participants_fixed(num):
    """
    `create_participants_fixed` creates a specified number of identical
    participants.

    Parameters
    ----------
    num
        The number of participants to create

    Returns
    -------
        An array of participant data dictionaries
    """
    experiment = []
    for i in range(num):
        ppt = {
            "ppt": i,
            "trials": np.array(
                [
                    [2, 3],
                    [1, 4],
                    [3, 2],
                    [4, 1],
                    [2, 3],
                    [2, 3],
                    [1, 4],
                    [3, 2],
                    [4, 1],
                    [2, 3],
                ]
            ),
            "feedback": np.array(
                [[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]
            ),
            "observed": np.array(
                [[1], [0], [1], [0], [1], [1], [0], [1], [0], [1]]
            ),
        }
        experiment.append(ppt)
    return experiment


def create_participants_random(num, complex):
    """
    `create_participants_fixed` creates a specified number of
    participants with random `trial`, `observed`, and `feedback` values.

    Note, this is a bit of a stub at the moment, because, apart from
    varying the number of trials, there is very little difference between
    simple and complex participants

    Parameters
    ----------
    num
        The number of participants to create
    complex
        A boolean that specifies whether to create simple (10 trial)
        participants, or complex (100 trial) participants

    Returns
    -------
        An array of participant data dictionaries
    """
    num_trials = 10
    trial_vals = list(range(4))
    feedback_vals = [0, 1]
    if complex:
        num_trials = 100
    experiment = []
    for i in range(num):
        trials = []
        feedback = []
        observed = []
        for t in range(num_trials):
            trials.append([np.random.choice(trial_vals),
                           np.random.choice(trial_vals)])
            feedback.append([np.random.choice(feedback_vals)])
            observed.append([np.random.choice(feedback_vals)])
        ppt = {
            "ppt": i,
            "trials": np.array(trials),
            "feedback": np.array(feedback),
            "observed": np.array(observed),
        }
        experiment.append(ppt)
    return experiment


def create_participants(num, fixed=False, complex=False):
    """
    `create_participants` chooses between fixed and random participants,
    based on the command line options used.

    Parameters
    ----------
    num
        The number of participants to create
    fixed
        Whether to used fixed or random participants
    complex
        A boolean that specifies whether to create simple (10 trial)
        participants, or complex (100 trial) participants

    Returns
    -------
        An array of participant data dictionaries
    """
    if fixed:
        return create_participants_fixed(num)
    return create_participants_random(num, complex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-rs", "--repeats", type=int, default=5)
    parser.add_argument("-ppts", "--participants", type=int, default=200)
    parser.add_argument("--fixed", action='store_true')
    parser.add_argument("--complex", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')

    args = parser.parse_args()

    ppt_gen_type = "Fixed" if args.fixed else "Random"
    ppt_type = "Complex" if args.complex else "Simple"
    print(f"Benchmarking for {args.participants} ({ppt_gen_type}, {ppt_type})")

    setup = SEQUENTIAL_TEMPLATE.format(args.participants, args.fixed,
                                       args.complex)
    cmd = 'fit_and_optimise(wrap, participants)'

    if args.verbose:
        print(setup)
        print(cmd)

    result = timeit.timeit(cmd,
                           setup=setup,
                           number=args.repeats)
    print(f"Sequential optimise (n={args.participants}): {result}")

    for cores in range(3, mp.cpu_count()+1):

        setup = PARALLEL_TEMPLATE.format(args.participants, args.fixed,
                                         args.complex, cores)
        cmd = 'fit_and_optimise(wrap, participants, parallel=True, cores=cores)'
        if args.verbose:
            print(setup)
            print(cmd)

        result = timeit.timeit(cmd,
                               setup=setup,
                               number=args.repeats)
        print(f"Parallel optimise (n={args.participants}, cores={cores}): {result}")
