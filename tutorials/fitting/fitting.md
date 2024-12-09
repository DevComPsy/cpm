# Fitting your models to data

Once you have specified your model and set up your parameters, you want to fit the model to your data.
`cpm` offers a set of optimisation techniques, a non-exhaustive list of some commonly used ones in Computational Psychiatry.
The main advantage of `cpm` here is that it takes the assumption that you want to fit the model to each subject's data separately, and that you want to fit the model to multiple subjects at once.
`cpm` covers most of the data cleaning and compilation steps for you, so it frees up time to do the more creative work.

## Optimisation

All optimisation routines already implemented will be found in the `cpm.optimisation` module, whereas `cpm.optimisation.minimise` handles all likelihood functions

```python
from cpm.optimisation import DifferentialEvolution, minimise

genetic = DifferentialEvolution(
    model=wrapped_model, # Wrapper class with the model we specified from before
    data=experiment, # the data as a list of dictionaries
    minimisation=minimise.LogLikelihood.bernoulli,
    parallel=True,
    ppt_identifier="ppt",
    display=False,
    maxiter=400, # additional arguments passed to scipy.optimise.differential_evolution
    tol=1e-10, # additional arguments passed to scipy.optimise.differential_evolution
)
```

There are a few things to note here.
First, you don't define bounds for the parameter space.
The reason for this is that you already defined bounds in the Parameter class for your model.
Bounds are part of the model specification and you will have to explicitly state them when you are building the model.
The second is that everything at the end that is not part of the named list of arguments for `cpm.optimisation.DifferentialEvolution` will be passed on the `scipy.optimize.differential_evolution` function.
`cpm` tries to avoid implementing methods from scratch, which is why we use `scipy` for the optimisation routines, and write most of the codebase with the intent on handling data cleaning and processing challenges.

### Extracting Parameters

Once the optimisation is done, you can extract the parameters from the optimisation object.
Here, you will have two options: `genetic.export()` and `genetic.parameters`.
The former will return a pandas DataFrame with the optimisation data, whereas the latter will return the parameters as a list dictionary.
You can use `genetic.export()` to save the optimisation data to a .csv file and do everything you would want to do with a pandas DataFrame.
`genetic.parameters` is useful if you want to use the parameters for further simulations or analyses.
You can directly feed it to the `cpm.generators.Simulator` class to simulate:

```python
simulation = Simulator(
    wrapper=wrapped_model, parameters=genetic.parameters, data=experiment
)
simulation.run()
simulation_results = simulation.export() # export the simulation results to a pandas DataFrame
simulation_results.to_csv("simulation.csv", index=False) # save the simulation results to a .csv file
```

## Parallel computing on Windows

Issue[#16](https://github.com/DevComPsy/modelling-toolbox/issues/16) is a known issue with Windows and parallelisation.
The way to circumvent this is to wrap your code in into `if __name__ == '__main__':` and load the parallelised optimisation bits as separate modules.

### Walkthrough

Assume that you have the following folder structure:

```bash

```bash
project_name/
├── docs/
│   ├── lab_report.ipynb
│   └── data/
│       └── big_data_file.csv
├── src/
│   ├── main.py
│   └── utils/
│       ├── __init__.py
│       ├── setup_model.py
│       └── fitting.py
└── README.md
```

Feel free to adjust this structure based on your specific project needs.
fitting.py and setup_model.py are the files that contain the fitting and model setup functions, respectively.

For setup_model.py, you can structure it as follows:

```python
def model_setup(data):
    """
    The following function sets up the model.

    Parameters
    ----------
    data : list
        The data as a list of dictionaries.

    Return
    ------
    parameters : cpm.generators.Parameters
        The parameter object for the model
    wrapper : cpm.generators.Wrapper
        The Wrapper object with the model specified below.
    """

    from cpm.models import learning, decision
    from cpm.generators import Value, Parameters, Wrapper
    import numpy as np

    parameters = Parameters(
        # ... your parameters here
    )

    def model(parameters, trial):
        # ... your model code here
        return output

    wrapper = Wrapper(model=model, parameters=parameters, data=data)

    print(wrapper)

    return parameters, wrapper
```

Your `fitting.py` file can be structured as follows:

```python
def fitting(wrapper, experiment):
    """
    The following function is used to fit the model to the data using the Differential Evolution algorithm. The function also saves the optimisation data and the simulation data as .csv files.

    Parameters
    ----------
    wrapper : cpm.generators.Wrapper
        The wrapper class with the model we specified from before.
    experiment : list
        The data as a list of dictionaries.

    Returns
    -------
    data : pandas.DataFrame
        The optimisation data.
    """

    from cpm.optimisation import minimise, DifferentialEvolution
    from cpm.generators import Wrapper, Simulator, Parameters
    import pickle as pkl

    genetic = DifferentialEvolution(
        model=wrapper,  # Wrapper class with the model we specified from before
        data=experiment,  # the data as a list of dictionaries
        minimisation=minimise.LogLikelihood.bernoulli,
        # ... additional arguments go here
    )

    genetic.optimise()

    data = genetic.export()
    data.to_csv("optimisation.csv", index=False) # make sure to save stuff

    simulation = Simulator(
        wrapper=wrapper, parameters=genetic.parameters, data=experiment
    )
    simulation.run()

    simulation_data = simulation.export()
    simulation_data.to_csv("simulation.csv", index=False) # make sure to save stuff

    return data

```

**The important part is that you have a `main.py` file that will be the entry point for your project.**
You can structure your `main.py` as follows:

```python
if __name__ == "__main__":
    import sys
    import os
    import time
    import pandas as pd
    from cpm.utils import pandas_to_dict
    from pickle import load

    # get the directory of the current file, and add to sys.path if necessary
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    print("Importing data")

    file_path = r"some_data_file.csv"

    experiment = pd.read_csv(file_path)
    experiment.head(10)

    experiment = experiment.astype(int)

    environment = pandas_to_dict(
        experiment,
        participant="ppt",
        stimuli="Stim",
        feedback="Feedback",
        observed="Response",
        trial_number="trial",
    )

    from utils.model import model_setup
    from utils.fitting import fitting

    print("\nSetting up model")
    # model_setup is a function that returns a cpm.generators.Parameters and a cpm.generators.Wrapper object
    parameters, learning_model = model_setup(environment[0])

    # time to fit the model
    start = time.time()

    print(f"\n {start} : Fitting model")
    # fitting is a function that returns a pandas dataframe with the fitted parameters and other information
    output = fitting(learning_model, environment)
    end = time.time()
    print(f"\n {end} : Model fitted in {end - start} seconds")
```

If you have this sorted, you can execute the `main.py` file from the command line, and the parallelisation should work as expected.

```bash
python src/main.py
```

### ImportError: DLL load failed while importing arrays: The paging file is too small for this operation to complete. python multiprocess

This error is due to the fact that the Windows operating system has a limit on the number of processes that can be spawned.
Possible solution are (1) to increase the size of the paging file; (2) to decrease the number of processes spawned; or (3) to use a different operating system via `docker`.
`cpm` provides a ready-to-ue `docker` container via [DockerHub](https://hub.docker.com/repository/docker/lenarddome/dcp-cpm-latest-dev/general), which you can use to run your code.