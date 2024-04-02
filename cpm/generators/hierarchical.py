from .simulator import Simulator
from .parameters import Parameters, Value


class Hierarchical(Simulator):
    """
    A `Hierarchical` class for a model in the CPM toolbox. It is designed to run a model for **multiple** participants and store the output in a format that can be used for further analysis.

    Parameters
    ----------
    model : Wrapper
        An initialised Wrapper object for the model.
    data : object
        The data required for the simulation.
    parameters : Parameters
        The parameters required for the simulation.

    Attributes
    ----------
    function : object
        The simulation function to be used.
    data : object
        The data required for the simulation.
    parameters : Parameters
        The sampled parameters for each participant.
    parameter_names : object
        The names of the parameters.
    simulation : numpy.ndarray
        The results of the simulation, including the policies and the states.
    generated : object
        The results of the simulation, only including the policies.

    Returns
    -------
    simulator : Simulator
        A Simulator object.

    """

    def __init__(self, model=None, data=None, parameters=None):
        super().__init__(model=model, data=data, parameters=parameters)

        # TODO: create parameters for each ppt sampled from Parameters
        if isinstance(parameters, Parameters):
            self.parameters = [parameters.sample() for _ in range(len(data))]
        else:
            raise ValueError(
                "Parameters must be an instance of Parameters for the Hierarchical model."
            )

    def __repr__(self):
        return f"Hierarchical(model={self.function}, data={self.data}, parameters={self.parameters})"

    def __str__(self):
        return f"Hierarchical model: {self.function.name} with {len(self.data)} participants"
