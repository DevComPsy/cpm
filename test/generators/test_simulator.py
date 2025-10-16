import pytest
import numpy as np
import pandas as pd
from cpm.generators.wrapper import Wrapper
from cpm.generators.simulator import Simulator
from cpm.generators.parameters import Parameters, Value


def dummy_model(parameters, trial):
    tmp = np.array([trial["stimulus"] * parameters.alpha])
    return {"dependent": tmp}


def test_simulator_initialization():
    data = pd.DataFrame(
        {"stimulus": [1, 2, 3, 1, 2, 3, 1, 2, 3], "ppt": [1, 1, 1, 2, 2, 2, 3, 3, 3]}
    )
    parameters = Parameters(alpha=Value(0.1, prior="norm"))
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    simulator = Simulator(
        wrapper=wrapper, data=data.groupby("ppt"), parameters=parameters.sample(3)
    )
    assert simulator.wrapper == wrapper, "Wrapper not set correctly"
    assert isinstance(
        simulator.data, pd.api.typing.DataFrameGroupBy
    ), "Data not set correctly, it should be identified as a DataFrameGroupBy object"
    assert simulator.groups == [1, 2, 3], "Groups not set correctly"
    assert simulator.parameter_names == ["alpha"], "Parameter names not set correctly"
    assert simulator.__pandas__ == True, "Pandas flag not set correctly"
    assert (
        simulator.__parameter__pandas__ == False
    ), "Parameter pandas flag not set correctly"
    assert len(simulator.parameters) == 3, "Parameters not set correctly"
    assert simulator.__run__ == False, "Run flag not set correctly"


def test_simulator_run():
    data = pd.DataFrame(
        {"stimulus": [1, 2, 3, 1, 2, 3, 1, 2, 3], "ppt": [1, 1, 1, 2, 2, 2, 3, 3, 3]}
    )
    parameters = Parameters(alpha=Value(0.1, prior="norm"))
    wrapper = Wrapper(
        model=dummy_model, data=data[data.ppt == 1], parameters=parameters
    )
    simulator = Simulator(
        wrapper=wrapper, data=data.groupby("ppt"), parameters=parameters.sample(3)
    )
    simulator.run()
    assert len(simulator.simulation) == 9, "Simulation not run correctly"
    assert len(simulator.simulation.iloc[0]) == 2, "Simulation output length is incorrect"


def test_simulator_export():
    data = pd.DataFrame(
        {"stimulus": [1, 2, 3, 1, 2, 3, 1, 2, 3], "ppt": [1, 1, 1, 2, 2, 2, 3, 3, 3]}
    )
    parameters = Parameters(alpha=Value(0.1, prior="norm"))
    wrapper = Wrapper(
        model=dummy_model, data=data[data.ppt == 1], parameters=parameters
    )
    simulator = Simulator(
        wrapper=wrapper, data=data.groupby("ppt"), parameters=parameters.sample(3)
    )
    simulator.run()
    exported_data = simulator.export()
    assert isinstance(exported_data, pd.DataFrame), "Exported data is not a DataFrame"
    assert exported_data.shape == (9, 2), "Exported data length is incorrect"


def test_simulator_reset():
    data = pd.DataFrame(
        {"stimulus": [1, 2, 3, 1, 2, 3, 1, 2, 3], "ppt": [1, 1, 1, 2, 2, 2, 3, 3, 3]}
    )
    parameters = Parameters(alpha=Value(0.1, prior="norm"))
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    simulator = Simulator(
        wrapper=wrapper, data=data.groupby("ppt"), parameters=parameters.sample(3)
    )
    simulator.run()
    simulator.reset()
    assert len(simulator.simulation) == 0, "Simulation not reset correctly"
    assert len(simulator.generated) == 0, "Generated data not reset correctly"


def test_simulator_generate():
    data = pd.DataFrame(
        {"stimulus": [1, 2, 3, 1, 2, 3, 1, 2, 3], "ppt": [1, 1, 1, 2, 2, 2, 3, 3, 3]}
    )
    parameters = Parameters(alpha=Value(0.1, prior="norm"))
    wrapper = Wrapper(
        model=dummy_model, data=data[data.ppt == 1], parameters=parameters
    )
    simulator = Simulator(
        wrapper=wrapper, data=data.groupby("ppt"), parameters=parameters.sample(3)
    )
    simulator.run()
    simulator.generate(variable="dependent")
    assert len(simulator.generated) == 3, "Generated data length is incorrect"


if __name__ == "__main__":
    pytest.main()
