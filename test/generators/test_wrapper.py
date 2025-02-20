import pytest
import numpy as np
import pandas as pd
from cpm.generators import Wrapper
from cpm.generators import Parameters, Value


def dummy_model(parameters, trial):
    tmp = np.array([trial["stimulus"] * parameters.alpha])
    return {"dependent": tmp}


def test_wrapper_initialization():
    data = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1, 2, 3]})
    parameters = Parameters(alpha=0.1)
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    assert wrapper.model == dummy_model, "Model not set correctly"
    assert wrapper.data.equals(data), "Data not set correctly"
    assert wrapper.parameters.alpha == 0.1, "Parameters not set correctly"
    assert wrapper.__len__ == 3, "Length not set correctly"


def test_wrapper_run():
    data = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1, 2, 3]})
    parameters = Parameters(alpha=Value(0.1))
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    wrapper.run()
    assert len(wrapper.simulation) == 3, "Output not generated correctly"
    assert np.allclose(
        wrapper.dependent, np.array([[0.1], [0.2], [0.3]])
    ), "Dependent not generated correctly"


def test_wrapper_reset():
    data = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1] * 3})
    parameters = Parameters(alpha=Value(0.1))
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    wrapper.run()
    wrapper.reset(parameters={"alpha": 0.2}, data=pd.DataFrame({"stimulus": [4, 5, 6]}))
    assert wrapper.parameters.alpha == 0.2
    assert len(wrapper.simulation) == 0
    assert wrapper.data.equals(pd.DataFrame({"stimulus": [4, 5, 6]}))


def test_wrapper_export():
    data = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1] * 3})
    parameters = Parameters(alpha=0.1)
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    wrapper.run()
    exported_data = wrapper.export()
    assert isinstance(exported_data, pd.DataFrame)
    assert len(exported_data) == 3


if __name__ == "__main__":
    pytest.main()
