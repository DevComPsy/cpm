import pytest
import numpy as np
import pandas as pd
from cpm.generators import Wrapper
from cpm.generators import Parameters, Value


def dummy_model(parameters, trial):
    tmp = np.array([trial["stimulus"] * parameters.alpha])
    return {"dependent": tmp}


def test_wrapper_initialization():
    data = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1, 2, 3], "observed": [0.1, 0.2, 0.3]})
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


def test_wrapper_reset_array_ignores_declaration_order():
    data = pd.DataFrame({"stimulus": [1, 2, 3], "observed": [0.1, 0.2, 0.3]})
    parameters = Parameters(
        values=np.array([0.5, 0.5]),
        alpha=Value(value=0.1, lower=0, upper=1, prior="uniform"),
        beta=Value(value=5.0, lower=0, upper=10, prior="uniform"),
    )
    wrapper = Wrapper(model=dummy_model, data=data, parameters=parameters)
    wrapper.reset(parameters=np.array([0.42, 7.0]))
    assert wrapper.parameters.alpha == 0.42
    assert wrapper.parameters.beta == 7.0
    assert np.allclose(wrapper.parameters.values.value, [0.5, 0.5])


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

def dummy_loss(predicted, observed):
    # Simple loss: sum of squared differences
    return np.sum((predicted - observed) ** 2)

def test_connector_returns_callable_and_computes_loss():
    # Create dummy data
    df = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1, 2, 3], "observed": [0.1, 0.2, 0.3]})
    parameters = Parameters(alpha=Value(0.1))

    # Create wrapper
    wrapper = Wrapper(model=dummy_model, data=df, parameters=parameters)

    # Get parser from connector
    parser = wrapper.connector(loss=dummy_loss)

    # Check that parser is callable
    assert callable(parser)

    # Call parser with parameters
    result = parser([2])

    # Should return a float/int
    assert isinstance(result, (float, int, np.floating, np.integer))

def test_connector_raises_without_loss():
    df = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1, 2, 3], "observed": [0.1, 0.2, 0.3]})
    parameters = Parameters(alpha=Value(0.1))
    wrapper = Wrapper(model=dummy_model, data=df, parameters=parameters)
    with pytest.raises(ValueError):
        wrapper.connector()

def test_run_warning_without_observed():
    df = pd.DataFrame({"stimulus": [1, 2, 3], "step": [1, 2, 3]})  # No 'observed' column
    parameters = Parameters(alpha=Value(0.1))
    with pytest.warns(UserWarning, match="The data does not contain an 'observed' column."):
        wrapper = Wrapper(model=dummy_model, data=df, parameters=parameters)
        wrapper.run()
    with pytest.raises(ValueError, match="Data must contain an 'observed' column to compute loss."):
        wrapper = Wrapper(model=dummy_model, data=df, parameters=parameters)
        wrapper.connector(loss=dummy_loss)


## ---------------------------------------------------------------------------
## the rows Wrapper.run hands to the model (cpm.core.data.trial_reader)

from cpm.core.data import trial_reader, unpack_trials


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": [0.5, 1.5], "b": [2.0, 3.0]}),
        pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]}, index=[10, 20]),
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        pd.DataFrame({"a": [True, False], "b": [3, 4]}),
        pd.DataFrame({"a": np.array([1, 2], dtype=np.uint8), "b": [3, 4]}),
        pd.DataFrame({"a": [1.0, 2.0]}),
        pd.DataFrame({"a": [1, 2], "label": ["x", "y"]}),
        pd.DataFrame({"a": pd.array([1, None], dtype="Int64"), "b": [1.0, 2.0]}),
    ],
    ids=["float", "int+float", "int", "bool+int", "uint8", "one column", "strings", "nullable"],
)
def test_trial_reader_gives_the_rows_iloc_gives(frame):
    read = trial_reader(frame, pandas=True)
    for i in range(len(frame)):
        expected, got = unpack_trials(frame, i, True), read(i)
        if isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(got, expected, check_exact=True)
        else:
            assert type(got) is type(expected) and got == expected


def test_trials_changed_by_a_model_do_not_change_the_data():
    data = pd.DataFrame({"stimulus": [1.0, 2.0, 3.0], "observed": [0.0, 0.0, 0.0]})

    def meddling(parameters, trial):
        trial["stimulus"] = 100.0
        return {"dependent": np.array([trial["stimulus"]])}

    wrapper = Wrapper(model=meddling, data=data, parameters=Parameters(alpha=0.1))
    wrapper.run()
    assert data.stimulus.tolist() == [1.0, 2.0, 3.0]


def test_runs_read_the_data_afresh():
    data = pd.DataFrame({"stimulus": [1.0, 2.0, 3.0], "observed": [0.0, 0.0, 0.0]})
    wrapper = Wrapper(model=dummy_model, data=data, parameters=Parameters(alpha=Value(1.0)))
    wrapper.run()
    wrapper.data["stimulus"] = [4.0, 5.0, 6.0]
    wrapper.reset()
    wrapper.run()
    np.testing.assert_array_equal(wrapper.dependent.ravel(), [4.0, 5.0, 6.0])


if __name__ == "__main__":
    pytest.main()
