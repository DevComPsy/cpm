import pytest
import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from cpm.generators.parameters import Value, Parameters, LogParameters


def test_value_initialization():
    v = Value(value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1})
    assert v.value == 0.5
    assert v.lower == 0
    assert v.upper == 1
    assert isinstance(v.prior, scipy.stats._distn_infrastructure.rv_continuous_frozen)


def test_value_fill():
    v = Value(value=0.5)
    v.fill(0.8)
    assert v.value == 0.8


def test_value_pdf():
    v = Value(value=0.5, prior="norm", args={"mean": 0.5, "sd": 0.1})
    assert np.isclose(v.PDF(), norm(loc=0.5, scale=0.1).pdf(0.5))


def test_value_sample():
    v = Value(value=0.5, prior="norm", args={"mean": 0.5, "sd": 0.1})
    v.sample()
    assert v.value != 0.5


def test_value_update_prior():
    v = Value(value=0.5, prior="truncated_normal", args={"mean": 0.5, "sd": 0.1})
    v.update_prior(mean=0.6, sd=0.2)
    assert np.isclose(v.prior.kwds["loc"], 0.6)
    assert np.isclose(v.prior.kwds["scale"], 0.2)
    assert isinstance(v.prior, scipy.stats._distn_infrastructure.rv_continuous_frozen)


def test_value_math_operations():
    v = Value(value=2)
    assert v * 2 == 4
    assert v + 3 == 5
    assert v - 1 == 1
    assert v / 2 == 1
    assert v**2 == 4


def test_value_comparison_operations():
    v = Value(value=2)
    assert v == 2
    assert v != 3
    assert v < 3
    assert v > 1.0


def test_value_copy():
    v = Value(value=0.5)
    v_copy = v.copy()
    assert v.value == v_copy.value
    assert v is not v_copy


def test_parameters_initialization():
    params = Parameters(a=0.5, b=Value(value=0.3, lower=0, upper=1))
    assert params.a.value == 0.5
    assert params.b.value == 0.3


def test_parameters_getitem():
    params = Parameters(a=0.5, b=0.3)
    assert params["a"].value == 0.5
    assert params["b"].value == 0.3


def test_parameters_setitem():
    params = Parameters(a=0.5, b=0.3)
    params["a"] = Value(value=0.7)
    assert params.a.value == 0.7


def test_parameters_call():
    params = Parameters(a=0.5, b=0.3)
    assert params() == {"a": params.a, "b": params.b}


def test_parameters_export():
    params = Parameters(a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        ), b=0.3)
    result = params.export()
    assert isinstance(result, pd.DataFrame), "Export should return a pandas DataFrame."


def test_parameters_update():
    params = Parameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        ),
        b=0.3,
    )
    params.update(a=0.7)
    assert params.a.value == 0.7
    assert isinstance(params.a, Value)
    assert params.b.value == 0.3
    assert isinstance(
        params.a.prior, scipy.stats._distn_infrastructure.rv_continuous_frozen
    )


def test_parameters_bounds():
    params = Parameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        )
    )
    lower, upper = params.bounds()
    assert lower == [0]
    assert upper == [1]


def test_parameters_pdf():
    params = Parameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        )
    )
    assert np.isclose(params.PDF(), norm(loc=0.5, scale=0.1).pdf(0.5))


def test_parameters_update_prior():
    params = Parameters(
        a=Value(
            value=0.5,
            lower=0,
            upper=1,
            prior="truncated_normal",
            args={"mean": 0.5, "sd": 0.1},
        )
    )
    params.update_prior(a={"mean": 0.6, "sd": 0.2})
    assert np.isclose(params.a.prior.kwds["loc"], 0.6)
    assert np.isclose(params.a.prior.kwds["scale"], 0.2)


def test_parameters_sample():
    params = Parameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        )
    )
    sample = params.sample()
    assert "a" in sample[0]
    assert sample[0]["a"] != 0.5


def test_parameters_free():
    params = Parameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        ),
        b=0.3,
    )
    free_params = params.free()
    assert "a" in free_params
    assert "b" not in free_params


def test_log_parameters_initialization():
    params = LogParameters(
        a=0.5,
        b=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        ),
    )
    assert np.isclose(
        params.a.value, 0.5, atol=1e-1
    ), "LogParameters should log-transform only parameters with priors specified."
    assert np.isclose(
        params.b.value, 0, atol=1e-3
    ), "LogParameters log_transform is incorrect."


def test_log_parameters_log_inverse_transform():
    params = LogParameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        ),
        b=0.3,
    )
    a = params.log_inverse_transform()
    assert np.isclose(
        a.get("a"), 0.5, atol=1e-1
    ), "Inverse transform is incorrect for value with prior."
    assert len(a) == 1, "Inverse transform should only return parameters with priors."


def test_log_parameters_update():
    params = LogParameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        )
    )
    params.update(a=0.7, log=True)
    assert np.isclose(params.a.value, 0.847, atol=1e-3)  # logit(0.7)


def test_log_parameters_bounds():
    params = LogParameters(
        a=Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        )
    )
    lower, upper = params.bounds()
    assert np.isclose(lower[0], -23.0258, atol=1e-4)  # logit(1e-10)
    assert np.isclose(upper[0], 23.0258, atol=1e-4)  # logit(1 - 1e-10)


if __name__ == "__main__":
    pytest.main()
