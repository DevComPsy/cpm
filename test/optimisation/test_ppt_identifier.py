import numpy as np
import pandas as pd
import pytest

from cpm.core.optimisers import group_identifier, prepare_data
from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import FminBound, minimise


def simple_model(parameters, trial):
    return {"dependent": np.array([trial["stimulus"] * parameters.alpha])}


def make_data():
    return pd.DataFrame(
        {
            "ppt": np.repeat([1, 2], 3),
            "stimulus": np.tile([1.0, 2.0, 3.0], 2),
            "observed": np.tile([0.1, 0.2, 0.3], 2),
        }
    )


def fit(data, **kwargs):
    parameters = Parameters(alpha=Value(value=0.5, lower=0.0, upper=1.0, prior="uniform"))
    model = Wrapper(model=simple_model, data=make_data().query("ppt == 1"), parameters=parameters)
    optimiser = FminBound(model=model, data=data, minimisation=minimise.Distance.SSE, approx_grad=True, **kwargs)
    optimiser.optimise(display=False)
    return optimiser


@pytest.mark.parametrize(
    "grouping, expected",
    [
        (lambda d: d.groupby("ppt"), "ppt"),
        (lambda d: d.groupby(["ppt"]), "ppt"),
        (lambda d: d.groupby(d["ppt"]), "ppt"),
        (lambda d: d.groupby(["ppt", "stimulus"]), None),
        (lambda d: d.groupby(d["ppt"].to_numpy()), None),
    ],
)
def test_group_identifier_reads_a_single_named_grouping(grouping, expected):
    assert group_identifier(grouping(make_data()), None) == expected


def test_group_identifier_keeps_a_given_identifier():
    assert group_identifier(make_data().groupby("ppt"), "participant") == "participant"
    assert group_identifier(make_data(), "ppt") == "ppt"
    assert group_identifier([{"observed": [1]}], None) is None


def test_prepare_data_needs_an_identifier_for_a_dataframe():
    with pytest.raises(ValueError, match="ppt_identifier"):
        prepare_data(make_data(), None)


def test_prepare_data_rejects_other_types():
    with pytest.raises(TypeError, match="ndarray"):
        prepare_data(np.zeros(3), None)


def test_fit_on_grouped_data_reports_the_participants():
    optimiser = fit(make_data().groupby("ppt"))
    assert optimiser.ppt_identifier == "ppt"
    assert list(optimiser.export()["ppt"]) == [1, 2]


def test_fit_on_a_dataframe_without_identifier_explains_the_error():
    with pytest.raises(ValueError, match="no ppt_identifier was given"):
        fit(make_data())
