import matplotlib

matplotlib.use("Agg")  # headless backend so diagnostics() cannot pop up a window

import warnings

import numpy as np
import pandas as pd
import pytest

from cpm.applications.reinforcement_learning import RLRW
from cpm.datasets import load_bandit_data
from cpm.hierarchical import EmpiricalBayes
from cpm.optimisation import FminBound, minimise

# EmpiricalBayes fits every participant on every EM iteration, so keep this
# tiny relative to the ~200 participants used in the bandit-task tutorial.
N_PARTICIPANTS = 3


@pytest.fixture
def bandit_data():
    data = load_bandit_data()
    data["observed"] = data["response"]
    ppts = sorted(data.ppt.unique())[:N_PARTICIPANTS]
    return data[data.ppt.isin(ppts)].copy()


@pytest.fixture
def optimiser(bandit_data):
    warnings.simplefilter("ignore")
    wrapper = RLRW(data=bandit_data[bandit_data.ppt == bandit_data.ppt.iloc[0]], dimensions=4)
    return FminBound(
        model=wrapper,
        data=bandit_data,
        minimisation=minimise.LogLikelihood.bernoulli,
        parallel=False,
        prior=True,
        ppt_identifier="ppt",
        display=False,
        number_of_starts=1,
        maxiter=25,
        approx_grad=True,
    )


@pytest.fixture
def eb(optimiser):
    return EmpiricalBayes(optimiser=optimiser, iteration=2, tolerance=1e-6, chain=1, quiet=True)


class TestInitialisation:
    def test_defaults(self, eb, optimiser):
        assert eb.iteration == 2
        assert eb.chain == 1
        assert eb.objective == "minimise"
        assert eb.__number_of_parameters__ == len(optimiser.model.parameters.free())
        assert isinstance(eb.hyperparameters, pd.DataFrame) and eb.hyperparameters.empty
        assert isinstance(eb.fit, pd.DataFrame) and eb.fit.empty

    def test_deep_copies_optimiser(self, eb, optimiser):
        # the EM loop mutates the optimiser's model priors, so it must not be
        # the same object the caller passed in
        assert eb.optimiser is not optimiser
        assert eb.function is not optimiser.model


class TestStep:
    def test_returns_parameters_hessian_and_fit_details(self, eb):
        parameters, hessian, details = eb.step()
        assert len(parameters) == N_PARTICIPANTS
        assert hessian.shape[0] == N_PARTICIPANTS
        # square hessian per participant, one row/col per free parameter
        n_free = eb.__number_of_parameters__
        assert hessian.shape[1:] == (n_free, n_free)
        assert len(details) == N_PARTICIPANTS
        assert all("hessian" in d for d in details)


class TestOptimise:
    def test_single_chain_runs(self, eb):
        warnings.simplefilter("ignore")
        eb.optimise()

        assert len(eb.output) == 1
        result = eb.output[0]
        assert set(result.keys()) == {"lme", "hyperparameters", "parameters"}
        assert len(result["lme"]) <= eb.iteration

        # hyperparameters track for each free parameter across iterations
        parameter_names = set(eb.optimiser.model.parameters.free())
        assert set(eb.hyperparameters.parameter.unique()) == parameter_names
        for column in ["chain", "iteration", "parameter", "mean", "sd", "lme", "reject"]:
            assert column in eb.hyperparameters.columns

        # the group-level hyperparameters returned from the last iteration
        # should match what update_prior actually applied
        for name in parameter_names:
            assert name in result["hyperparameters"]
            assert "mean" in result["hyperparameters"][name]
            assert "sd" in result["hyperparameters"][name]
            assert np.isfinite(result["hyperparameters"][name]["mean"])
            assert np.isfinite(result["hyperparameters"][name]["sd"])
            assert result["hyperparameters"][name]["sd"] > 0

    def test_multiple_chains_produce_independent_results(self, optimiser):
        warnings.simplefilter("ignore")
        multi = EmpiricalBayes(optimiser=optimiser, iteration=2, chain=2, quiet=True)
        multi.optimise()

        assert len(multi.output) == 2
        assert set(multi.hyperparameters.chain.unique()) == {0, 1}
        # each chain (after the first) starts from a different random prior
        assert set(multi.fit.chain.unique()) == {0, 1}

    def test_parameters_method_returns_fit_history(self, eb):
        warnings.simplefilter("ignore")
        eb.optimise()
        fit = eb.parameters()
        assert fit is eb.fit
        assert set(["ppt", "iteration", "chain"]).issubset(fit.columns)
        assert set(fit.ppt.unique()) == set(range(N_PARTICIPANTS))


class TestDiagnostics:
    def test_diagnostics_does_not_raise(self, eb):
        warnings.simplefilter("ignore")
        eb.optimise()
        eb.diagnostics(show=False, save=False)


if __name__ == "__main__":
    pytest.main()
