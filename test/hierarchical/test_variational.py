import matplotlib

matplotlib.use("Agg")  # headless backend so diagnostics() cannot pop up a window

import warnings

import numpy as np
import pandas as pd
import pytest

from cpm.applications.reinforcement_learning import RLRW
from cpm.datasets import load_bandit_data
from cpm.hierarchical import VariationalBayes
from cpm.optimisation import FminBound, minimise

# VariationalBayes, like EmpiricalBayes, refits every participant on every
# iteration, so keep this tiny relative to the ~200 participants used in the
# bandit-task tutorial.
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
def vb(optimiser):
    warnings.simplefilter("ignore")  # default-hyperpriors warning
    return VariationalBayes(optimiser=optimiser, iteration=2, chain=1, quiet=True)


class TestInitialisation:
    def test_defaults(self, vb, optimiser):
        assert vb.iteration == 2
        assert vb.chain == 1
        assert vb.objective == "minimise"
        assert vb.__n_param__ == len(optimiser.model.parameters.free())
        assert vb.__n_ppt__ == N_PARTICIPANTS
        assert vb.__param_names__ == optimiser.model.parameters.free()
        assert isinstance(vb.hyperparameters, pd.DataFrame) and vb.hyperparameters.empty
        assert isinstance(vb.fit, pd.DataFrame) and vb.fit.empty

    def test_default_hyperpriors_warns_and_has_expected_shape(self, optimiser):
        with pytest.warns(UserWarning, match="default hyperpriors"):
            local_vb = VariationalBayes(optimiser=optimiser, iteration=2, chain=1, quiet=True)
        n_param = len(optimiser.model.parameters.free())
        assert np.asarray(local_vb.hyperpriors.a0).shape == (n_param,)
        assert local_vb.hyperpriors.b == 1
        assert local_vb.hyperpriors.v == 0.5
        assert np.asarray(local_vb.hyperpriors.s).shape == (n_param,)

    def test_custom_hyperpriors(self, optimiser):
        n_param = len(optimiser.model.parameters.free())
        custom = {
            "a0": np.zeros(n_param),
            "b": 2,
            "v": 1.0,
            "s": np.repeat(0.05, n_param),
        }
        local_vb = VariationalBayes(
            optimiser=optimiser, iteration=2, chain=1, quiet=True, hyperpriors=custom
        )
        assert local_vb.hyperpriors.b == 2
        assert local_vb.hyperpriors.v == 1.0

    def test_deep_copies_optimiser(self, vb, optimiser):
        # the algorithm mutates the optimiser's model priors, so it must not
        # be the same object the caller passed in
        assert vb.optimiser is not optimiser
        assert vb.function is not optimiser.model


class TestUpdateParticipants:
    def test_returns_params_log_posterior_and_hessian(self, vb):
        param, log_posterior, hessian = vb.update_participants(iter_idx=0, chain_idx=0)
        assert param.shape == (N_PARTICIPANTS, vb.__n_param__)
        assert log_posterior.shape == (N_PARTICIPANTS,)
        assert hessian.shape == (N_PARTICIPANTS, vb.__n_param__, vb.__n_param__)
        # the participant-wise fit history should have been recorded
        assert len(vb.fit) == N_PARTICIPANTS
        assert set(["ppt", "iteration", "chain"]).issubset(vb.fit.columns)


class TestGetLme:
    def test_returns_finite_vector_and_sum(self, vb):
        param, log_posterior, hessian = vb.update_participants(iter_idx=0, chain_idx=0)
        lme, lme_sum = vb.get_lme(log_post=log_posterior, hessian=hessian)
        assert lme.shape == (N_PARTICIPANTS,)
        assert np.isfinite(lme_sum)


class TestUpdatePopulation:
    def test_updates_priors_and_returns_snr(self, vb):
        param, log_posterior, hessian = vb.update_participants(iter_idx=0, chain_idx=0)
        _, lme_sum = vb.get_lme(log_post=log_posterior, hessian=hessian)
        population_updates, param_snr = vb.update_population(
            param=param, hessian=hessian, lme=lme_sum, iter_idx=0, chain_idx=0
        )
        parameter_names = set(vb.__param_names__)
        assert set(population_updates.keys()) == parameter_names
        for name in parameter_names:
            assert np.isfinite(population_updates[name]["mean"])
            assert population_updates[name]["sd"] > 0
        assert param_snr.shape == (vb.__n_param__,)

        # hyperparameters dataframe should have accumulated one row per parameter
        assert len(vb.hyperparameters) == len(parameter_names)
        for column in ["chain", "iteration", "parameter", "mean", "mean_se", "sd", "lme"]:
            assert column in vb.hyperparameters.columns


class TestCheckConvergence:
    def test_first_iteration_never_converges(self, vb):
        converged = vb.check_convergence(
            lme_new=-100.0, lme_old=np.nan, param_snr_new=np.array([1.0]), param_snr_old=np.nan, iter_idx=0
        )
        assert converged is False

    def test_converges_when_deltas_below_tolerance(self, vb):
        converged = vb.check_convergence(
            lme_new=-100.0,
            lme_old=-100.0 + vb.tolerance_lme / 10,
            param_snr_new=np.array([1.0, 2.0]),
            param_snr_old=np.array([1.0, 2.0]),
            iter_idx=1,
        )
        assert converged is True

    def test_does_not_converge_when_deltas_above_tolerance(self, vb):
        converged = vb.check_convergence(
            lme_new=-100.0,
            lme_old=-500.0,
            param_snr_new=np.array([1.0, 2.0]),
            param_snr_old=np.array([10.0, 20.0]),
            iter_idx=1,
        )
        assert converged is False


class TestOptimise:
    def test_single_chain_runs(self, vb):
        warnings.simplefilter("ignore")
        vb.optimise()

        assert len(vb.output) == 1
        result = vb.output[0]
        assert set(result.keys()) == {"lme", "hyperparameters", "parameters"}
        assert len(result["lme"]) <= vb.iteration

        parameter_names = set(vb.optimiser.model.parameters.free())
        assert set(vb.hyperparameters.parameter.unique()) == parameter_names
        for name in parameter_names:
            assert np.isfinite(result["hyperparameters"][name]["mean"])
            assert result["hyperparameters"][name]["sd"] > 0

    def test_multiple_chains_produce_independent_results(self, optimiser):
        warnings.simplefilter("ignore")
        multi = VariationalBayes(optimiser=optimiser, iteration=2, chain=2, quiet=True)
        multi.optimise()

        assert len(multi.output) == 2
        # chain indices are 1-based in VariationalBayes (see run_vb/optimise)
        assert set(multi.hyperparameters.chain.unique()) == {1, 2}
        assert set(multi.fit.chain.unique()) == {1, 2}


class TestTtest:
    def test_dict_null_matches_dataframe_null(self, vb):
        warnings.simplefilter("ignore")
        vb.optimise()

        null = {"alpha": 0.5, "temperature": 5}
        t_from_dict = vb.ttest(null=null)
        assert set(["parameter", "null", "t_stat", "p_val"]).issubset(t_from_dict.columns)
        assert np.isfinite(t_from_dict.t_stat).all()
        assert ((t_from_dict.p_val >= 0) & (t_from_dict.p_val <= 1)).all()

        null_df = pd.DataFrame(list(null.items()), columns=["parameter", "null"])
        t_from_df = vb.ttest(null=null_df)
        np.testing.assert_allclose(
            t_from_dict.t_stat.to_numpy(), t_from_df.t_stat.to_numpy()[: len(t_from_dict)]
        )
        # the caller's dataframe must not be mutated in place
        assert null_df.columns.tolist() == ["parameter", "null"]

    def test_raises_without_null(self, vb):
        warnings.simplefilter("ignore")
        vb.optimise()
        with pytest.raises(ValueError):
            vb.ttest(null=None)

    def test_raises_with_invalid_type(self, vb):
        warnings.simplefilter("ignore")
        vb.optimise()
        with pytest.raises(ValueError):
            vb.ttest(null=[0.5, 5])


class TestDiagnostics:
    def test_diagnostics_does_not_raise(self, vb):
        warnings.simplefilter("ignore")
        vb.optimise()
        vb.diagnostics(show=False, save=False)


if __name__ == "__main__":
    pytest.main()
