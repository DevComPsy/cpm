"""SciPy 1.18.0 removed the `disp` and `iprint` options of the L-BFGS-B solver.

`FminBound` used to forward `disp=self.display` to `scipy.optimize.fmin_l_bfgs_b`
unconditionally, so every fit raised `TypeError: fmin_l_bfgs_b() got an
unexpected keyword argument 'disp'` on SciPy 1.18.0 and later, and emitted a
`DeprecationWarning` per participant on SciPy 1.15.0 to 1.17.x.
"""

import numpy as np
import pandas as pd
import pytest

import cpm.optimisation.fmin as fmin_module
from cpm.generators import Parameters, Value, Wrapper
from cpm.optimisation import FminBound, minimise


def simple_model(parameters, trial):
    return {"dependent": np.array([trial["stimulus"] * parameters.alpha])}


def make_data(n_ppt=2, n_trials=3):
    return pd.DataFrame(
        {
            "ppt": np.repeat(np.arange(1, n_ppt + 1), n_trials),
            "stimulus": np.tile(np.arange(1, n_trials + 1), n_ppt) * 1.0,
            "observed": np.tile(np.linspace(0.1, 0.3, n_trials), n_ppt),
        }
    )


def make_wrapper(data):
    parameters = Parameters(
        alpha=Value(
            value=0.5,
            lower=0.0,
            upper=1.0,
            prior="norm",
            args={"mean": 0.5, "sd": 0.25},
        )
    )
    return Wrapper(model=simple_model, data=data[data.ppt == 1], parameters=parameters)


class TestLbfgsbOptions:
    """`lbfgsb_options` decides what may reach the solver."""

    def test_supported_scipy_forwards_display(self, monkeypatch):
        monkeypatch.setattr(fmin_module, "LBFGSB_VERBOSITY_SUPPORTED", True)
        assert fmin_module.lbfgsb_options(display=True) == {"disp": True}

    def test_supported_scipy_omits_disp_when_display_is_off(self, monkeypatch):
        """The default must not pass `disp` at all: SciPy 1.15.0 to 1.17.x warn
        about it even when it is switched off, once per participant."""
        monkeypatch.setattr(fmin_module, "LBFGSB_VERBOSITY_SUPPORTED", True)
        assert fmin_module.lbfgsb_options(display=False, approx_grad=True) == {
            "approx_grad": True
        }

    def test_supported_scipy_keeps_explicit_options(self, monkeypatch):
        monkeypatch.setattr(fmin_module, "LBFGSB_VERBOSITY_SUPPORTED", True)
        options = fmin_module.lbfgsb_options(display=False, iprint=1, disp=True)
        assert options == {"iprint": 1, "disp": True}

    def test_unsupported_scipy_drops_verbosity_options(self, monkeypatch):
        monkeypatch.setattr(fmin_module, "LBFGSB_VERBOSITY_SUPPORTED", False)
        with pytest.warns(RuntimeWarning, match="disp, iprint"):
            options = fmin_module.lbfgsb_options(
                display=True, iprint=1, approx_grad=True, maxiter=200
            )
        assert options == {"approx_grad": True, "maxiter": 200}

    def test_unsupported_scipy_is_quiet_when_nothing_is_dropped(self, monkeypatch):
        monkeypatch.setattr(fmin_module, "LBFGSB_VERBOSITY_SUPPORTED", False)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fmin_module.lbfgsb_options(display=False, maxiter=200) == {
                "maxiter": 200
            }


class TestFminBoundOnScipyWithoutVerbosity:
    """The fit itself must survive a SciPy that rejects the removed options."""

    @pytest.fixture
    def scipy_118(self, monkeypatch):
        real = fmin_module.fmin_l_bfgs_b

        def strict_fmin_l_bfgs_b(*args, **kwargs):
            for removed in ("disp", "iprint"):
                if removed in kwargs:
                    raise TypeError(
                        "fmin_l_bfgs_b() got an unexpected keyword argument "
                        f"'{removed}'"
                    )
            return real(*args, **kwargs)

        monkeypatch.setattr(fmin_module, "fmin_l_bfgs_b", strict_fmin_l_bfgs_b)
        monkeypatch.setattr(fmin_module, "LBFGSB_VERBOSITY_SUPPORTED", False)

    @pytest.mark.parametrize("display", [False, True], ids=["quiet", "display"])
    def test_optimise_completes(self, scipy_118, display, capsys):
        data = make_data()
        optimiser = FminBound(
            model=make_wrapper(data),
            data=data,
            minimisation=minimise.LogLikelihood.continuous,
            ppt_identifier="ppt",
            display=display,
            approx_grad=True,
        )

        if display:
            with pytest.warns(RuntimeWarning, match="disp"):
                optimiser.optimise()
        else:
            optimiser.optimise()

        assert len(optimiser.parameters) == 2
        assert all(np.isfinite(fit["fun"]) for fit in optimiser.fit)

    def test_display_still_reports_the_multistart_loop(self, scipy_118, capsys):
        """Dropping the solver's own verbosity must not silence cpm's."""
        data = make_data()
        optimiser = FminBound(
            model=make_wrapper(data),
            data=data,
            minimisation=minimise.LogLikelihood.continuous,
            ppt_identifier="ppt",
            display=True,
            approx_grad=True,
        )
        with pytest.warns(RuntimeWarning):
            optimiser.optimise()
        assert "Starting optimization 1/1" in capsys.readouterr().out
