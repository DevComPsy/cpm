import numpy as np
from scipy.stats import norm

from cpm.generators import Parameters, Value
from cpm.hierarchical._chains import starting_priors

SIGNED = (np.array([0.0, -5.0]), np.array([1.0, 5.0]))


def draw(names, bounds, priors, seed=2026):
    np.random.seed(seed)
    return starting_priors(names, bounds, priors)


class TestStartingPriors:
    def test_seed_reproduces_the_draws(self):
        assert draw(["a", "b"], SIGNED, {}) == draw(["a", "b"], SIGNED, {})

    def test_draws_come_from_the_global_generator(self):
        np.random.seed(2026)
        first = starting_priors(["a", "b"], SIGNED, {})
        second = starting_priors(["a", "b"], SIGNED, {})
        assert first != second
        assert draw(["a", "b"], SIGNED, {}) == first

    def test_means_cover_signed_bounds(self):
        bounds = (np.array([-0.49]), np.array([0.49]))
        np.random.seed(0)
        draws = [starting_priors(["eta"], bounds, {})["eta"] for _ in range(2000)]
        means = np.array([d["mean"] for d in draws])
        sds = np.array([d["sd"] for d in draws])
        assert np.all((means >= -0.49) & (means <= 0.49))
        assert means.min() < -0.4 and means.max() > 0.4, "both halves of the range are explored"
        assert np.all((sds > 0) & (sds <= 0.49))

    def test_bounds_from_zero_keep_the_previous_draws(self):
        np.random.seed(3)
        updates = starting_priors(["x"], (np.array([0.0]), np.array([10.0])), {})
        np.random.seed(3)
        mean = np.random.beta(a=2, b=2) * 10.0
        sd = np.random.beta(a=2, b=2) * (10.0 / 2)
        assert updates["x"] == {"mean": mean, "sd": sd}

    def test_infinite_bounds_draw_from_the_starting_prior(self):
        prior = norm(loc=1.5, scale=2.0)
        updates = draw(["x"], (np.array([-np.inf]), np.array([np.inf])), {"x": prior}, seed=4)
        np.random.seed(4)
        assert updates["x"] == {"mean": prior.rvs(), "sd": 2.0}

    def test_half_infinite_bounds_stay_within_the_finite_bound(self):
        parameters = Parameters(
            x=Value(value=1.0, lower=0, upper=np.inf, prior="truncated_normal", args={"mean": 1.0, "sd": 2.0})
        )
        priors = {"x": parameters.x.prior}
        np.random.seed(5)
        draws = [starting_priors(["x"], parameters.bounds(), priors)["x"] for _ in range(500)]
        assert all(d["mean"] >= 0 and np.isfinite(d["mean"]) for d in draws)
        assert all(0 < d["sd"] < np.inf for d in draws)

    def test_update_prior_accepts_the_draws(self):
        parameters = Parameters(
            eta=Value(value=0.0, lower=-0.49, upper=0.49, prior="truncated_normal", args={"mean": 0.0, "sd": 0.2}),
            x=Value(value=0.0, lower=-np.inf, upper=np.inf, prior="norm", args={"mean": 0.0, "sd": 1.0}),
        )
        names = parameters.free()
        priors = {name: getattr(parameters, name).prior for name in names}
        parameters.update_prior(**draw(names, parameters.bounds(), priors))
        assert np.isfinite(parameters.PDF(log=True))
