import unittest
import numpy as np
import scipy
from scipy.stats import norm
from cpm.generators.parameters import Value, Parameters
from cpm.generators.parameters import LogParameters


class TestValue(unittest.TestCase):

    def test_initialization(self):
        v = Value(
            value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
        )
        self.assertEqual(v.value, 0.5)
        self.assertEqual(v.lower, 0)
        self.assertEqual(v.upper, 1)
        self.assertIsInstance(
            v.prior, scipy.stats._distn_infrastructure.rv_continuous_frozen
        ), "Prior not set correctly"

    def test_fill(self):
        v = Value(value=0.5)
        v.fill(0.8)
        self.assertEqual(v.value, 0.8)

    def test_pdf(self):
        v = Value(value=0.5, prior="norm", args={"mean": 0.5, "sd": 0.1})
        self.assertAlmostEqual(v.PDF(), norm(loc=0.5, scale=0.1).pdf(0.5))

    def test_sample(self):
        v = Value(value=0.5, prior="norm", args={"mean": 0.5, "sd": 0.1})
        v.sample()
        self.assertNotEqual(v.value, 0.5)

    def test_update_prior(self):
        v = Value(value=0.5, prior="truncated_normal", args={"mean": 0.5, "sd": 0.1})
        v.update_prior(mean=0.6, sd=0.2)
        self.assertAlmostEqual(v.prior.kwds["loc"], 0.6)
        self.assertAlmostEqual(v.prior.kwds["scale"], 0.2)
        self.assertIsInstance(
            v.prior, scipy.stats._distn_infrastructure.rv_continuous_frozen
        ), "Prior updates change prior type()"

    def test_math_operations(self):
        v = Value(value=2)
        self.assertEqual(v * 2, 4)
        self.assertEqual(v + 3, 5)
        self.assertEqual(v - 1, 1)
        self.assertEqual(v / 2, 1)
        self.assertEqual(v**2, 4)

    def test_comparison_operations(self):
        v = Value(value=2)
        self.assertTrue(v == 2)
        self.assertTrue(v != 3)
        self.assertTrue(v < 3)
        self.assertTrue(v > 1.0)

    def test_copy(self):
        v = Value(value=0.5)
        v_copy = v.copy()
        self.assertEqual(v.value, v_copy.value)
        self.assertIsNot(v, v_copy)


class TestParameters(unittest.TestCase):

    def test_initialization(self):
        params = Parameters(a=0.5, b=Value(value=0.3, lower=0, upper=1))
        self.assertEqual(params.a.value, 0.5)
        self.assertEqual(params.b.value, 0.3)

    def test_getitem(self):
        params = Parameters(a=0.5, b=0.3)
        self.assertEqual(params["a"].value, 0.5)
        self.assertEqual(params["b"].value, 0.3)

    def test_setitem(self):
        params = Parameters(a=0.5, b=0.3)
        params["a"] = Value(value=0.7)
        self.assertEqual(params.a.value, 0.7)

    def test_call(self):
        params = Parameters(a=0.5, b=0.3)
        self.assertEqual(params(), {"a": params.a, "b": params.b})

    def test_export(self):
        params = Parameters(a=0.5, b=0.3)
        self.assertEqual(params.export(), {"a": params.a, "b": params.b})

    def test_update(self):
        params = Parameters(
            a=Value(
                value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
            ),
            b=0.3,
        )
        params.update(a=0.7)
        self.assertEqual(params.a.value, 0.7)
        self.assertIsInstance(params.a, Value)
        self.assertEqual(params.b.value, 0.3)
        self.assertIsInstance(
            params.a.prior, scipy.stats._distn_infrastructure.rv_continuous_frozen
        ), "Updating parameter changed priro type()"

    def test_bounds(self):
        params = Parameters(
            a=Value(
                value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
            )
        )
        lower, upper = params.bounds()
        self.assertEqual(lower, [0])
        self.assertEqual(upper, [1])

    def test_pdf(self):
        params = Parameters(
            a=Value(
                value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
            )
        )
        self.assertAlmostEqual(params.PDF(), norm(loc=0.5, scale=0.1).pdf(0.5))

    def test_update_prior(self):
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
        self.assertAlmostEqual(params.a.prior.kwds["loc"], 0.6)
        self.assertAlmostEqual(params.a.prior.kwds["scale"], 0.2)

    def test_sample(self):
        params = Parameters(
            a=Value(
                value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
            )
        )
        sample = params.sample()
        self.assertIn("a", sample[0])
        self.assertNotEqual(sample[0]["a"], 0.5)

    def test_free(self):
        params = Parameters(
            a=Value(
                value=0.5, lower=0, upper=1, prior="norm", args={"mean": 0.5, "sd": 0.1}
            ),
            b=0.3,
        )
        free_params = params.free()
        self.assertIn("a", free_params)
        self.assertNotIn("b", free_params)


if __name__ == "__main__":
    unittest.main()
