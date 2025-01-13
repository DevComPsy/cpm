import unittest
import pandas as pd
from cpm.optimisation.fmin import Fmin
from cpm.generators import Wrapper, Parameters, Value
from cpm.applications.reinforcement_learning import RLRW


class TestFmin(unittest.TestCase):

    def setUp(self):

        # Create an instance of Fmin
        self.fmin = Fmin(model=self.model, data=self.data, ppt_identifier="participant")

    def test_initialization(self):
        self.assertIsInstance(self.fmin, Fmin)
        self.assertEqual(self.fmin.ppt_identifier, "participant")
        self.assertEqual(self.fmin.number_of_starts, 1)

    def test_optimise(self):
        self.fmin.optimise()
        self.assertGreater(len(self.fmin.fit), 0)
        self.assertGreater(len(self.fmin.details), 0)
        self.assertGreater(len(self.fmin.parameters), 0)

    def test_reset(self):
        self.fmin.optimise()
        self.fmin.reset()
        self.assertEqual(len(self.fmin.fit), 0)
        self.assertEqual(len(self.fmin.details), 0)
        self.assertEqual(len(self.fmin.parameters), 0)

    def test_export(self):
        self.fmin.optimise()
        df = self.fmin.export()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


class TestFminParallel(unittest.TestCase):

    def setUp(self):

        # Create an instance of Fmin
        self.fmin = Fmin(
            model=self.model,
            data=self.data,
            ppt_identifier="participant",
            parallel=True,
        )

    def test_initialization(self):
        self.assertIsInstance(self.fmin, Fmin)
        self.assertEqual(self.fmin.ppt_identifier, "participant")
        self.assertEqual(self.fmin.number_of_starts, 1)

    def test_optimise(self):
        self.fmin.optimise()
        self.assertGreater(len(self.fmin.fit), 0)
        self.assertGreater(len(self.fmin.details), 0)
        self.assertGreater(len(self.fmin.parameters), 0)

    def test_reset(self):
        self.fmin.optimise()
        self.fmin.reset()
        self.assertEqual(len(self.fmin.fit), 0)
        self.assertEqual(len(self.fmin.details), 0)
        self.assertEqual(len(self.fmin.parameters), 0)

    def test_export(self):
        self.fmin.optimise()
        df = self.fmin.export()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
