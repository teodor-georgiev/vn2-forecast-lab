"""Regression checks for sweep geometry and causal policy construction."""
import unittest
import numpy as np
from vn2_from_scratch.ensemble_sweep import candidates, grouped_quantile, robust, HOLDOUT, TRAIN_ORIGINS

class SweepTests(unittest.TestCase):
    def test_grid_is_unique_and_contains_incumbent(self):
        grid = candidates()
        self.assertEqual(len(grid), 204)
        self.assertEqual(len(set(grid)), len(grid))
        self.assertIn(("point", .5, 1.1, .3, 0., 0.), grid)

    def test_group_quantile_increases_for_intermitttent_demand(self):
        q = grouped_quantile(.8, .05, np.array([False, True]))
        np.testing.assert_allclose(q, [.8, .85])

    def test_holdout_disjoint(self):
        self.assertTrue(all(o + 8 <= HOLDOUT[0] for o in TRAIN_ORIGINS))
        self.assertLessEqual(HOLDOUT[0] + 8, HOLDOUT[1])
        self.assertLessEqual(HOLDOUT[1] + 8, 157)

    def test_robust_penalizes_dispersion(self):
        self.assertGreater(robust([5, 15]), robust([10, 10]))

if __name__ == "__main__":
    unittest.main()
