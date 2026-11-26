import unittest
import numpy as np
from vn2_from_scratch.hierarchical_research import pool_mean, calibration, monotone_quantiles, SPECS, BASELINE

class HierarchyTests(unittest.TestCase):
    def test_pooling(self):
        np.testing.assert_allclose(pool_mean(np.array([1.,3.,8.,np.nan]),np.array([0,0,1,1])),[2,2,8,8])
    def test_calibration_shrinks_and_bounds(self):
        a=np.array([10.,0.,4.]); p=np.array([5.,1.,4.])
        group=np.array([0,0,1])
        v=calibration(a,p,group)
        self.assertTrue(np.all((v>=.8)&(v<=1.2)))
        self.assertEqual(v[0],v[1])
    def test_quantiles_do_not_cross(self):
        np.testing.assert_allclose(monotone_quantiles(np.array([[3.,2.],[-1.,4.]])),[[2,3],[0,4]])
    def test_incumbent_present_and_grid_unique(self):
        self.assertIn(BASELINE,SPECS)
        self.assertEqual(len(SPECS),len(set(SPECS)))
if __name__=="__main__":
    unittest.main()
