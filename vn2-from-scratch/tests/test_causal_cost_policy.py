"""Tests for the strict pre-demand cost-aware ordering contract."""
import unittest
import numpy as np
from vn2_from_scratch.causal_cost_policy import project_order, causal_rollout

class CausalCostTests(unittest.TestCase):
    def test_both_committed_arrivals_and_forecast_weeks(self):
        pred=np.array([[4., 7., 8.]])
        # 3+5-4=4; 4+2-7 -> 0; target=8.
        np.testing.assert_array_equal(project_order(np.array([3.]),np.array([5.]),np.array([2.]),pred,np.array([1.]),0.,1.),[8.])

    def test_multiplier_scales_entire_quantile_target(self):
        pred=np.array([[4.,7.,8.]])
        np.testing.assert_array_equal(project_order(np.array([3.]),np.array([5.]),np.array([2.]),pred,np.array([2.]),1.,1.5),[15.])

    def test_no_negative_orders(self):
        pred=np.array([[1.,1.,1.]])
        np.testing.assert_array_equal(project_order(np.array([30.]),np.array([0.]),np.array([0.]),pred,np.array([1.]),0.,1.),[0.])

    def test_current_and_future_demand_never_changes_decision_prefix(self):
        demand=np.arange(16,dtype=float).reshape(8,2)%5
        opening=[np.array([1.,2.]),np.array([2.,1.]),np.array([3.,2.])]
        forecasts=np.full((6,2,3),2.)
        def decide(t,oh,a0,a1,history):
            self.assertEqual(len(history),t)
            pred=forecasts[t].copy()
            if t: pred+=history.mean(axis=0)[:,None]*.1
            return project_order(oh,a0,a1,pred,np.ones(2),.4,1.)
        orders=causal_rollout(demand,opening,decide)
        for t in range(6):
            changed=demand.copy()
            changed[t:]+=100
            np.testing.assert_array_equal(causal_rollout(changed,opening,decide)[:t+1],orders[:t+1])

    def test_round_one_arrives_in_week_three(self):
        demand=np.zeros((8,1))
        opening=[np.zeros(1) for _ in range(3)]
        seen=[]
        def decide(t,oh,a0,a1,history):
            seen.append((float(a0[0]),float(a1[0])))
            return np.array([10. if t==0 else 0.])
        causal_rollout(demand,opening,decide)
        self.assertEqual(seen[:3],[(0.,0.),(0.,10.),(10.,0.)])

if __name__=="__main__":
    unittest.main()
