"""Order placement before revealing current-week demand; two committed arrivals."""
from __future__ import annotations
import numpy as np
from .simulator import pipeline_at

def project_order(on_hand, arrivals_now, arrivals_next, forecast, sigma, z, multiplier):
    """Project through two pre-arrival weeks, then cover arrival-week demand.

    Forecast columns are single-week horizons 1, 2 and 3, not cumulative.
    Lost sales in either pre-arrival week cannot become negative inventory.
    """
    fc=np.maximum(np.asarray(forecast,float),0)
    end1=np.maximum(0,on_hand+arrivals_now-fc[:,0])
    end2=np.maximum(0,end1+arrivals_next-fc[:,1])
    target=np.maximum(0,multiplier*(fc[:,2]+z*np.maximum(sigma,0)))
    return np.maximum(0,np.rint(target-end2))

def causal_rollout(demand, opening, decide):
    """Policy sees only revealed history, on-hand, and committed arrivals."""
    demand=np.asarray(demand,float)
    if demand.ndim!=2 or demand.shape[0]!=8:
        raise ValueError("expected eight demand weeks")
    oh,p1,p2=[np.asarray(v,float).copy() for v in opening]
    orders=np.zeros((6,demand.shape[1]))
    for t in range(6):
        a0,a1=pipeline_at(orders,t+1,p1,p2)
        q=np.asarray(decide(t,oh.copy(),a0.copy(),a1.copy(),demand[:t].copy()),float)
        if q.shape!=oh.shape or not np.isfinite(q).all() or (q<0).any():
            raise ValueError("invalid order")
        orders[t]=np.rint(q)
        oh=np.maximum(0,oh+a0-demand[t])
    return orders
