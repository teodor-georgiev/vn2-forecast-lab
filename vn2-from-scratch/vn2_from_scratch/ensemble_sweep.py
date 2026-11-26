"""Chronological CatBoost/seasonal/empirical replenishment sweep.

Run: python -m vn2_from_scratch.ensemble_sweep --out-dir <unique directory>
No selection uses the replay or the two reserved late historical windows.
Historical stress multiplies sales for the simulator, not forecast inputs,
matching the previous benchmark's frozen forecast stress convention.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from .config import CV_ORIGINS, CV_INITIAL_STATE_SEEDS, DEMAND_STRESS
from .data import load_dataset
from .organizer import full_panel, run_organizer_benchmark
from .simulator import simulate, pipeline_at
from .verify import independent_simulate
from .progress import Progress

TRAIN_ORIGINS = tuple(CV_ORIGINS)
HOLDOUT = (135, 144)
INCUMBENT = ("point", .5, 1.1, .3, 0., 0.)

def candidates():
    point = [("point", w, c, b, 0., 0.) for w,c,b in itertools.product(
        (0.,.25,.5,.75,1.), (1.,1.1,1.2,1.3), (0.,.3,.6))]
    hybrid = [("hybrid", w, a, shift, mix, b) for w,a,shift,mix,b in itertools.product(
        (.25,.5,.75,1.), (.7,.8,.9), (-.05,0.,.05), (.5,1.), (0.,.3))]
    return point + hybrid

def robust(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()+.45*a.std()+.15*(a.max()-a.mean()))

def grouped_quantile(alpha, shift, intermittent):
    return np.clip(alpha+shift*np.asarray(intermittent,dtype=float),.05,.99)

def mean(x):
    count = np.isfinite(x).sum(axis=1)
    return np.divide(np.nansum(x,axis=1), count, out=np.zeros(len(x)), where=count>0)

class Forecasts:
    def __init__(self,c,woy):
        self.c,self.woy=c,woy
        self.n=len(c)
        self.seasons={}
        self.features={}
        self.seasonals={}

    def season(self,t):
        if t not in self.seasons:
            col=np.nanmean(self.c[:,:t],axis=0)
            s=np.ones(54)
            seen=np.unique(self.woy[:t])
            for w in seen:
                v=col[self.woy[:t]==w]
                s[w]=np.nanmean(v) if np.isfinite(v).any() else 1.
            self.seasons[t]=np.maximum(s/max(s[seen].mean(),1e-8),.05)
        return self.seasons[t]

    def seasonal(self,t):
        if t not in self.seasonals:
            s=self.season(t); idx=np.arange(max(0,t-13),t)
            self.seasonals[t]=mean(self.c[:,idx]/s[self.woy[idx]])[:,None]*s[self.woy[t:t+3]][None,:]
        return self.seasonals[t]

    def feature(self,t,h):
        key=(t,h)
        if key not in self.features:
            x=[]
            for w in (4,8,13,26,52):
                b=self.c[:,max(0,t-w):t]
                x.extend([mean(b),np.nan_to_num(np.nanstd(b,axis=1)),np.isfinite(b).mean(axis=1),mean(np.where(np.isfinite(b),b==0,np.nan))])
            for lag in (1,2,3,4,13,26,52):
                x.append(np.nan_to_num(self.c[:,t-lag],nan=-1.) if t>=lag else np.full(self.n,-1.))
            x.extend([self.seasonal(t)[:,h-1],np.full(self.n,h),np.full(self.n,np.sin(2*np.pi*self.woy[t+h-1]/52)),np.full(self.n,np.cos(2*np.pi*self.woy[t+h-1]/52))])
            self.features[key]=np.column_stack(x).astype(np.float32)
        return self.features[key]

    def fit(self,o,progress,fold,done,total):
        xx=[]; yy=[]
        for t in range(26,o):
            for h in (1,2,3):
                if t+h-1>=o: continue
                target=self.c[:,t+h-1]; valid=np.isfinite(target)
                xx.append(self.feature(t,h)[valid]); yy.append(target[valid])
        model=CatBoostRegressor(iterations=100,depth=6,learning_rate=.08,loss_function="RMSE",
            thread_count=4,verbose=False,random_seed=42,allow_writing_files=False)
        class Callback:
            def after_iteration(self,info):
                if info.iteration % 10 == 0:
                    progress.update("training",info.iteration,100,model="CatBoost",origin=o,
                        completed_folds=done,total_folds=total,fold=fold)
                return True
        progress.update("training",0,100,model="CatBoost",origin=o,completed_folds=done,total_folds=total,fold=fold)
        model.fit(np.concatenate(xx),np.concatenate(yy),callbacks=[Callback()])
        return model

    def path_quantiles(self,t):
        # Causal empirical 3-week total distribution; normalize by its own mean.
        # Complete joint paths preserve week-to-week dependence.
        s=self.season(t); a=np.arange(t-52,t)
        hist=self.c[:,a]/s[self.woy[a]][None,:]
        observed=np.isfinite(hist)
        intermittent=mean(np.where(observed,hist==0,np.nan))>=.35
        paths=np.stack([hist[:,k:k+3] for k in range(50)],axis=1)*s[self.woy[t:t+3]][None,None,:]
        for i in range(self.n):
            if np.isfinite(paths[i]).all(axis=1).sum()<5:
                vals=hist[i,observed[i]]
                if len(vals):
                    k=np.arange(50)
                    paths[i]=np.column_stack([vals[k%len(vals)],vals[(7*k+3)%len(vals)],vals[(11*k+5)%len(vals)]])*s[self.woy[t:t+3]]
                else: paths[i]=0.
        totals=paths.sum(axis=2)
        avg=mean(totals)
        ratios={}
        for q in sorted({round(a+d,2) for a in (.7,.8,.9) for d in (-.05,0.,.05)}):
            value=np.nanquantile(totals,q,axis=1)
            ratios[q]=np.divide(value,avg,out=np.ones(self.n),where=avg>0)
        return intermittent,ratios

    def prepare(self,o,progress,fold,done,total):
        model=self.fit(o,progress,fold,done,total)
        out=[]
        for t in range(o,o+6):
            cat=np.column_stack([np.maximum(model.predict(self.feature(t,h)),0.) for h in (1,2,3)])
            out.append((self.seasonal(t).sum(axis=1),cat.sum(axis=1),*self.path_quantiles(t)))
        return out

def targets(prepared,spec):
    kind,w,a,b,mix,buf=spec
    result=[]
    for seasonal,cat,intermittent,ratios in prepared:
        lead=(1-w)*seasonal+w*cat
        if kind=="point":
            target=a*lead+b*np.sqrt(lead)
        else:
            q0=ratios[round(a,2)]
            q1=ratios[round(a+b,2)]
            q=np.where(intermittent,q1,q0)
            target=(1-mix)*1.1*lead+mix*q*lead+buf*np.sqrt(lead)*(1+.5*intermittent)
        result.append(target)
    return np.stack(result)

def rollout(target,demand,opening):
    oh,p1,p2=[np.array(v,copy=True,dtype=float) for v in opening]
    initial=oh.copy(); orders=np.zeros((6,len(oh)))
    for t in range(6):
        a0,a1=pipeline_at(orders,t+1,p1,p2)
        orders[t]=np.maximum(0,np.rint(target[t]-oh-a0-a1))
        oh=np.maximum(0,oh+a0-demand[t])
    assert np.isfinite(orders).all()
    return simulate(orders,demand,initial,p1,p2),orders

def evaluate(o,y,c,prepared,specs,progress,done,total):
    base=np.rint(mean(c[:,o-13:o]))
    scenarios=[]
    for seed in CV_INITIAL_STATE_SEEDS:
        rng=np.random.default_rng(seed*1000+o)
        opening=[base.copy() for _ in range(3)]
        if seed: opening=[v*rng.uniform(.6,1.4,size=len(v)) for v in opening]
        for stress in DEMAND_STRESS:
            scenarios.append((y[:,o:o+8].T*(1+stress),opening))
    result={}
    progress.update("policy_sweep",0,len(specs),origin=o,completed_folds=done,total_folds=total)
    for i,spec in enumerate(specs):
        target=targets(prepared,spec)
        rs=[rollout(target,d,opening)[0] for d,opening in scenarios]
        result[spec]={"cost":float(np.mean([r.scored_cost for r in rs])),
                      "fill":float(np.mean([r.fill_rate_scored for r in rs]))}
        if (i+1)%10==0 or i+1==len(specs):
            progress.update("policy_sweep",i+1,len(specs),origin=o,completed_folds=done,total_folds=total,
                best_fold_cost=min(r["cost"] for r in result.values()))
    return result

def run(out_dir):
    out_dir=Path(out_dir)
    with Progress(out_dir/"progress.json") as p:
        ds=load_dataset()
        _,anchor=run_organizer_benchmark(ds)
        assert abs(anchor.scored_cost-4334)<1e-6
        y=np.hstack([ds.history,ds.game_sales]); c,woy,_=full_panel(ds)
        model=Forecasts(c,woy); specs=candidates()
        total=len(TRAIN_ORIGINS)+len(HOLDOUT)+1
        scores={s:[] for s in specs}
        for i,o in enumerate(TRAIN_ORIGINS):
            prepared=model.prepare(o,p,i+1,i,total)
            row=evaluate(o,y,c,prepared,specs,p,i,total)
            for s in specs: scores[s].append(row[s])
        def record(s):
            return {"spec":list(s),"robust":robust([v["cost"] for v in scores[s]]),
                "cv_cost":float(np.mean([v["cost"] for v in scores[s]])),
                "cv_fill":float(np.mean([v["fill"] for v in scores[s]]))}
        ranking=sorted(specs,key=lambda s:record(s)["robust"])
        # Freeze global and per-family winners and incumbent before holdout.
        selected=list(dict.fromkeys([ranking[0],next(s for s in ranking if s[0]=="point"),
                 next(s for s in ranking if s[0]=="hybrid"),INCUMBENT]))
        feasible=[s for s in ranking if record(s)["cv_fill"]>=.95]
        if feasible: selected=list(dict.fromkeys(selected+[feasible[0]]))
        frozen={"selected_on":"eight original CV folds only","holdout_origins":list(HOLDOUT),
            "ranking":[record(s) for s in ranking],"selected":[record(s) for s in selected],
            "cv_95pct_feasible":len(feasible)}
        (out_dir/"frozen_selection.json").write_text(json.dumps(frozen,indent=2))
        validation={s:[] for s in selected}
        for i,o in enumerate(HOLDOUT):
            done=len(TRAIN_ORIGINS)+i
            prepared=model.prepare(o,p,done+1,done,total)
            row=evaluate(o,y,c,prepared,selected,p,done,total)
            for s in selected: validation[s].append(row[s])
        prepared=model.prepare(157,p,total,total-1,total)
        report=[]
        for i,s in enumerate(selected):
            p.update("replay",i,len(selected),completed_folds=total-1,total_folds=total)
            r,orders=rollout(targets(prepared,s),ds.game_sales.T,[ds.on_hand0,ds.pipe_w1,ds.pipe_w2])
            independent=independent_simulate(orders,ds.game_sales.T,ds.on_hand0,ds.pipe_w1,ds.pipe_w2)
            assert abs(independent[3]-r.scored_cost)<1e-6
            assert np.allclose(r.weekly_total[:2],[380.6,533.2])
            report.append(record(s)|{"holdout_cost":float(np.mean([x["cost"] for x in validation[s]])),
                "holdout_fill":float(np.mean([x["fill"] for x in validation[s]])),
                "replay_cost":r.scored_cost,"holding":r.scored_holding,
                "lost":r.lost_units_scored,"fill":r.fill_rate_scored})
        incumbent=next(r for r in report if tuple(r["spec"])==INCUMBENT)
        assert abs(incumbent["replay_cost"]-3775.2)<1e-6, "incumbent replay drift"
        result={"validity":"retrospective; historical holdout reserved for this sweep only",
                "candidates":len(specs),"cv_folds":len(TRAIN_ORIGINS),"holdout_folds":len(HOLDOUT),
                "cv_95pct_feasible":len(feasible),"results":report}
        (out_dir/"report.json").write_text(json.dumps(result,indent=2))
        p.update("replay",len(selected),len(selected),completed_folds=total,total_folds=total,results=report)
        print("VN2_SWEEP_REPORT "+json.dumps(result),flush=True)
        return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True)
    run(parser.parse_args().out_dir)
