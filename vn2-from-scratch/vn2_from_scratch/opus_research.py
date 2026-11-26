"""Adapt Opus's diverse ensemble and cost-aware policy to strict VN2 timing.

This is a research comparison, not a byte-for-byte reproduction of MLForecast.
Features and direct horizons reuse our causal hierarchical research code.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from . import ensemble_sweep as old
from .hierarchical_research import Hierarchy, scenarios, forecast_stats, aggregate
from .causal_cost_policy import project_order, causal_rollout
from .progress import Progress
from .simulator import simulate
from .verify import independent_simulate

BASELINE=("incumbent",)
PUBLISHED_OPUS=("diverse",.9,.33800656001887974,.5813428492464154,1.18,1)
README_OPUS=("pair",.5,.25,.65,1.,1)
SPECS=[BASELINE,PUBLISHED_OPUS,README_OPUS]+[
    (family,share,seasonal,alpha,multiplier,horizon)
    for family,share,seasonal,alpha,multiplier,horizon in itertools.product(
        ("diverse",),(.5,.9),(.25,.338),(.55,.65,.75,5/6),(1.,1.15),(1,3))]
SPECS=list(dict.fromkeys(SPECS))

def fit_models(features,origin,progress,done,total,tag):
    x,y=features.training(origin)
    lgb=dict(n_estimators=200,learning_rate=.05,min_child_samples=10,
             subsample=.8,subsample_freq=1,colsample_bytree=.8,
             verbosity=-1,n_jobs=4,random_state=42)
    cb=dict(iterations=200,learning_rate=.05,depth=6,verbose=False,
            thread_count=4,random_seed=42,allow_writing_files=False,
            cat_features=features.cat_names)
    models={
        "lgb_default":LGBMRegressor(**lgb,num_leaves=31),
        "lgb_shallow":LGBMRegressor(**(lgb|dict(num_leaves=7,max_depth=3,n_estimators=300,learning_rate=.04,min_child_samples=20))),
        "lgb_tweedie":LGBMRegressor(**lgb,num_leaves=31,objective="tweedie",tweedie_variance_power=1.5),
        "cb_default":CatBoostRegressor(**cb),
        "cb_deep":CatBoostRegressor(**(cb|dict(iterations=150,learning_rate=.04,depth=10))),
    }
    for i,(name,model) in enumerate(models.items()):
        progress.update("fit_"+tag,i,len(models),origin=origin,model=name,completed_folds=done,total_folds=total)
        if name.startswith("lgb"):
            model.fit(x,y,categorical_feature=features.cat_names)
        else:
            model.fit(x,y)
    progress.update("fit_"+tag,len(models),len(models),origin=origin,completed_folds=done,total_folds=total)
    return models

def component_forecasts(features,models,origins):
    outputs={k:[] for k in ("seasonal","lgb_default","cb_default","lgb_mean","cb_mean")}
    for t in origins:
        p={k:features.predict_point(m,t) for k,m in models.items()}
        outputs["seasonal"].append(features.seasonal(t))
        outputs["lgb_default"].append(p["lgb_default"])
        outputs["cb_default"].append(p["cb_default"])
        outputs["lgb_mean"].append(np.mean([v for k,v in p.items() if k.startswith("lgb")],axis=0))
        outputs["cb_mean"].append(np.mean([v for k,v in p.items() if k.startswith("cb")],axis=0))
    return {k:np.stack(v) for k,v in outputs.items()}

def blend(c,family,share,seasonal):
    suffix="default" if family=="pair" else "mean"
    return (1-seasonal)*(share*c["lgb_"+suffix]+(1-share)*c["cb_"+suffix])+seasonal*c["seasonal"]

def residual_sigma(actual,pred):
    residual=np.where(np.isfinite(actual),actual-pred,np.nan)
    count=np.isfinite(residual).sum(axis=0)
    sq=np.nansum(residual**2,axis=0)
    # Shrink per-series variance toward a pooled horizon variance.
    prior=np.divide(sq.sum(axis=0),count.sum(axis=0),out=np.ones(3),where=count.sum(axis=0)>0)
    return np.maximum(.5,np.sqrt((sq+5*prior)/(count+5)))

def prepare(features,origin,progress,done,total):
    # All uncertainty calibration outcomes end strictly before the outer origin.
    cut=origin-26
    inner=fit_models(features,cut,progress,done,total,"calibration")
    inner_origins=range(cut,origin-2)
    inner_fc=component_forecasts(features,inner,inner_origins)
    actual=np.stack([features.c[:,t:t+3] for t in inner_origins])
    models=fit_models(features,origin,progress,done,total,"outer")
    components=component_forecasts(features,models,range(origin,origin+6))
    # Reuse the previously verified incumbent's exact direct-forecast fit.
    baseline=old.Forecasts.fit(features,origin,progress,done+1,done,total)
    inc=[]
    for t in range(origin,origin+6):
        cb=np.column_stack([np.maximum(baseline.predict(features.feature(t,h)),0.) for h in (1,2,3)])
        inc.append(.5*features.seasonal(t)+.5*cb)
    prepared={BASELINE:(np.stack(inc),None)}
    cache={}
    for s in SPECS[1:]:
        family,share,seasonal,alpha,mult,horizon=s
        key=(family,share,seasonal)
        if key not in cache:
            pred=blend(components,*key)
            sigma=residual_sigma(actual,blend(inner_fc,*key))
            cache[key]=(pred,sigma)
        pred,sigma=cache[key]
        prepared[s]=(pred,sigma[:,horizon-1])
    return prepared

def orders_for(demand,opening,prepared,spec):
    fc,sigma=prepared[spec]
    if spec==BASELINE:
        lead=fc.sum(axis=2)
        return old.rollout(1.1*lead+.3*np.sqrt(lead),demand,opening)[1]
    _,_,_,alpha,mult,_=spec
    z=NormalDist().inv_cdf(alpha)
    def decide(t,oh,a0,a1,history):
        return project_order(oh,a0,a1,fc[t],sigma,z,mult)
    return causal_rollout(demand,opening,decide)

def evaluate(prepared,y,c,o,specs,progress,done,total):
    ss=list(scenarios(y,c,o));out={}
    for i,s in enumerate(specs):
        runs=[simulate(orders_for(d,opening,prepared,s),d,*opening) for d,opening in ss]
        out[s]={"cost":float(np.mean([r.scored_cost for r in runs])),
                "fill":float(np.mean([r.fill_rate_scored for r in runs]))}
        if (i+1)%10==0 or i+1==len(specs):
            progress.update("policy_sweep",i+1,len(specs),origin=o,completed_folds=done,total_folds=total,
                            best_fold_cost=min(r["cost"] for r in out.values()))
    return out

def run(out_dir):
    out=Path(out_dir)
    with Progress(out/"progress.json") as p:
        ds=old.load_dataset()
        _,anchor=old.run_organizer_benchmark(ds)
        assert abs(anchor.scored_cost-4334)<1e-6
        y=np.hstack([ds.history,ds.game_sales]);c,woy,_=old.full_panel(ds)
        features=Hierarchy(c,woy,ds)
        origins=tuple(old.TRAIN_ORIGINS)
        holdout=tuple(old.HOLDOUT)
        total=len(origins)+len(holdout)+1
        scores={s:[] for s in SPECS}
        for i,o in enumerate(origins):
            prepared=prepare(features,o,p,i,total)
            row=evaluate(prepared,y,c,o,SPECS,p,i,total)
            for s in SPECS: scores[s].append(row[s])
        def record(s):
            return {"spec":list(s),"robust":old.robust([v["cost"] for v in scores[s]]),
                    "cv_cost":float(np.mean([v["cost"] for v in scores[s]])),
                    "cv_fill":float(np.mean([v["fill"] for v in scores[s]]))}
        rank=sorted(SPECS,key=lambda s:record(s)["robust"])
        selected=list(dict.fromkeys([BASELINE,README_OPUS,PUBLISHED_OPUS,rank[0],next(s for s in rank if s!=BASELINE)]))
        (out/"frozen_selection.json").write_text(json.dumps({"rank":[record(s) for s in rank],"selected":[record(s) for s in selected]},indent=2))
        validation={s:[] for s in selected}
        for i,o in enumerate(holdout):
            done=len(origins)+i
            prepared=prepare(features,o,p,done,total)
            row=evaluate(prepared,y,c,o,selected,p,done,total)
            for s in selected: validation[s].append(row[s])
        prepared=prepare(features,157,p,total-1,total)
        demand=ds.game_sales.T
        opening=[ds.on_hand0,ds.pipe_w1,ds.pipe_w2]
        results=[]
        for i,s in enumerate(selected):
            p.update("replay",i,len(selected),completed_folds=total-1,total_folds=total)
            orders=orders_for(demand,opening,prepared,s)
            # Verify same-week demand never changes the already placed order.
            for t in range(6):
                altered=demand.copy();altered[t:]+=100
                assert np.array_equal(orders_for(altered,opening,prepared,s)[:t+1],orders[:t+1])
            r=simulate(orders,demand,*opening)
            assert abs(independent_simulate(orders,demand,*opening)[3]-r.scored_cost)<1e-6
            assert np.allclose(r.weekly_total[:2],[380.6,533.2])
            results.append(record(s)|{"holdout_cost":float(np.mean([x["cost"] for x in validation[s]])),
                "holdout_fill":float(np.mean([x["fill"] for x in validation[s]])),
                "replay_cost":r.scored_cost,"holding":r.scored_holding,"lost":r.lost_units_scored,"fill":r.fill_rate_scored})
        assert abs(results[0]["replay_cost"]-3775.2)<1e-6,"incumbent replay drift"
        report={"validity":"retrospective; historical holdout excluded from this sweep only",
                "implementation":"Opus-inspired diverse direct ensemble, strict two-week projection",
                "candidates":len(SPECS),"results":results}
        (out/"report.json").write_text(json.dumps(report,indent=2))
        p.update("replay",len(selected),len(selected),completed_folds=total,total_folds=total,results=results)
        print("VN2_CAUSAL_OPUS_REPORT "+json.dumps(report),flush=True)
        return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True)
    run(parser.parse_args().out_dir)
