"""Research-only, causal hierarchical and distributional forecasting for VN2.

Run with a unique --out-dir. The original 50/50 CatBoost incumbent is a
mandatory replay anchor. Selection uses original CV only; the two later
history windows and public replay are reported after freezing.
"""
from __future__ import annotations
import argparse, itertools, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from . import ensemble_sweep as old
from .progress import Progress
from .simulator import simulate
from .verify import independent_simulate

BASELINE=("incumbent",1.1,.3,0.)
SPECS=[(m,c,b,0.) for m,c,b in itertools.product(
    ("incumbent","hierarchy","lightgbm","ensemble","calibrated"),
    (1.,1.1,1.2,1.3),(0.,.3,.6))]
SPECS += [("quantile",q,b,mix) for q,b,mix in itertools.product(
    (0.65,.8,.9),(0.,.3),(.5,1.))]
ALPHAS=(.65,.8,.9)

def pool_mean(x,groups):
    x=np.asarray(x,float); groups=np.asarray(groups,int)
    valid=np.isfinite(x)
    n=np.bincount(groups,weights=valid.astype(float))
    sums=np.bincount(groups,weights=np.where(valid,x,0.))
    return np.divide(sums,n,out=np.zeros_like(sums),where=n>0)[groups]

def calibration(actual,pred,groups):
    actual=np.asarray(actual,float); pred=np.asarray(pred,float)
    groups=np.asarray(groups,int)
    valid=np.isfinite(actual)&np.isfinite(pred)
    a=np.bincount(groups,weights=np.where(valid,actual,0.))
    p=np.bincount(groups,weights=np.where(valid,pred,0.))
    # Prior equivalent to 100 forecast units, shrinks rare segments.
    return np.clip((a+100)/(p+100),.8,1.2)[groups]

def monotone_quantiles(x):
    return np.sort(np.maximum(np.asarray(x,float),0),axis=-1)

class Hierarchy(old.Forecasts):
    def __init__(self,c,woy,ds):
        super().__init__(c,woy)
        master=ds.master.set_index(["Store","Product"])
        if not master.index.is_unique:
            raise ValueError("master keys must be unique")
        aligned=master.reindex(pd.MultiIndex.from_arrays(ds.keys.T,names=["Store","Product"]))
        if aligned.isna().any().any():
            raise ValueError("missing master metadata")
        self.ids={}
        for name,val in [("Store",ds.keys[:,0]),("Product",ds.keys[:,1])]+[(k,aligned[k].to_numpy()) for k in ("ProductGroup","Division","Department","DepartmentGroup","StoreFormat","Format")]:
            self.ids[name]=pd.factorize(val,sort=True)[0].astype(int)
        self.cat_names=list(self.ids)
        self.rich_cache={}
        self.seg_cache={}
        self.group_cache={}

    def segments(self,t):
        if t not in self.seg_cache:
            b=self.c[:,max(0,t-52):t]
            level=old.mean(b)
            intermittent=old.mean(np.where(np.isfinite(b),b==0,np.nan))>=.35
            self.seg_cache[t]=2*np.digitize(level,[1.,4.])+intermittent.astype(int)
        return self.seg_cache[t]

    def rich(self,t,h):
        key=(t,h)
        if key not in self.rich_cache:
            if t not in self.group_cache:
                extra=[]
                for name,ids in self.ids.items():
                    for win in (4,13,26):
                        b=self.c[:,max(0,t-win):t]
                        count=np.isfinite(b).sum(axis=1)
                        values=np.divide(np.nansum(b,axis=1),count,out=np.full(self.n,np.nan),where=count>0)
                        pooled=pool_mean(values,ids)
                        extra.extend([pooled,np.nan_to_num(values)/(pooled+.5)])
                extra.extend([self.seasonal(t).sum(axis=1),self.segments(t)])
                self.group_cache[t]=np.column_stack(extra).astype(np.float32)
            x=np.column_stack([self.feature(t,h),self.group_cache[t]]).astype(np.float32)
            frame=pd.DataFrame(x,columns=[f"x{i}" for i in range(x.shape[1])])
            for name,ids in self.ids.items():
                frame[name]=ids.astype(np.int32)
            self.rich_cache[key]=frame
        return self.rich_cache[key]

    def training(self,o,quantile=False):
        xx=[]; yy=[]
        for t in range(26,o):
            for h in ((3,) if quantile else (1,2,3)):
                if t+h-1>=o: continue
                target=self.c[:,t:t+3].sum(axis=1) if quantile else self.c[:,t+h-1]
                valid=np.isfinite(target)
                if valid.any():
                    xx.append(self.rich(t,h).loc[valid]); yy.append(target[valid])
        return pd.concat(xx,ignore_index=True),np.concatenate(yy)

    def fit_rich(self,o,progress,done,total,mode):
        quantile=mode=="quantile"
        x,y=self.training(o,quantile)
        progress.update("training_"+mode,0,100,origin=o,completed_folds=done,total_folds=total)
        if mode=="lightgbm":
            model=LGBMRegressor(objective="poisson",n_estimators=100,num_leaves=23,learning_rate=.06,
                min_child_samples=120,reg_lambda=5,n_jobs=4,verbosity=-1,random_state=42)
            model.fit(x,y,categorical_feature=self.cat_names)
        else:
            loss="MultiQuantile:alpha=0.65,0.8,0.9" if quantile else "RMSE"
            class CB:
                def after_iteration(self,info):
                    if info.iteration%10==0:
                        progress.update("training_"+mode,info.iteration,100,origin=o,
                            completed_folds=done,total_folds=total)
                    return True
            model=CatBoostRegressor(iterations=100,depth=6,learning_rate=.08,loss_function=loss,
                thread_count=4,verbose=False,random_seed=42,allow_writing_files=False,
                cat_features=self.cat_names)
            model.fit(x,y,callbacks=[CB()])
        progress.update("training_"+mode,100,100,origin=o,completed_folds=done,total_folds=total)
        return model

    def predict_point(self,m,t):
        return np.column_stack([np.maximum(m.predict(self.rich(t,h)),0.) for h in (1,2,3)])

    def prepare_all(self,o,progress,done,total):
        baseline=super().fit(o,progress,done+1,done,total)
        # Calibrate on an inner 8-week slice excluded from the calibration fit.
        inner=self.fit_rich(o-8,progress,done,total,"calibration_catboost")
        a=[];pred=[];seg=[]
        for t in range(o-8,o-2):
            s=self.seasonal(t); cf=self.predict_point(inner,t)
            a.append(self.c[:,t:t+3].ravel())
            pred.append((.5*s+.5*cf).ravel())
            seg.append(np.repeat(self.segments(t),3))
        a=np.concatenate(a);pred=np.concatenate(pred);seg=np.concatenate(seg)
        factors=np.ones(6)
        fitted=calibration(a,pred,seg)
        for k in np.unique(seg): factors[k]=fitted[np.flatnonzero(seg==k)[0]]
        cat=self.fit_rich(o,progress,done,total,"catboost")
        lgb=self.fit_rich(o,progress,done,total,"lightgbm")
        quant=self.fit_rich(o,progress,done,total,"quantile")
        points={k:[] for k in ("seasonal","incumbent","hierarchy","lightgbm","ensemble","calibrated")}
        qs=[]
        for t in range(o,o+6):
            sf=self.seasonal(t)
            bc=np.column_stack([np.maximum(baseline.predict(self.feature(t,h)),0.) for h in (1,2,3)])
            cp=self.predict_point(cat,t); lp=self.predict_point(lgb,t)
            points["seasonal"].append(sf)
            points["incumbent"].append(.5*sf+.5*bc)
            points["hierarchy"].append(.5*sf+.5*cp)
            points["lightgbm"].append(.5*sf+.5*lp)
            points["ensemble"].append(.4*sf+.3*cp+.3*lp)
            points["calibrated"].append((.5*sf+.5*cp)*factors[self.segments(t),None])
            qs.append(monotone_quantiles(quant.predict(self.rich(t,3))))
        return {k:np.stack(v) for k,v in points.items()},np.stack(qs),factors.tolist()

def target(points,qs,spec):
    m,c,b,mix=spec
    if m=="quantile":
        lead=points["ensemble"].sum(axis=2)
        q=qs[:,:,ALPHAS.index(c)]
        return (1-mix)*1.1*lead+mix*q+b*np.sqrt(lead)
    lead=points[m].sum(axis=2)
    return c*lead+b*np.sqrt(lead)

def forecast_stats(points,c,o):
    actual=np.stack([c[:,t:t+3] for t in range(o,o+6)])
    mask=np.isfinite(actual)
    result={}
    for m,pred in points.items():
        err=np.where(mask,pred-actual,0.)
        result[m]=[float(np.abs(err).sum()),float(np.where(mask,actual,0).sum()),float(err.sum())]
    return result

def scenarios(y,c,o):
    base=np.rint(old.mean(c[:,o-13:o]))
    for seed in old.CV_INITIAL_STATE_SEEDS:
        rng=np.random.default_rng(seed*1000+o)
        opening=[base.copy() for _ in range(3)]
        if seed: opening=[v*rng.uniform(.6,1.4,size=len(v)) for v in opening]
        for stress in old.DEMAND_STRESS:
            yield y[:,o:o+8].T*(1+stress),opening

def evaluate(points,qs,y,c,o,specs,p,done,total):
    ss=list(scenarios(y,c,o))
    result={}
    p.update("policy_sweep",0,len(specs),origin=o,completed_folds=done,total_folds=total)
    for i,sp in enumerate(specs):
        tg=target(points,qs,sp)
        runs=[old.rollout(tg,d,opening)[0] for d,opening in ss]
        result[sp]={"cost":float(np.mean([r.scored_cost for r in runs])),
                    "fill":float(np.mean([r.fill_rate_scored for r in runs]))}
        if (i+1)%10==0 or i+1==len(specs):
            p.update("policy_sweep",i+1,len(specs),origin=o,completed_folds=done,total_folds=total,
                best_fold_cost=min(r["cost"] for r in result.values()))
    return result

def aggregate(rows):
    out={}
    for m in rows[0]:
        ae,den,bias=np.sum([r[m] for r in rows],axis=0)
        out[m]={"wape":float(ae/den),"bias":float(bias/den)}
    return out

def run(out_dir):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    out=Path(out_dir)
    with Progress(out/"progress.json") as p:
        ds=old.load_dataset()
        _,anchor=old.run_organizer_benchmark(ds)
        assert abs(anchor.scored_cost-4334)<1e-6
        y=np.hstack([ds.history,ds.game_sales]);c,woy,_=old.full_panel(ds)
        model=Hierarchy(c,woy,ds)
        scores={s:[] for s in SPECS}; fstats={"cv":[],"holdout":[],"replay":[]}
        total=len(old.TRAIN_ORIGINS)+len(old.HOLDOUT)+1
        for i,o in enumerate(old.TRAIN_ORIGINS):
            points,qs,factors=model.prepare_all(o,p,i,total)
            fstats["cv"].append(forecast_stats(points,c,o))
            row=evaluate(points,qs,y,c,o,SPECS,p,i,total)
            for s in SPECS: scores[s].append(row[s])
        def record(s):
            return {"spec":list(s),"robust":old.robust([v["cost"] for v in scores[s]]),
                "cv_cost":float(np.mean([v["cost"] for v in scores[s]])),
                "cv_fill":float(np.mean([v["fill"] for v in scores[s]]))}
        rank=sorted(SPECS,key=lambda s:record(s)["robust"])
        selected=list(dict.fromkeys([BASELINE,rank[0]]+
            [next(s for s in rank if s[0]==m) for m in sorted(set(s[0] for s in SPECS))]))
        (out/"frozen_selection.json").write_text(json.dumps({"rank":[record(s) for s in rank],
            "selected":[record(s) for s in selected]},indent=2))
        validation={s:[] for s in selected}
        for i,o in enumerate(old.HOLDOUT):
            done=len(old.TRAIN_ORIGINS)+i
            points,qs,factors=model.prepare_all(o,p,done,total)
            fstats["holdout"].append(forecast_stats(points,c,o))
            row=evaluate(points,qs,y,c,o,selected,p,done,total)
            for s in selected: validation[s].append(row[s])
        points,qs,factors=model.prepare_all(157,p,total-1,total)
        fstats["replay"].append(forecast_stats(points,c,157))
        results=[]
        for i,s in enumerate(selected):
            p.update("replay",i,len(selected),completed_folds=total-1,total_folds=total)
            r,orders=old.rollout(target(points,qs,s),ds.game_sales.T,[ds.on_hand0,ds.pipe_w1,ds.pipe_w2])
            assert abs(independent_simulate(orders,ds.game_sales.T,ds.on_hand0,ds.pipe_w1,ds.pipe_w2)[3]-r.scored_cost)<1e-6
            assert np.allclose(r.weekly_total[:2],[380.6,533.2])
            results.append(record(s)|{"holdout_cost":float(np.mean([x["cost"] for x in validation[s]])),
                "holdout_fill":float(np.mean([x["fill"] for x in validation[s]])),
                "replay_cost":r.scored_cost,"holding":r.scored_holding,"lost":r.lost_units_scored,
                "fill":r.fill_rate_scored})
        assert abs(results[0]["replay_cost"]-3775.2)<1e-6,"incumbent replay drift"
        report={"validity":"retrospective; historical holdout excluded from this sweep only",
            "candidates":len(SPECS),"forecast_metrics":{g:aggregate(v) for g,v in fstats.items()},
            "replay_calibration_factors":factors,"results":results}
        (out/"report.json").write_text(json.dumps(report,indent=2))
        p.update("replay",len(selected),len(selected),completed_folds=total,total_folds=total,results=results)
        print("VN2_HIERARCHY_REPORT "+json.dumps(report),flush=True)
        return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True)
    run(parser.parse_args().out_dir)
