import sys, time
import numpy as np
import pandas as pd
ARENA="C:/Users/Toni/Documents/Default Project/vn2-model-arena"
sys.path.insert(0, ARENA+"/tools")
import inventory_sim as sim
hb_path="C:/Users/Toni/Documents/Default Project/vn2-model-arena/runs3/w04"
sys.path.insert(0, hb_path)
import helper_backtest as hb

# Load data
sales=pd.read_csv(ARENA+"/data/sales.csv")
avail=pd.read_csv(ARENA+"/data/availability.csv")
initial=pd.read_csv(ARENA+"/data/initial_state.csv")

series_sorted=sorted(sales['series_id'].unique().tolist())
# map
sid_to_idx={s:i for i,s in enumerate(series_sorted)}
n=len(series_sorted)
T=157
# demand matrix (n,T)
D=np.zeros((n,T))
A=np.zeros((n,T),dtype=bool)
weeks=sorted(sales['week'].unique().tolist())
w_to_t={w:t for t,w in enumerate(weeks)}
for _,r in sales.iterrows():
    D[sid_to_idx[r['series_id']], w_to_t[r['week']]]=r['y']
for _,r in avail.iterrows():
    A[sid_to_idx[r['series_id']], w_to_t[r['week']]]=bool(r['is_available'])

# initial state aligned
init_state=initial.set_index('series_id').loc[series_sorted][['on_hand','arrival_w1','arrival_w2']].to_numpy(float)

# corrected demand: fill censored with seasonal mean by week-of-year using available
Ycorr=D.copy()
# week of year index (t % 52)
woy=np.array([t%52 for t in range(T)])
for i in range(n):
    for t in range(T):
        if not A[i,t]:
            # seasonal mean over available same woy
            mask=(woy==woy[t]) & A[i]
            if mask.sum()>0:
                Ycorr[i,t]=D[i][mask].mean()
            else:
                Ycorr[i,t]=D[i][A[i]].mean() if A[i].sum()>0 else 0.0

# Simple seasonal naive forecast function
def make_forecast_func():
    # precompute series seasonal mean by woy for all horizons? just average same woy
    def forecast(week_index, state, revealed):
        # state (n,3); revealed (w,n) game demand
        # combined history not needed; use Ycorr and append revealed
        # determine current time index in global history: assume 157 + week_index? for forecast horizon
        # We'll just use seasonal average by woy for h=1..3
        # current global t = 157 + week_index? but for backtest cutoff unknown; use 157
        n_series=state.shape[0]
        # map series? In backtest series same order as sales sorted? initial_state order maybe different.
        # Need map by series? forecast receives state in whatever order (backtest uses initial order from initial.csv not sorted sales?).
        # For real, initial_state order maybe sorted? Let's check: series in initial_state maybe not sorted. We'll handle by global mapping later.
        out=np.zeros((3,n_series))
        return out
    return forecast

# Instead test with series alignment. Backtest uses initial_state order as given. Need map.
print("initial series order matches sorted?", list(initial['series_id'])==series_sorted)
"