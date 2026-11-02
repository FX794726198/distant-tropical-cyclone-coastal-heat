"""WRF CTL vs NOTC pilot summary: Guangdong station-mean 14 LT T2, 08-18 LT Tmax, Q2 and coastal onshore wind.

Usage: python wrf_analyse.py  (expects $TCH/wrf/case_XXX/{ctl,notc,wide}/wrfout_d02_*)
"""
import glob, os
import numpy as np, pandas as pd, netCDF4

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
STN = pd.read_csv(os.path.join(TCH, 'inputs', 'gd_stations.csv'), index_col=0)
COAST = [59493099999, 59501099999, 59663099999, 59673099999, 59658099999]
CASES = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv')).set_index('case')


def series(run_dir, dom='d02'):
    fs = sorted(glob.glob(os.path.join(run_dir, f'wrfout_{dom}_*')))
    T, Q, U, V, tt = [], [], [], [], []
    for f in fs:
        nc = netCDF4.Dataset(f)
        if not tt:
            la, lo = nc['XLAT'][0], nc['XLONG'][0]
            idx = [np.unravel_index(np.argmin((la - a) ** 2 + ((lo - b) * np.cos(np.radians(a))) ** 2), la.shape)
                   for a, b in zip(STN.lat, STN.lon)]
            jj, ii = np.array(idx).T
        times = [b''.join(x).decode() for x in nc['Times'][:]]
        for k, t in enumerate(times):
            tt.append(pd.Timestamp(t.replace('_', ' ')))
            T.append(nc['T2'][k][jj, ii]); Q.append(nc['Q2'][k][jj, ii] * 1000)
            U.append(nc['U10'][k][jj, ii]); V.append(nc['V10'][k][jj, ii])
        nc.close()
    cols = STN.index
    mk = lambda a: pd.DataFrame(np.array(a), index=tt, columns=cols)
    return mk(T), mk(Q), mk(U), mk(V)


rows = []
for d in sorted(glob.glob(os.path.join(TCH, 'wrf', 'case_*'))):
    c = int(d.split('_')[-1]); Vt = pd.Timestamp(CASES.loc[c, 'V'])
    out = {'case': c, 'V': str(Vt)}
    res = {}
    for run in ('ctl', 'notc', 'wide', 'weak'):
        if not glob.glob(os.path.join(d, run, 'wrfout_d02_*')):
            continue
        T, Q, U, V = series(os.path.join(d, run))
        day = T[(T.index >= Vt.normalize()) & (T.index <= Vt.normalize() + pd.Timedelta('10h'))]
        res[run] = dict(T14=float(T.loc[Vt].mean()), Tmax=float(day.max().mean()), Q14=float(Q.loc[Vt].mean()),
                        onshore=float((V.loc[Vt, [x for x in COAST if x in V.columns]] > 0).mean()))
    for run, r in res.items():
        for k, v in r.items():
            out[f'{k}_{run}'] = v
    for run in ('notc', 'wide'):
        if 'ctl' in res and run in res:
            for k in ('T14', 'Tmax', 'Q14', 'onshore'):
                out[f'd{k}_{run}'] = res[run][k] - res['ctl'][k]
    rows.append(out)
P = pd.DataFrame(rows)
P.to_csv(os.path.join(TCH, 'wrf', 'wrf_summary.csv'), index=False)
print(P.round(2).to_string())
for run in ('notc', 'wide'):
    if f'dTmax_{run}' in P:
        Q = P.dropna(subset=[f'dTmax_{run}'])
        print(f"\n{run.upper()} - CTL: 14 LT T2 {Q[f'dT14_{run}'].mean():+.2f} K, Tmax {Q[f'dTmax_{run}'].mean():+.2f} K, "
              f"Q2 {Q[f'dQ14_{run}'].mean():+.2f} g/kg, onshore share {Q[f'donshore_{run}'].mean():+.2f}; "
              f"cases with Tmax drop >= 0.8 K: {(Q[f'dTmax_{run}'] <= -0.8).sum()} of {len(Q)}, negative: {(Q[f'dTmax_{run}'] < 0).sum()} of {len(Q)}")
