"""Full-precision source table of Table S5 (fourth review, 2026-10-30): every WRF control/edited pair, 19 stations.

Same definitions as wrf_analyse.py: 2-m temperature and specific humidity at the station grid points (nearest d02 cell)
at the case time V (06 UTC = 14 LT), and the daily maximum temperature as the maximum of the hourly 2-m temperature over
08-18 LT (00-10 UTC) of the case day at each station, then averaged over the 19 stations.  Differences are control minus
edited run, so a positive value is warming by the storm.
Pairs: (ctl, wide) and (ctl, nullw) for the 54-h runs, (ctl_s30, wide_s30) for the runs started 30 h before V.
Usage: TCH=/scratch/$USER/tcheat python wrf_tabS5.py
Output ($TCH/wrf/): wrf_tabS5.csv (station means, one row per pair), wrf_tabS5_stations.csv (per station and run)
"""
import glob, os
import numpy as np, pandas as pd, netCDF4

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
STN = pd.read_csv(os.path.join(TCH, 'inputs', 'gd_stations.csv'), index_col=0)
CASES = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv')).set_index('case')
PAIRS = [('ctl', 'wide', 54), ('ctl', 'nullw', 54), ('ctl_s30', 'wide_s30', 30)]


def station_values(run_dir, V):
    """T2 and Q2 (g/kg) at V and the 08-18 LT maximum of T2 at each station."""
    T, Q, tt, jj = [], [], [], None
    for f in sorted(glob.glob(os.path.join(run_dir, 'wrfout_d02_*'))):
        nc = netCDF4.Dataset(f)
        if jj is None:
            la, lo = nc['XLAT'][0], nc['XLONG'][0]
            jj, ii = np.array([np.unravel_index(np.argmin((la - a) ** 2 + ((lo - b) * np.cos(np.radians(a))) ** 2), la.shape)
                               for a, b in zip(STN.lat, STN.lon)]).T
        for k, t in enumerate(nc['Times'][:]):
            tt.append(pd.Timestamp(b''.join(t).decode().replace('_', ' ')))
            T.append(np.asarray(nc['T2'][k][jj, ii], float)); Q.append(1000 * np.asarray(nc['Q2'][k][jj, ii], float))
        nc.close()
    T = pd.DataFrame(np.array(T), index=tt, columns=STN.index); Q = pd.DataFrame(np.array(Q), index=tt, columns=STN.index)
    day = T[(T.index >= V.normalize()) & (T.index <= V.normalize() + pd.Timedelta('10h'))]
    assert len(day) == 11, (run_dir, len(day))                           # hourly output, 00..10 UTC
    return pd.DataFrame(dict(T14=T.loc[V], Tmax=day.max(), Q14=Q.loc[V]))


rows, st_rows = [], []
for d in sorted(glob.glob(os.path.join(TCH, 'wrf', 'case_*'))):
    c = int(d.split('_')[-1]); V = pd.Timestamp(CASES.loc[c, 'V'])
    for ctl, exp, start in PAIRS:
        if not (glob.glob(os.path.join(d, ctl, 'wrfout_d02_*')) and glob.glob(os.path.join(d, exp, 'wrfout_d02_*'))):
            continue
        a, b = station_values(os.path.join(d, ctl), V), station_values(os.path.join(d, exp), V)
        for run, s in ((ctl, a), (exp, b)):
            st_rows.append(s.assign(case=c, run=run, pair=f'{ctl}-{exp}').reset_index(names='station'))
        r = dict(case=c, V=str(V), control=ctl, edited=exp, start_h=start, n_stations=int(a.T14.notna().sum()))
        for k in ('T14', 'Tmax', 'Q14'):
            r[f'{k}_control'] = a[k].mean(); r[f'{k}_edited'] = b[k].mean(); r[f'd{k}'] = (a[k] - b[k]).mean()
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(TCH, 'wrf', 'wrf_tabS5.csv'), index=False, float_format='%.6f')
pd.concat(st_rows).to_csv(os.path.join(TCH, 'wrf', 'wrf_tabS5_stations.csv'), index=False, float_format='%.6f')
print(R[['case', 'control', 'edited', 'start_h', 'dT14', 'dTmax', 'dQ14']].round(4).to_string(index=False))
