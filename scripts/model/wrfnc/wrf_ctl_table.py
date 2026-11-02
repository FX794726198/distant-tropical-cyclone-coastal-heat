"""Summary and SI table of the WRF control-run check (wrf_ctl_check.py output).
Usage: python3 wrf_ctl_table.py CHECKDIR TABLE_MD"""
import sys
import numpy as np, pandas as pd

D, OUT = sys.argv[1], sys.argv[2]
C = pd.read_csv(f'{D}/ctl_check_cases.csv'); S = pd.read_csv(f'{D}/ctl_check_stations.csv')
S['rain_obs'] = S.rain_obs.clip(lower=0)                       # ISD-Lite trace = -0.1 mm
ORDER = ['W17', 'W3', 'W5', 'GD', 'W22', 'C16']
CN = {'W17': 'NW Mexico', 'W3': 'N Mozambique', 'W5': 'SE India', 'GD': 'Guangdong', 'W22': 'NE US', 'C16': 'SE US'}
C['o'] = C.coast.map(ORDER.index); C = C.sort_values(['o', 'case'])
g = S.groupby('case')
C = C.set_index('case')
C['n_sta'] = g.size().reindex(C.index).fillna(0).astype(int)
C['rain_obs'] = g.rain_obs.mean().reindex(C.index); C['n_rain'] = g.rain_obs.count().reindex(C.index).fillna(0).astype(int)
C['rain_e5sta'] = S[S.rain_obs.notna()].groupby('case').rain_e5.mean().reindex(C.index)
d = (S.TV_wrf - S.TV_obs); S['dW'] = d; S['dE'] = S.TV_e5 - S.TV_obs
C['dTV_wrf'] = S.groupby('case').dW.mean().reindex(C.index); C['dTV_e5'] = S.groupby('case').dE.mean().reindex(C.index)
C['nTV'] = S.groupby('case').dW.count().reindex(C.index).fillna(0).astype(int)
C['dTmax_obs_wrf'] = np.nan      # region Tmax of WRF and station means are different samples; use the case-time comparison
C['dTmax'] = C.Tmax_wrf - C.Tmax_e5; C['don'] = C.onshore_wrf - C.onshore_e5; C['dSW'] = C.SW_wrf - C.SW_e5


def f(x, n=1, sign=False):
    if pd.isna(x):
        return '–'
    x = 0.0 if round(x, n) == 0 else x
    s = f'{x:+.{n}f}' if sign else f'{x:.{n}f}'
    s = '0.' + '0' * n if s.lstrip('+-') == '0.' + '0' * n else s
    return s.replace('-', '−')


L = ['| Storm | Tmax, WRF − ERA5 (K) | T2 at V, WRF − stations (K; n) | Onshore wind, WRF / ERA5 (m s^−1^) | Sunshine, WRF / ERA5 (W m^−2^) | Cloud, WRF / ERA5 | Rain, WRF / ERA5 / stations (mm) | Storm offset (km) | Storm pressure, WRF / ERA5 (hPa) |',
     '|---|---|---|---|---|---|---|---|---|']
last = None
for k, r in C.iterrows():
    nm = r['name'].split(':')[0].title()
    if r.coast != last:
        L.append(f"| {CN[r.coast]} |" + ' |' * 8); last = r.coast
    rs = f"{f(r.rain_day_wrf, 1)} / {f(r.rain_day_e5, 1)} / {f(r.rain_obs, 1)}"
    L.append(f"| {nm} | {f(r.dTmax, 1, True)} | {f(r.dTV_wrf, 1, True)}; {r.nTV} | "
             f"{f(r.onshore_wrf, 1, True)} / {f(r.onshore_e5, 1, True)} | {f(r.SW_wrf, 0)} / {f(r.SW_e5, 0)} | "
             f"{f(r.cloud_wrf, 2)} / {f(r.cloud_e5, 2)} | {rs} | {f(r.storm_dist_wrf, 0)} | {f(r.storm_slp_wrf, 0)} / {f(r.storm_slp_e5, 0)} |")
open(OUT, 'w').write('\n'.join(L) + '\n')
pd.set_option('display.width', 250)
cols = ['coast', 'name', 'dTmax', 'dTV_wrf', 'dTV_e5', 'nTV', 'onshore_wrf', 'onshore_e5', 'onshore_obs', 'SW_wrf', 'SW_e5', 'cloud_wrf', 'cloud_e5',
        'rain_day_wrf', 'rain_day_e5', 'rain_obs', 'rain_e5sta', 'n_rain', 'storm_dist_wrf', 'storm_slp_wrf', 'storm_slp_e5', 'storm_wind_kt']
print(C[cols].round(2).to_string())
print('\nmeans over the 17 runs:', C[['dTmax', 'dTV_wrf', 'dTV_e5', 'don', 'dSW']].mean().round(2).to_dict())
print('abs means:', C[['dTmax', 'dTV_wrf', 'dTV_e5', 'don']].abs().mean().round(2).to_dict())
print('|dTmax| <= 1 K:', int((C.dTmax.abs() <= 1).sum()), '| |dTV_wrf| <= 1.5:', int((C.dTV_wrf.abs() <= 1.5).sum()), 'of', int(C.dTV_wrf.notna().sum()))
print('onshore sign WRF = ERA5:', int((np.sign(C.onshore_wrf) == np.sign(C.onshore_e5)).sum()), '; corr', round(C.onshore_wrf.corr(C.onshore_e5), 2))
print('rain corr WRF-ERA5', round(C.rain_day_wrf.corr(C.rain_day_e5), 2), '; SW corr', round(C.SW_wrf.corr(C.SW_e5), 2), '; cloud corr', round(C.cloud_wrf.corr(C.cloud_e5), 2))
print('storm dist median/max', C.storm_dist_wrf.median(), C.storm_dist_wrf.max(), '; slp WRF-ERA5 median', (C.storm_slp_wrf - C.storm_slp_e5).median())
print('stations total', len(S), 'with T at V', int(S.dW.notna().sum()))
