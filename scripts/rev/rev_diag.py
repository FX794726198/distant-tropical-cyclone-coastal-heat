"""What the removal changes in the GraphCast initial state (independent review, 2026-10-29). CPU only, no model.

For every hot case, lead time and input time: the edit (variant minus CTL) at the 19 stations for NOTC, WIDE, WIDEL,
W1000 and NULLW: wind change at 850 hPa and 10 m, geopotential height change at 850 and 500 hPa, sea-level pressure
change, the layer-mean temperature implied by the edited geopotential through the hypsometric equation (1000-850 and
850-500 hPa; temperature itself is not edited outside the core) and its largest value anywhere; the distance between
the null point and the real storm.  Maps of the first input time of the 48-h lead for cases 6 and 16.
Usage: TASK=i NTASK=n python rev_diag.py   ->  $TCH/rev/rev_diag_<i>.csv, $TCH/rev/rev_diag_maps_<i>.npz
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'scripts'))
import numpy as np, pandas as pd
import rev_build  # noqa: F401
import gc_build as B
from tcfilter import find_centre, gc_dist

TCH = B.TCH; OUT = os.path.join(TCH, 'rev'); os.makedirs(OUT, exist_ok=True)
RD, G0 = 287.05, 9.80665
cases = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv')); cases = cases[cases.kind == 'heat']
TASK, NTASK = int(os.environ.get('TASK', 0)), int(os.environ.get('NTASK', 1)); cases = cases.iloc[TASK::NTASK]
tracks = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_tracks.csv'))
st = B.STN
rows, maps = [], {}
for _, c in cases.iterrows():
    V = pd.Timestamp(c.V)
    for L in (24, 30, 36, 42, 48):
        t0 = V - pd.Timedelta(hours=L)
        for t in (t0 - pd.Timedelta('6h'), t0):
            g = tracks[(tracks.case == c.case) & (tracks.time == t.strftime('%Y-%m-%dT%H'))]
            if g.empty:
                continue
            f, lat, lon, lev = B.load_time(t)
            LON, LAT = np.meshgrid(lon, lat)
            cc = find_centre(f['mean_sea_level_pressure'], LAT, LON, g.lon.iloc[0], g.lat.iloc[0])
            R, dmin = B.radius_for(*cc)
            if R < B.R_MIN:
                continue
            k = {int(p): n for n, p in enumerate(lev)}
            jj = np.abs(lat[:, None] - st.lat.values[None, :]).argmin(0); ii = np.abs(lon[:, None] - (st.lon.values[None, :] % 360)).argmin(0)
            dstorm = gc_dist(LON, LAT, *cc)
            nc = B.null_centre_for(*cc)
            for kind in ('NOTC', 'WIDE', 'WIDEL', 'W1000', 'NULLW'):
                o, info = B.modify(f, lat, lon, cc, kind, null_centre=nc)
                dz = {p: (o['geopotential'][k[p]] - f['geopotential'][k[p]]) / G0 for p in (1000, 850, 500)}
                imp_lo = (dz[850] - dz[1000]) * G0 / (RD * np.log(1000 / 850))     # K implied, 1000-850 hPa
                imp_hi = (dz[500] - dz[850]) * G0 / (RD * np.log(850 / 500))       # K implied, 850-500 hPa
                du8 = o['u_component_of_wind'][k[850]] - f['u_component_of_wind'][k[850]]
                dv8 = o['v_component_of_wind'][k[850]] - f['v_component_of_wind'][k[850]]
                du1 = o['10m_u_component_of_wind'] - f['10m_u_component_of_wind']; dv1 = o['10m_v_component_of_wind'] - f['10m_v_component_of_wind']
                dp = (o['mean_sea_level_pressure'] - f['mean_sea_level_pressure']) / 100
                dt8 = o['temperature'][k[850]] - f['temperature'][k[850]]
                s = (jj, ii); near = dstorm < 1600
                rows.append(dict(case=int(c.case), lead=L, time=t.strftime('%Y-%m-%dT%H'), variant=kind, R=R, dmin=dmin,
                                 R_out=info.get('R_out', np.nan), null_sep=gc_dist(nc[0], nc[1], *cc) if nc else np.nan,
                                 stn_dV850=float(np.hypot(du8, dv8)[s].mean()), stn_dV850_max=float(np.hypot(du8, dv8)[s].max()),
                                 stn_dV10=float(np.hypot(du1, dv1)[s].mean()), stn_dz850=float(dz[850][s].mean()), stn_dz500=float(dz[500][s].mean()),
                                 stn_dmsl=float(dp[s].mean()), stn_dT850=float(dt8[s].mean()), stn_imp_lo=float(imp_lo[s].mean()),
                                 stn_imp_hi=float(imp_hi[s].mean()), max_abs_imp_lo=float(np.abs(imp_lo[near]).max()),
                                 max_abs_imp_hi=float(np.abs(imp_hi[near]).max()), max_dmsl=float(np.abs(dp[near]).max())))
                if c.case in (6, 16) and L == 48 and t == t0 - pd.Timedelta('6h') and kind in ('WIDE', 'WIDEL'):
                    wj = (lat >= 5) & (lat <= 40); wi = (lon >= 95) & (lon <= 145)
                    sub = lambda a_: a_[wj][:, wi][::4, ::4].astype('float32')
                    for nm, a_ in (('du850', du8), ('dv850', dv8), ('dmsl', dp), ('dz500', dz[500]), ('imp_1000_500', (dz[500] - dz[1000]) * G0 / (RD * np.log(2)))):
                        maps[f'{c.case}|{kind}|{nm}'] = sub(a_)
                    maps['lat'] = lat[wj][::4]; maps['lon'] = lon[wi][::4]; maps[f'{c.case}|centre'] = np.array(cc)
            print(c.case, L, t, 'done', flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, f'rev_diag_{TASK}.csv'), index=False, float_format='%.4g')
np.savez_compressed(os.path.join(OUT, f'rev_diag_maps_{TASK}.npz'), **maps)
print(D.groupby('variant')[['stn_dV850', 'stn_dV850_max', 'stn_dV10', 'stn_dz850', 'stn_dz500', 'stn_dmsl', 'stn_dT850', 'stn_imp_lo', 'stn_imp_hi',
                            'max_abs_imp_lo', 'max_abs_imp_hi', 'null_sep']].mean().round(3).to_string())
print('NULLW separation < 1600 km:', int((D[D.variant == 'NULLW'].null_sep < 1600).sum()), 'of', int(D[D.variant == 'NULLW'].null_sep.notna().sum()))
