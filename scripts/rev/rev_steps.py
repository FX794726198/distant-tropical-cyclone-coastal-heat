"""Early evolution after the edit (independent review, 2026-10-29): every 6-h step of the 48-h forecast for cases 6 and
16, CTL, WIDE and WIDEL, both models.  Station means of 2-m and 850-hPa temperature, sea-level pressure, 10-m wind and
500-hPa vertical velocity at each step, and maps of the edit's effect at +6 and +12 h.
Usage (GPU job): python rev_steps.py --model small|oper  ->  $TCH/rev/steps_<model>.csv, steps_<model>_maps.npz
"""
import argparse, dataclasses, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'scripts'))
# rev13 split: see rev_run.py; a separate job owns steps_<model>.*, so skip it here once claimed or written
_m = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'small'
_r = os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'rev')
if os.environ.get(f'REV_{_m.upper()}_OWNER') != '1' and \
        (os.path.exists(os.path.join(_r, f'{_m}_split.claimed')) or os.path.exists(os.path.join(_r, f'steps_{_m}.csv'))):
    print(f'rev_steps: {_m} model belongs to the separate job (rev/{_m}_split.claimed), skipped', flush=True); sys.exit(0)
import numpy as np, pandas as pd, xarray as xr
import rev_build  # noqa: F401
import gc_run as R
B = R.B
import jax
from graphcast import data_utils, rollout
from tcfilter import find_centre

ap = argparse.ArgumentParser(); ap.add_argument('--model', default='small'); ap.add_argument('--cases', default='6,16')
a = ap.parse_args()
TCH = B.TCH; OUT = os.path.join(TCH, 'rev'); os.makedirs(OUT, exist_ok=True)
fn, task_config = R.load_model(a.model); static = B.load_static(); sub = R.SUB[a.model]
cases = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv')); tracks = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_tracks.csv'))
pts = dict(lat=xr.DataArray(B.STN.lat.values, dims='station'), lon=xr.DataArray(B.STN.lon.values, dims='station'))
rows, maps = [], {}
L = 48
for cn in [int(x) for x in a.cases.split(',')]:
    c = cases[cases.case == cn].iloc[0]; V = pd.Timestamp(c.V)
    t0 = V - pd.Timedelta(hours=L); times = [t0 - pd.Timedelta('6h'), t0]
    base, cen = [], []
    for t in times:
        g = tracks[(tracks.case == cn) & (tracks.time == t.strftime('%Y-%m-%dT%H'))]
        f, lat, lon, lev = B.load_time(t); LON, LAT = np.meshgrid(lon, lat)
        base.append(f); cen.append(find_centre(f['mean_sea_level_pressure'], LAT, LON, g.lon.iloc[0], g.lat.iloc[0]))
    preds = {}
    for kind in ('CTL', 'WIDE', 'WIDEL'):
        fl = [B.modify(f, lat, lon, cc, kind)[0] for f, cc in zip(base, cen)]
        ds = B.to_dataset(times, fl, lat, lon, lev, static, n_targets=L // 6, sub=sub)
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(ds, target_lead_times=slice('6h', f'{L}h'),
                                                                               **dataclasses.asdict(task_config))
        p = rollout.chunked_prediction(fn, rng=jax.random.PRNGKey(0), inputs=inputs, targets_template=targets * np.nan, forcings=forcings).isel(batch=0)
        preds[kind] = p
        for n in range(p.sizes['time']):
            q = p.isel(time=n)
            rows.append(dict(case=cn, variant=kind, step_h=6 * (n + 1), t2m=float(q['2m_temperature'].interp(**pts).mean()),
                             t850=float(q['temperature'].sel(level=850).interp(**pts).mean()),
                             msl=float((q['mean_sea_level_pressure'] / 100).interp(**pts).mean()),
                             u10=float(q['10m_u_component_of_wind'].interp(**pts).mean()), v10=float(q['10m_v_component_of_wind'].interp(**pts).mean()),
                             w500=float(q['vertical_velocity'].sel(level=500).interp(**pts).mean())))
        print(cn, kind, 'done', flush=True)
    for kind in ('WIDE', 'WIDEL'):
        for n, h in ((0, 6), (1, 12), (7, 48)):
            d = (preds['CTL'] - preds[kind]).isel(time=n).sel(lat=slice(5, 40), lon=slice(95, 145))
            s_ = 4 if a.model == 'oper' else 1                     # maps on a 1-degree grid
            for nm, v in (('t850', d['temperature'].sel(level=850)), ('t2m', d['2m_temperature']), ('msl', d['mean_sea_level_pressure'] / 100)):
                maps[f'{cn}|{kind}|{h}|{nm}'] = v.values[::s_, ::s_].astype('float32')
            maps['lat'] = d.lat.values[::s_]; maps['lon'] = d.lon.values[::s_]
pd.DataFrame(rows).to_csv(os.path.join(OUT, f'steps_{a.model}.csv'), index=False, float_format='%.5g')
np.savez_compressed(os.path.join(OUT, f'steps_{a.model}_maps.npz'), **maps)
S = pd.DataFrame(rows).pivot_table(index=['case', 'step_h'], columns='variant', values=['t2m', 't850', 'msl'])
for v in ('t2m', 't850', 'msl'):
    S[(v, 'CTL-WIDE')] = S[(v, 'CTL')] - S[(v, 'WIDE')]; S[(v, 'CTL-WIDEL')] = S[(v, 'CTL')] - S[(v, 'WIDEL')]
print(S[[(v, k) for v in ('t2m', 't850', 'msl') for k in ('CTL-WIDE', 'CTL-WIDEL')]].round(2).to_string())
