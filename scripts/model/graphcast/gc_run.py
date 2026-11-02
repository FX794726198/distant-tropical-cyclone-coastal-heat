"""Run GraphCast CTL / NOTC / WEAK / SHIFT / NULL rollouts for the TC dry-heat pilot.

Usage (inside a GPU job):
  python gc_run.py --model small --task $SLURM_ARRAY_TASK_ID --ntask 8
  python gc_run.py --model oper  --cases 30-60 --variants CTL,NOTC,SHIFT,NULL
  python gc_run.py --model small --variants CTL,WIDE,WIDEH,NULLW --tag _wide
Outputs: $TCH/out_<model><tag>/case_XXX.nc (+ case_XXX.json with filter diagnostics)
"""
import argparse, dataclasses, functools, json, os, sys, time
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/src/graphcast_v011'))
import jax, haiku as hk
from graphcast import autoregressive, casting, checkpoint, data_utils, graphcast, normalization, rollout
import gc_build as B
from tcfilter import find_centre, plane_deficit

TCH = B.TCH
LEADS = [24, 30, 36, 42, 48]
PARAMS = {
    'small': 'GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz',
    'oper': 'GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz',
}
SUB = {'small': 4, 'oper': 1}
WIN = dict(lat=slice(5, 40), lon=slice(95, 140))
COAST = ['59493099999', '59501099999', '59663099999', '59673099999', '59658099999']


def load_model(name):
    gdir = os.path.join(TCH, 'gc')
    with open(os.path.join(gdir, 'params', PARAMS[name]), 'rb') as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    stats = {k: xr.load_dataset(os.path.join(gdir, 'stats', k + '.nc')).compute()
             for k in ('diffs_stddev_by_level', 'mean_by_level', 'stddev_by_level')}
    mc, tc = ckpt.model_config, ckpt.task_config

    def construct():
        p = graphcast.GraphCast(mc, tc)
        p = casting.Bfloat16Cast(p)
        p = normalization.InputsAndResiduals(p, diffs_stddev_by_level=stats['diffs_stddev_by_level'],
                                             mean_by_level=stats['mean_by_level'], stddev_by_level=stats['stddev_by_level'])
        return autoregressive.Predictor(p, gradient_checkpointing=True)

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        return construct()(inputs, targets_template=targets_template, forcings=forcings)

    fwd = jax.jit(run_forward.apply)
    params, state = ckpt.params, {}
    fn = lambda **kw: fwd(params, state, **kw)[0]
    return fn, tc


def extract(ds_pred_last, sub_lat, sub_lon):
    """Station values and a regional window from one prediction time (dims lat, lon[, level])."""
    p = ds_pred_last
    st = B.STN
    pts = dict(lat=xr.DataArray(st.lat.values, dims='station'), lon=xr.DataArray(st.lon.values, dims='station'))
    out = {}
    fields = {'t2m': p['2m_temperature'], 'msl': p['mean_sea_level_pressure'] / 100, 'u10': p['10m_u_component_of_wind'],
              'v10': p['10m_v_component_of_wind'], 't850': p['temperature'].sel(level=850),
              'q850': p['specific_humidity'].sel(level=850) * 1000, 'w500': p['vertical_velocity'].sel(level=500),
              'w850': p['vertical_velocity'].sel(level=850), 'z500': p['geopotential'].sel(level=500) / 9.80665}
    if 'total_precipitation_6hr' in p:
        fields['tp6'] = p['total_precipitation_6hr'] * 1000
    for k, f in fields.items():
        out['stn_' + k] = f.interp(**pts).values.astype('float32')
        out['win_' + k] = f.sel(**WIN).values.astype('float32')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='small', choices=list(PARAMS))
    ap.add_argument('--cases', default=None)
    ap.add_argument('--task', type=int, default=None); ap.add_argument('--ntask', type=int, default=1)
    ap.add_argument('--variants', default=','.join(B.VARIANTS))
    ap.add_argument('--leads', default=','.join(map(str, LEADS)))
    ap.add_argument('--tag', default='', help='suffix of the output directory, e.g. _wide')
    a = ap.parse_args()
    variants = a.variants.split(','); leads = [int(x) for x in a.leads.split(',')]
    cases = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv'))
    tracks = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_tracks.csv'))
    if a.cases:
        lo, hi = map(int, a.cases.split('-')); cases = cases[(cases.case >= lo) & (cases.case <= hi)]
    if a.task is not None:
        cases = cases.iloc[a.task::a.ntask]
    odir = os.path.join(TCH, 'out_' + a.model + a.tag); os.makedirs(odir, exist_ok=True)
    fn, task_config = load_model(a.model)
    static = B.load_static(); sub = SUB[a.model]
    print('devices', jax.devices(), 'cases', list(cases.case), flush=True)
    for _, c in cases.iterrows():
        ofile = os.path.join(odir, f'case_{c.case:03d}.nc')
        if os.path.exists(ofile):
            continue
        V = pd.Timestamp(c.V); t_start = time.time()
        truth_f, lat, lon, lev = B.load_time(V)
        LON, LAT = np.meshgrid(lon, lat)
        res = {v: {} for v in variants}; meta = {'case': int(c.case), 'V': c.V, 'kind': c.kind, 'leads': {}}
        # storm centre at V in ERA5 (for the regeneration check)
        gV = tracks[(tracks.case == c.case) & (tracks.time == V.strftime('%Y-%m-%dT%H'))]
        cV = find_centre(truth_f['mean_sea_level_pressure'], LAT, LON, gV.lon.iloc[0], gV.lat.iloc[0]) if len(gV) else None
        for L in leads:
            t0 = V - pd.Timedelta(hours=L); times = [t0 - pd.Timedelta('6h'), t0]
            base, cen, ok = [], [], True
            for t in times:
                g = tracks[(tracks.case == c.case) & (tracks.time == t.strftime('%Y-%m-%dT%H'))]
                if g.empty: ok = False; break
                f, *_ = B.load_time(t)
                cc = find_centre(f['mean_sea_level_pressure'], LAT, LON, g.lon.iloc[0], g.lat.iloc[0])
                R, dmin = B.radius_for(*cc)
                if R < B.R_MIN: ok = False; break
                base.append(f); cen.append(cc)
            if not ok:
                meta['leads'][L] = 'skipped (storm missing or too close to Guangdong)'; continue
            meta['leads'][L] = {'centres': cen}
            for kind in variants:
                fl, infos = [], []
                for f, cc in zip(base, cen):
                    o, info = B.modify(f, lat, lon, cc, kind, null_centre=B.null_centre_for(*cc), shift_centre=B.shift_centre_for(*cc))
                    fl.append(o); infos.append(info)
                meta['leads'][L][kind] = infos
                ds = B.to_dataset(times, fl, lat, lon, lev, static, n_targets=L // 6, sub=sub)
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    ds, target_lead_times=slice('6h', f'{L}h'), **dataclasses.asdict(task_config))
                pred = rollout.chunked_prediction(fn, rng=jax.random.PRNGKey(0), inputs=inputs,
                                                  targets_template=targets * np.nan, forcings=forcings)
                last = pred.isel(batch=0, time=-1)
                res[kind][L] = extract(last, None, None)
                if cV is not None and kind in ('CTL', 'NOTC', 'WIDE'):
                    pl = last['mean_sea_level_pressure'].values / 100
                    la_, lo_ = last.lat.values, last.lon.values
                    LO2, LA2 = np.meshgrid(lo_, la_)
                    cc_pred = find_centre(pl, LA2, LO2, *cV, search_km=400)
                    meta['leads'][L][kind + '_deficit_at_V'] = plane_deficit(pl, LA2, LO2, *(cc_pred if kind == 'CTL' else cV), 500)
            print(f'case {c.case} lead {L} done', flush=True)
        # truth at V on the model grid
        tds = B.to_dataset([V - pd.Timedelta('6h'), V], [truth_f, truth_f], lat, lon, lev, static, 0, sub=sub).isel(batch=0, time=-1)
        truth = extract(tds, None, None)
        # pack
        Ls = [L for L in leads if isinstance(meta['leads'].get(L), dict)]
        if not Ls:
            json.dump(meta, open(ofile.replace('.nc', '.json'), 'w'), default=float, indent=1); continue
        dv = {}
        for k in truth:
            arr = np.stack([np.stack([res[v][L][k] for L in Ls]) for v in variants])
            dims = ('variant', 'lead', 'station') if k.startswith('stn_') else ('variant', 'lead', 'wlat', 'wlon')
            dv[k] = (dims, arr)
            dv['truth_' + k] = (dims[2:], truth[k])
        wl = tds.sel(**WIN)
        out = xr.Dataset(dv, coords={'variant': variants, 'lead': Ls, 'station': B.STN.index.astype(str).values,
                                     'wlat': wl.lat.values, 'wlon': wl.lon.values})
        out.attrs.update(case=int(c.case), V=c.V, kind=c.kind, tid=int(c.tid), model=a.model)
        out.to_netcdf(ofile)
        json.dump(meta, open(ofile.replace('.nc', '.json'), 'w'), default=float, indent=1)
        print(f'case {c.case} saved in {time.time() - t_start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
