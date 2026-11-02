"""Storm removal in GraphCast on the coasts of the global map: GraphCast CTL / NOTC / WIDE / WIDEL / NULLW rollouts for the
cases of gcg_select.py, one region per case.  Edits as in the pilot (gc_build, rev_build): NOTC = core disk infill with
R = min(800, dmin - 150) km; WIDE = NOTC + axisymmetric outer circulation removed to 1,500 km; WIDEL = NOTC + outer removal
ending 50 km before the nearest region cell; NULLW = the WIDE procedure at a storm-free sea point on the far side of the
region.  Lagged ensemble of 24, 36 and 48 h leads, valid at V (14 local time at the region centroid).

Usage (GPU job): python gcg_run.py --model oper|small [--cases 0-63] [--variants CTL,NOTC,WIDE,WIDEL,NULLW] [--leads 24,36,48]
Needs $TCH/inputs/{gc_cases.csv, gc_tracks.csv, gc_others.csv, gcg_regions.json, gd_stations.csv (placeholder for gc_build)},
$TCH/era5/<time>.nc (gc_stage.py), $TCH/gc/{params,stats}; gc_build.py, tcfilter.py, symvortex.py, rev_build.py, gc_run.py
beside this file.
Output: $TCH/out_<model>/case_XXX.nc (cell values and window maps per variant and lead) + case_XXX.json (edit diagnostics).
"""
import argparse, dataclasses, json, os, sys, time
import numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/src/graphcast_v011'))
import jax
from graphcast import data_utils, rollout
import gc_build as B
import rev_build  # noqa: F401  (adds WIDEL to B.modify)
import gc_run as R
import gcg_common as C
from tcfilter import find_centre, plane_deficit

TCH = B.TCH
VARIANTS = ['CTL', 'NOTC', 'WIDE', 'WIDEL', 'NULLW']
LEADS = [24, 36, 48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='oper', choices=list(R.PARAMS))
    ap.add_argument('--cases', default=None)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--leads', default=','.join(map(str, LEADS)))
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    variants = a.variants.split(','); leads = [int(x) for x in a.leads.split(',')]
    cases = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv'))
    tracks = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_tracks.csv'))
    others = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_others.csv'))
    regions = C.load_regions()
    if a.cases:
        lo, hi = map(int, a.cases.split('-')); cases = cases[(cases.case >= lo) & (cases.case <= hi)]
    odir = os.path.join(TCH, 'out_' + a.model + a.tag); os.makedirs(odir, exist_ok=True)
    fn, task_config = R.load_model(a.model)
    static = B.load_static(); sub = R.SUB[a.model]
    s0 = xr.open_dataset(os.path.join(TCH, 'era5', 'static.nc')).sortby('latitude')
    lat_s, lon_s = s0.latitude.values, s0.longitude.values; s0.close()
    lsm = static['land_sea_mask']
    print('devices', jax.devices(), 'cases', list(cases.case), flush=True)
    for _, c in cases.iterrows():
        ofile = os.path.join(odir, f'case_{c.case:03d}.nc')
        if os.path.exists(ofile):
            continue
        reg = regions[c.region]; C.set_region(reg); cen = B.GD
        V = pd.Timestamp(c.V); t_start = time.time()
        truth_f, lat, lon, lev = B.load_time(V)
        LON, LAT = np.meshgrid(lon, lat)
        res = {v: {} for v in variants}
        meta = {'case': int(c.case), 'V': c.V, 'kind': c.kind, 'region': c.region, 'sid': c.sid, 'leads': {}}
        gV = tracks[(tracks.case == c.case) & (tracks.time == V.strftime('%Y-%m-%dT%H'))]
        cV = find_centre(truth_f['mean_sea_level_pressure'], LAT, LON, gV.lon.iloc[0] % 360, gV.lat.iloc[0]) if len(gV) else None
        for L in leads:
            t0 = V - pd.Timedelta(hours=L); times = [t0 - pd.Timedelta('6h'), t0]
            base, cen_t, oth, ok = [], [], [], True
            for t in times:
                ts = t.strftime('%Y-%m-%dT%H')
                g = tracks[(tracks.case == c.case) & (tracks.time == ts)]
                if g.empty: ok = False; break
                f, *_ = B.load_time(t)
                cc = find_centre(f['mean_sea_level_pressure'], LAT, LON, g.lon.iloc[0] % 360, g.lat.iloc[0])
                Rk, dmin = B.radius_for(*cc)
                if Rk < B.R_MIN: ok = False; break
                o = others[(others.case == c.case) & (others.time == ts)]
                base.append(f); cen_t.append(cc); oth.append(list(zip(o.lon.values % 360, o.lat.values)))
            if not ok:
                meta['leads'][L] = 'skipped (storm missing or too close to the region)'; continue
            meta['leads'][L] = {'centres': cen_t}
            for kind in variants:
                fl, infos = [], []
                for f, cc, ot in zip(base, cen_t, oth):
                    nc = C.null_centre_for(cc[0], cc[1], cen, lsm, lat_s, lon_s, other_tc=ot) if kind == 'NULLW' else None
                    o, info = B.modify(f, lat, lon, cc, kind, null_centre=nc)
                    if kind == 'NULLW':
                        info['null_centre'] = nc
                    fl.append(o); infos.append(info)
                meta['leads'][L][kind] = infos
                ds = B.to_dataset(times, fl, lat, lon, lev, static, n_targets=L // 6, sub=sub)
                inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
                    ds, target_lead_times=slice('6h', f'{L}h'), **dataclasses.asdict(task_config))
                pred = rollout.chunked_prediction(fn, rng=jax.random.PRNGKey(0), inputs=inputs,
                                                  targets_template=targets * np.nan, forcings=forcings)
                last = pred.isel(batch=0, time=-1)
                res[kind][L], wlat, wlon = C.extract(last, cen)
                if cV is not None and kind in ('CTL', 'NOTC', 'WIDE', 'WIDEL'):
                    pl = last['mean_sea_level_pressure'].values / 100
                    LO2, LA2 = np.meshgrid(last.lon.values, last.lat.values)
                    cc_pred = find_centre(pl, LA2, LO2, *cV, search_km=400)
                    meta['leads'][L][kind + '_deficit_at_V'] = plane_deficit(pl, LA2, LO2, *(cc_pred if kind == 'CTL' else cV), 500)
            print(f'case {c.case} {c.region} lead {L} done', flush=True)
        tds = B.to_dataset([V - pd.Timedelta('6h'), V], [truth_f, truth_f], lat, lon, lev, static, 0, sub=sub).isel(batch=0, time=-1)
        truth, wlat, wlon = C.extract(tds, cen)
        Ls = [L for L in leads if isinstance(meta['leads'].get(L), dict)]
        if not Ls:
            json.dump(meta, open(ofile.replace('.nc', '.json'), 'w'), default=float, indent=1); continue
        dv = {}
        for k in truth:
            arr = np.stack([np.stack([res[v][L][k] for L in Ls]) for v in variants])
            dims = ('variant', 'lead', 'station') if k.startswith('stn_') else ('variant', 'lead', 'wlat', 'wlon')
            dv[k] = (dims, arr)
            dv['truth_' + k] = (dims[2:], truth[k])
        out = xr.Dataset(dv, coords={'variant': variants, 'lead': Ls, 'station': B.STN.index.values, 'wlat': wlat, 'wlon': wlon})
        out.attrs.update(case=int(c.case), V=c.V, kind=c.kind, region=c.region, sid=c.sid, tid=int(c.tid), model=a.model,
                         lon_c=float(cen[0]), lat_c=float(cen[1]))
        out.to_netcdf(ofile)
        json.dump(meta, open(ofile.replace('.nc', '.json'), 'w'), default=float, indent=1)
        print(f'case {c.case} saved in {time.time() - t_start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
