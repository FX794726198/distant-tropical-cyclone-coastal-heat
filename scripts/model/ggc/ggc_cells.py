"""Post-hoc, cell-level reading of the global storm-removal runs (written 2026-10-31 08:00Z after the frozen verdict; not part
of it).  Question: is the storm's effect organised in space -- strongest near the coast and nearer the storm -- so that a
region mean over a large region dilutes it?
Readings listed before running: (1) per cell d_free = CTL - WIDEL (mean over leads), d_full, d_null, with the cell's distance to
the nearest sea point (0.25-deg ERA5 land-sea mask < 0.5) and to the storm at V; (2) warm cases: storm-weighted mean d_free by
distance to coast (0-50, 50-150, 150-300, > 300 km) and by distance to storm (550-800, 800-1000, 1000-1300 km); (3) per region
the mean over its coastal strip (<= 150 km) beside the region mean, and the number of usable warm regions whose strip mean
>= 0.5 K; (4) the same for cool regions (mirror); (5) both models.  No GPU, reads the existing case_*.nc only.
Usage (login node): ~/venvs/gc/bin/python -I ggc_cells.py ROOT   -> ROOT/cells/cells_<model>.csv, ROOT/cells/cells_summary.txt
"""
import glob, json, os, sys
import numpy as np, pandas as pd, xarray as xr

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, 'cells'); os.makedirs(OUT, exist_ok=True)
R = 6371.0


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


st = xr.open_dataset(os.path.join(ROOT, 'era5', 'static.nc'))
lname = [v for v in st.data_vars if 'land' in v.lower()][0]
lsm = st[lname].squeeze().values; lat_s = st['latitude'].values; lon_s = st['longitude'].values
if lsm.shape != (len(lat_s), len(lon_s)):
    lsm = lsm.T
LO, LA = np.meshgrid(lon_s, lat_s)
sea_lon, sea_lat = LO[lsm < 0.5], LA[lsm < 0.5]
print('static', lname, lsm.shape, 'sea points', len(sea_lon), flush=True)

regions = json.load(open(os.path.join(ROOT, 'inputs', 'gcg_regions.json')))
tracks = pd.read_csv(os.path.join(ROOT, 'inputs', 'gc_tracks.csv'))
cases = pd.read_csv(os.path.join(ROOT, 'inputs', 'gc_cases.csv')).set_index('case')
coast = {}
for g, reg in regions.items():
    for k, (x, y) in enumerate(reg['cells']):
        x = x % 360
        m = (np.abs(sea_lat - y) <= 12) & (np.abs(((sea_lon - x + 180) % 360) - 180) <= 12)
        coast[(g, k)] = float(hav(x, y, sea_lon[m], sea_lat[m]).min()) if m.any() else np.nan
print('coast distances done', flush=True)

L = []
for model in ('oper', 'small'):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, f'out_{model}', 'case_*.nc'))):
        ds = xr.open_dataset(f); c = int(ds.attrs['case']); g = ds.attrs['region']; V = ds.attrs['V']
        reg = regions[g]; em = ds.stn_t2m.mean('lead')
        t = tracks[(tracks.case == c) & (tracks.time == V)]
        slon, slat = (float(t.lon.iloc[0]) % 360, float(t.lat.iloc[0])) if len(t) else (np.nan, np.nan)
        for k, (x, y) in enumerate(reg['cells']):
            r = dict(model=model, case=c, region=g, kind=ds.attrs['kind'], sid=ds.attrs['sid'], cell=k, lon=x, lat=y,
                     d_coast=coast[(g, k)], d_storm=float(hav(x % 360, y, slon, slat)), anom=float(cases.anom.loc[c]))
            T = {v: float(em.sel(variant=v).isel(station=k)) for v in ds.variant.values}
            r['d_free'] = T['CTL'] - T['WIDEL']; r['d_full'] = T['CTL'] - T['WIDE']; r['d_core'] = T['CTL'] - T['NOTC']
            r['d_null'] = T['CTL'] - T['NULLW']; r['ctl_err'] = T['CTL'] - float(ds.truth_stn_t2m.isel(station=k))
            rows.append(r)
        ds.close()
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, f'cells_{model}.csv'), index=False, float_format='%.4f')
    D['w'] = 1.0 / D.groupby(['region', 'sid']).sid.transform('size')          # each storm weight 1 within a region
    L.append(f'\n================ {model}: {len(D)} case-cells, {D.case.nunique()} cases')
    for kind in ('warm', 'cool'):
        W = D[D.kind == kind]
        usable = W.groupby('region').case.nunique(); usable = usable[usable >= 3].index
        W = W[W.region.isin(usable)]
        L.append(f'--- {kind} cases in {len(usable)} usable regions: {len(W)} case-cells; mean d_coast {W.d_coast.mean():.0f} km')
        for name, edges in (('distance to coast', [0, 50, 150, 300, 1e9]), ('distance to storm', [0, 550, 800, 1000, 1300, 1e9])):
            col = 'd_coast' if name.endswith('coast') else 'd_storm'
            bins = pd.cut(W[col], edges, right=False)
            for b, d in W.groupby(bins, observed=True):
                if len(d) == 0: continue
                wm = lambda s: float((s * d.w).sum() / d.w.sum())
                L.append(f'  {name} {str(b):>14}: {len(d):4d} cells  d_free {wm(d.d_free):+.2f}  d_full {wm(d.d_full):+.2f}  d_null {wm(d.d_null):+.2f}  '
                         f'ctl_err {wm(d.ctl_err):+.2f}  share(d_free >= 0.5) {float((d.d_free >= 0.5).mean()):.0%}')
        rr = []
        for g, d in W.groupby('region'):
            wm = lambda s, dd=d: float((s * dd.w).sum() / dd.w.sum())
            strip = d[d.d_coast <= 150]
            rr.append(dict(region=g, ncell=d.cell.nunique(), n_strip=strip.cell.nunique(), mean=wm(d.d_free), null=wm(d.d_null),
                           strip=float((strip.d_free * strip.w).sum() / strip.w.sum()) if len(strip) else np.nan,
                           strip_null=float((strip.d_null * strip.w).sum() / strip.w.sum()) if len(strip) else np.nan,
                           inland=float(((d[d.d_coast > 150].d_free) * d[d.d_coast > 150].w).sum() / d[d.d_coast > 150].w.sum()) if (d.d_coast > 150).any() else np.nan,
                           frac_cells_05=float((d.groupby('cell').d_free.mean() >= 0.5).mean())))
        RR = pd.DataFrame(rr).set_index('region'); RR.to_csv(os.path.join(OUT, f'strip_{model}_{kind}.csv'), float_format='%.3f')
        L.append(RR.round(2).to_string())
        thr = 0.5 if kind == 'warm' else -0.3
        ok = (RR.strip >= thr) if kind == 'warm' else (RR.strip <= thr)
        okm = (RR['mean'] >= thr) if kind == 'warm' else (RR['mean'] <= thr)
        L.append(f'  regions with region mean {"≥" if kind == "warm" else "≤"} {thr}: {int(okm.sum())}/{len(RR)}; with coastal-strip mean: {int(ok.sum())}/{int(RR.strip.notna().sum())}; '
                 f'corr(ncell, mean) {np.corrcoef(RR.ncell, RR["mean"])[0, 1]:+.2f}; strip minus inland, mean over regions {float((RR.strip - RR.inland).mean()):+.2f} K')
open(os.path.join(OUT, 'cells_summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
