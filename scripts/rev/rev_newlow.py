"""New lows anywhere in the GraphCast window at the case time (independent review, 2026-10-29).

For each saved forecast (out_*/case_XXX.nc, window 95-140E, 5-40N): closed lows are grid points that are the minimum of
sea-level pressure within 300 km and lie at least 2 hPa below the mean of the 300-500 km ring around them.  A low in an
edited run is new when no low of the control run of the same case and lead lies within 400 km.
Usage: python rev_newlow.py out_small_wide out_oper_v3 out_small_rev ...  ->  $TCH/rev/newlows.csv
"""
import os, sys, glob
import numpy as np, pandas as pd, xarray as xr
TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')); OUT = os.path.join(TCH, 'rev'); os.makedirs(OUT, exist_ok=True)
GD = (113.5, 23.0)


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def lows(p, la, lo):
    from scipy.ndimage import minimum_filter
    LO, LA = np.meshgrid(lo, la); out = []
    n = max(1, int(round(300 / 111 / abs(la[1] - la[0]))))
    pf = np.where(np.isfinite(p), p, np.inf)
    cand = np.argwhere(pf == minimum_filter(pf, size=2 * n + 1, mode='nearest'))
    for j, i in cand[np.argsort(pf[tuple(cand.T)])]:
        d = hav(LO, LA, lo[i], la[j])
        if p[j, i] > np.nanmin(p[d <= 300]) + 1e-6:
            continue
        ring = (d > 300) & (d <= 500)
        if ring.sum() < 8:
            continue
        if LA[ring].min() <= la.min() or LA[ring].max() >= la.max() or LO[ring].min() <= lo.min() or LO[ring].max() >= lo.max():
            continue                                         # ring leaves the window
        dep = np.nanmean(p[ring]) - p[j, i]
        if dep >= 2.0 and not any(hav(x, y, lo[i], la[j]) < 300 for x, y, *_ in out):
            out.append((float(lo[i]), float(la[j]), float(p[j, i]), float(dep)))
    return out


rows = []
for dname in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(TCH, dname, 'case_*.nc'))):
        ds = xr.open_dataset(f)
        if ds.attrs.get('kind') != 'heat':
            continue
        la, lo = ds.wlat.values, ds.wlon.values
        for L in ds.lead.values:
            ctl = lows(ds.win_msl.sel(variant='CTL', lead=L).values, la, lo)
            for v in ds.variant.values:
                if v == 'CTL':
                    continue
                for x, y, pmin, dep in lows(ds.win_msl.sel(variant=v, lead=L).values, la, lo):
                    if any(hav(x, y, a_, b_) < 400 for a_, b_, *_ in ctl):
                        continue
                    rows.append(dict(dir=dname, case=int(ds.attrs['case']), lead=int(L), variant=str(v), lon=x, lat=y, pmin=pmin, depth=dep,
                                     dist_gd=float(hav(x, y, *GD))))
            rows.append(dict(dir=dname, case=int(ds.attrs['case']), lead=int(L), variant='_runs', lon=np.nan, lat=np.nan, pmin=np.nan,
                             depth=np.nan, dist_gd=np.nan))
        ds.close()
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, 'newlows.csv'), index=False, float_format='%.4g')
runs = D[D.variant == '_runs'].groupby('dir').size()
nl = D[D.variant != '_runs']
print('runs (case x lead) per directory:', runs.to_dict())
print(nl.assign(near=nl.dist_gd < 1000).groupby(['dir', 'variant']).agg(n_new=('case', 'size'), n_near=('near', 'sum'),
      median_depth=('depth', 'median'), min_dist=('dist_gd', 'min')).to_string())
