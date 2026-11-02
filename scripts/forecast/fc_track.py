"""Forecast check, part 2 (exploratory): is the forecast heat tied to forecasting the storm itself?

For every exposure box-day of fc_test.py (warm and cool regions) and lead 1-9 days, the forecast storm is looked for in
the forecast sea-level pressure valid at V: the lowest pressure within 600 km of the IBTrACS position at V, kept as a
storm if it is at least 2 hPa below the mean of the 800-1000 km ring around it.  Position error = distance from the
IBTrACS position.  Composites of the forecast anomaly (as in fc_test.py) are then split by position error.
Usage: python -I fc_track.py MODEL IBTRACS.csv FC_OUT_DIR   (reads FC_OUT_DIR/fc_<MODEL>_days.parquet)
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import gcsfs, zarr, xarray as xr

MODEL, IBT, OUT = sys.argv[1:4]
PATHS = {'hres': 'weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr',
         'pangu': 'weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr'}
fs = gcsfs.GCSFileSystem(token='anon')
D = pd.read_parquet(os.path.join(OUT, f'fc_{MODEL}_days.parquet'))
D = D[D.kind == 'exp'].copy()
D['V'] = pd.to_datetime(D.V)
lines = []


def say(s=''):
    print(s, flush=True); lines.append(str(s))


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# IBTrACS positions at V (6-hourly fixes)
tr = pd.read_csv(IBT, skiprows=[1], low_memory=False, usecols=['SID', 'ISO_TIME', 'LAT', 'LON'])
tr['time'] = pd.to_datetime(tr.ISO_TIME); tr['LAT'] = pd.to_numeric(tr.LAT, errors='coerce'); tr['LON'] = pd.to_numeric(tr.LON, errors='coerce')
tr = tr[tr.time.dt.minute == 0].drop_duplicates(['SID', 'time']).set_index(['SID', 'time'])
pairs = D[['sid', 'V']].drop_duplicates()
pos = tr.reindex(pd.MultiIndex.from_frame(pairs[['sid', 'V']].rename(columns={'sid': 'SID', 'V': 'time'})))[['LON', 'LAT']].values
pairs = pairs.assign(olon=pos[:, 0], olat=pos[:, 1]).dropna()
say(f'{MODEL}: {len(pairs)} storm-times to track')

g = zarr.open_group(fs.get_mapper(PATHS[MODEL]), mode='r'); xd = xr.open_zarr(fs.get_mapper(PATHS[MODEL]), consolidated=True)
z = g['mean_sea_level_pressure']; it = pd.DatetimeIndex(xd.time.values)
lv = xd.prediction_timedelta.values
lt = (lv / np.timedelta64(1, 'h')).astype(int) if lv.dtype.kind == 'm' else lv.astype(int)
LON = np.asarray(g['longitude'][:]); LAT = np.asarray(g['latitude'][:])
LO2, LA2 = np.meshgrid(LON, LAT, indexing='ij')                     # (lon, lat) as the arrays
ipos = pd.Series(np.arange(len(it)), index=it); lpos = pd.Series(np.arange(len(lt)), index=lt)
req = {}
for r in pairs.itertuples():
    h0 = r.V.hour % 12
    for d in range(1, 10):
        L = 24 * d + h0; t0 = r.V - pd.Timedelta(hours=L)
        if L in lpos.index and t0 in ipos.index:
            req.setdefault(int(ipos[t0]), []).append((int(lpos[L]), d, r.sid, r.V, r.olon, r.olat))


def work(i):
    a = np.asarray(z[i]) / 100.0                                      # (lead, lon, lat) hPa
    out = []
    for li, d, sid, V, olon, olat in req[i]:
        f = a[li]
        dd = hav(LO2, LA2, olon, olat)
        m = dd < 600
        if not m.any():
            continue
        k = np.argmin(np.where(m, f, np.inf)); ci, cj = np.unravel_index(k, f.shape)
        d0 = hav(LO2, LA2, LON[ci], LAT[cj]); ring = (d0 > 800) & (d0 < 1000)
        out.append(dict(sid=sid, V=V, lead=d, flon=LON[ci], flat=LAT[cj], err_km=float(dd[ci, cj]), deficit=float(f[ring].mean() - f[ci, cj])))
    return out


t0 = time.time(); rows = []
with ThreadPoolExecutor(24) as ex:
    for o in ex.map(work, sorted(req)):
        rows.extend(o)
T = pd.DataFrame(rows); say(f'tracked {len(T)} storm-time-leads in {time.time() - t0:.0f} s')
T.to_csv(os.path.join(OUT, f'fc_{MODEL}_tracks.csv'), index=False)

# composites by position error
say('\n== forecast storm found (deficit >= 2 hPa) and median position error by lead')
for d in range(1, 10):
    x = T[T.lead == d]
    say(f'lead {d}: found {np.mean(x.deficit >= 2):.2f}, median error {x.err_km[x.deficit >= 2].median():.0f} km')
BINS = [('< 250 km', lambda e, f: (f >= 2) & (e < 250)), ('250-500 km', lambda e, f: (f >= 2) & (e >= 250) & (e < 500)),
        ('>= 500 km or no storm', lambda e, f: (f < 2) | (e >= 500))]
rows = []
for d in range(1, 10):
    x = D.merge(T[T.lead == d][['sid', 'V', 'err_km', 'deficit']], on=['sid', 'V'], how='left')
    x['deficit'] = x.deficit.fillna(0); x['err_km'] = x.err_km.fillna(9999)
    for s in ('warm', 'cool'):
        y = x[x.sign == s]
        for name, fn in BINS:
            q = y[fn(y.err_km, y.deficit)]
            if len(q) < 30:
                continue
            reg = q.groupby(['region', 'box'])[['obs', f'f{d}']].mean().groupby(level=0).mean()
            rows.append(dict(lead=d, sign=s, bin=name, n=len(q), n_regions=len(reg), obs=reg.obs.mean(), fc=reg[f'f{d}'].mean(),
                             frac=reg[f'f{d}'].mean() / reg.obs.mean()))
B = pd.DataFrame(rows); B.to_csv(os.path.join(OUT, f'fc_{MODEL}_by_track.csv'), index=False)
say('\n== exposure-day composites split by forecast storm position error (region means of box means)')
say(B.round(2).to_string(index=False))
open(os.path.join(OUT, f'fc_{MODEL}_track_result.txt'), 'w').write('\n'.join(lines) + '\n')
