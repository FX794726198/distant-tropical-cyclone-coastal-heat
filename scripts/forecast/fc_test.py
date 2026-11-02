"""Forecast check (exploratory): do weather forecasts see the coastal heat of distant storms,
and how many days ahead?

Same definitions as the global map (gl_map.py) and the station check, on the 1.5-degree WeatherBench2 grid:
  - region boxes: the 1.5-degree boxes holding the map's 1-degree region cells (gcg_regions.json; land fraction >= 0.2);
  - value: 2-m temperature at the 6-hourly time closest to 14 local time (as the GraphCast case time V);
  - exposure code per box and local solar day: nearest tropical-storm fix (IBTrACS, NATURE TS, >= 34 kt, 6-hourly)
    300-1300 km away -> sector 0..7 of its bearing; -1 reference (none within 1300 km); -2 one within 300 km;
  - anomalies: 3 harmonics + trend removed, then each year's mean and the +-15-day calendar mean of the in-season
    reference days removed; done separately for ERA5 and for the forecast at each lead, so each lead is measured
    against its own storm-free days (removes lead-dependent drift);
  - exposure days: in the region's months with the storm in the region's sector.
Forecast at lead d days: initialised 24 d + h0 hours before V (h0 = 0 or 6 h so that the start is 00 or 12 UTC).
Composite per region = mean over boxes of the mean over that box's exposure days (common sample over leads 1-9);
pooled = mean over warm (cool) regions; 90% intervals from 1,000 bootstrap draws of storms.
Also: hot exposure days (ERA5 anomaly >= the box's in-season P90) and how often the forecast also put the day above
its own P90 at that lead, against the same for hot reference days.

Usage: python -I fc_test.py MODEL(hres|pangu) IBTRACS.csv gcg_regions.json gl_regions.csv OUT_DIR
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import gcsfs, zarr, xarray as xr

MODEL, IBT, RJ, GLR, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
PATHS = {'hres': ('weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr', 2016, 2022),
         'pangu': ('weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr', 2018, 2022)}
ERA5 = 'weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
FPATH, Y0, Y1 = PATHS[MODEL]
LEADS_D = list(range(0, 10))
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
rng = np.random.default_rng(1)
fs = gcsfs.GCSFileSystem(token='anon')
lines = []


def say(s=''):
    print(s, flush=True); lines.append(str(s))


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ---- grid, regions, boxes
eg = zarr.open_group(fs.get_mapper(ERA5), mode='r')
LON, LAT = np.asarray(eg['longitude'][:]), np.asarray(eg['latitude'][:])
lsm = np.asarray(eg['land_sea_mask'][:]) if eg['land_sea_mask'].ndim == 2 else np.asarray(eg['land_sea_mask'][0])
R = json.load(open(RJ)); G = pd.read_csv(GLR)
boxes = {}                                     # (i_lon, j_lat) -> info
for k, v in R.items():
    j = ((G.lon - v['lon']).abs() + (G.lat - v['lat']).abs()).idxmin(); v['months'] = G.loc[j, 'months']
    cl = np.array(v['cells'], float)
    ii = np.rint((cl[:, 0] % 360) / 1.5).astype(int) % len(LON); jj = np.rint((cl[:, 1] + 90) / 1.5).astype(int)
    bx = sorted(set(zip(ii, jj)))
    keep = [b for b in bx if lsm[b] >= 0.2] or [max(bx, key=lambda b: lsm[b])]
    v['boxes'] = keep
    for b in keep:
        boxes.setdefault(b, []).append(k)
B = sorted(boxes); NB = len(B)
bi = np.array([b[0] for b in B]); bj = np.array([b[1] for b in B])
blon = ((LON[bi] + 180) % 360) - 180; blat = LAT[bj]
say(f'{MODEL}: {len(R)} regions, {NB} boxes, years {Y0}-{Y1}')

# ---- days and the case time per box (6-hourly UTC closest to 14 LT)
days = pd.date_range(f'{Y0}-01-01', f'{Y1}-12-31', freq='D'); ND = len(days)
vh = (np.rint((14.0 - blon / 15.0) / 6.0) * 6).astype(int)        # hours after 00 UTC of the local date (can be 24 or -6)
Vt = days.values[:, None] + vh[None, :].astype('timedelta64[h]')  # [day, box]

# ---- exposure codes (nearest TS fix over the box's local solar day)
tr = pd.read_csv(IBT, skiprows=[1], low_memory=False, usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'NATURE'])
for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND'):
    tr[c] = pd.to_numeric(tr[c], errors='coerce')
tr['time'] = pd.to_datetime(tr.ISO_TIME); tr['wind'] = tr.USA_WIND.fillna(tr.WMO_WIND)
tr = tr[tr.time.dt.hour.isin([0, 6, 12, 18]) & (tr.time.dt.minute == 0) & (tr.NATURE.str.strip() == 'TS') & (tr.wind >= 34)].dropna(subset=['LAT', 'LON'])
tr = tr[tr.time.dt.year.between(Y0 - 1, Y1 + 1)]
sids = pd.Index(tr.SID.unique()); tr['k'] = sids.get_indexer(tr.SID)
DM = np.full((ND, NB), np.inf, np.float32); BR = np.full((ND, NB), np.nan, np.float32); FK = np.full((ND, NB), -1, np.int32)
di = pd.Series(np.arange(ND), index=days)
tt = tr.time.values; tlon, tlat, tk = tr.LON.values, tr.LAT.values, tr.k.values
lday = (tr.time + pd.to_timedelta(tr.LON / 15.0, unit='h')).dt.normalize().values   # local day of the fix, as gl_map.py
lday = pd.DatetimeIndex(lday); ok = lday.isin(days)
for b in range(NB):
    d = hav(blon[b], blat[b], tlon[ok], tlat[ok])
    s = pd.DataFrame({'i': di[lday[ok]].values, 'd': d, 'k': tk[ok], 'lo': tlon[ok], 'la': tlat[ok]})
    s = s.loc[s.groupby('i').d.idxmin()]
    DM[s.i, b] = s.d; FK[s.i, b] = s.k; BR[s.i, b] = bearing(blon[b], blat[b], s.lo.values, s.la.values)
code = np.where(DM > 1300, -1, np.where(DM < 300, -2, np.floor(((np.nan_to_num(BR) + 22.5) % 360) / 45).astype(int)))
say(f'exposure codes: {len(sids)} storms; box-days with a storm 300-1300 km: {((code >= 0)).sum()}')


# ---- data: ERA5 at V, forecast at V for each lead
def read_era5():
    z = eg['2m_temperature']
    t0 = pd.Timestamp(xr.open_zarr(fs.get_mapper(ERA5), consolidated=True).time.values[0])   # decoded coordinate
    assert t0 == pd.Timestamp('1959-01-01'), t0
    idx = ((Vt - np.datetime64(t0)) / np.timedelta64(6, 'h')).astype(int)
    lo, hi = idx.min(), idx.max() + 1
    ch = z.chunks[0]; blocks = range(lo // ch * ch, hi, ch)
    def rd(s):
        a = np.asarray(z[s:s + ch]); return s, a[:, bi, bj]
    out = {}
    with ThreadPoolExecutor(24) as ex:
        for s, a in ex.map(rd, blocks):
            for r in range(a.shape[0]):
                out[s + r] = a[r]
    X = np.full((ND, NB), np.nan, np.float32)
    for b in range(NB):
        X[:, b] = [out[i][b] if i in out else np.nan for i in idx[:, b]]
    return X


def read_fc():
    g = zarr.open_group(fs.get_mapper(FPATH), mode='r')
    z = g['2m_temperature']; xd = xr.open_zarr(fs.get_mapper(FPATH), consolidated=True)
    it = pd.DatetimeIndex(xd.time.values)
    lv = xd.prediction_timedelta.values
    lt = (lv / np.timedelta64(1, 'h')).astype(int) if lv.dtype.kind == 'm' else lv.astype(int)   # stored as hours
    assert len(it) == z.shape[0] and len(lt) == z.shape[1]
    dims = z.attrs.get('_ARRAY_DIMENSIONS', ['time', 'prediction_timedelta', 'longitude', 'latitude'])
    lonfirst = dims.index('longitude') < dims.index('latitude')
    def rd(i):
        a = np.asarray(z[i]); return i, (a[:, bi, bj] if lonfirst else a[:, bj, bi])
    t0 = time.time(); F = {}
    with ThreadPoolExecutor(32) as ex:
        for i, a in ex.map(rd, range(len(it))):
            F[i] = a
    say(f'read {len(it)} forecast starts in {time.time() - t0:.0f} s')
    ipos = pd.Series(np.arange(len(it)), index=it); lpos = pd.Series(np.arange(len(lt)), index=lt)
    X = np.full((len(LEADS_D), ND, NB), np.nan, np.float32)
    for b in range(NB):
        h0 = vh[b] % 12
        for q, d in enumerate(LEADS_D):
            L = 24 * d + h0
            if L not in lpos.index:
                continue
            st = pd.DatetimeIndex(Vt[:, b] - np.timedelta64(L, 'h'))
            m = st.isin(ipos.index)
            X[q, m, b] = [F[i][lpos[L], b] for i in ipos[st[m]].values]
    return X


def anomalies(X, code_b, seas):
    ok = np.isfinite(X)
    if ok.sum() < 700:
        return np.full(ND, np.nan)
    t = (days.year.values - 2019.5) / 10.0; w = 2 * np.pi * days.dayofyear.values / 365.25
    D = np.stack([np.ones_like(t), t] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)], 1)
    Bc = np.linalg.lstsq(D[ok], X[ok], rcond=None)[0]
    A = X - D @ Bc
    ref = seas & (code_b == -1) & ok
    yi = days.year.values - Y0
    for y in range(Y1 - Y0 + 1):
        m = yi == y
        if ref[m].sum() > 0:
            A[m] -= A[m][ref[m]].mean()
    doy = days.dayofyear.values
    S = np.bincount(doy[ref] - 1, A[ref], minlength=366); C = np.bincount(doy[ref] - 1, minlength=366).astype(float)
    Sw = sum(np.roll(S, r) for r in range(-15, 16)); Cw = sum(np.roll(C, r) for r in range(-15, 16))
    A = A - (Sw / np.maximum(Cw, 1))[doy - 1]
    A[~seas | ~ok] = np.nan
    return A


cache = os.path.join(OUT, f'fc_{MODEL}_raw.npz')
if os.path.exists(cache):
    z = np.load(cache); E, F = z['E'], z['F']
else:
    E = read_era5(); F = read_fc(); np.savez_compressed(cache, E=E, F=F)
say(f'ERA5 finite {np.isfinite(E).mean():.3f}; forecast finite by lead ' + ' '.join(f'{np.isfinite(F[q]).mean():.2f}' for q in range(len(LEADS_D))))

# ---- per region: anomalies per box (region's months), composites
mon = days.month.values
recs, rows = [], []
for k, v in R.items():
    seas = np.array([v['months'][m - 1] == 'x' for m in mon])
    sec = SECT.index(v['sector'])
    for b in v['boxes']:
        q = B.index(b)
        cb = code[:, q]
        Ao = anomalies(E[:, q].astype(float), cb, seas)
        Af = np.stack([anomalies(F[l, :, q].astype(float), cb, seas) for l in range(len(LEADS_D))])
        for kind, sel in (('exp', seas & (cb == sec)), ('ref', seas & (cb == -1))):
            m = sel & np.isfinite(Ao) & np.isfinite(Af[1:]).all(0)       # lead 0 may be missing (Pangu starts at 6 h)
            if m.sum() == 0:
                continue
            p90o = np.nanpercentile(Ao[seas], 90); p90f = np.nanpercentile(Af[:, seas], 90, axis=1)
            for i in np.where(m)[0]:
                recs.append([k, v['sign'], q, kind, int(FK[i, q]), sids[FK[i, q]] if FK[i, q] >= 0 else '', days[i].year,
                             pd.Timestamp(Vt[i, q]), float(DM[i, q]), Ao[i], *Af[:, i], Ao[i] >= p90o, *(Af[:, i] >= p90f)])
cols = ['region', 'sign', 'box', 'kind', 'storm', 'sid', 'year', 'V', 'dist', 'obs'] + [f'f{d}' for d in LEADS_D] + ['hot_obs'] + [f'hot_f{d}' for d in LEADS_D]
D_ = pd.DataFrame(recs, columns=cols)
D_.to_parquet(os.path.join(OUT, f'fc_{MODEL}_days.parquet'))
FC = [f'f{d}' for d in LEADS_D]


def region_comp(d):
    """mean over boxes of box means."""
    return d.groupby('box')[['obs'] + FC].mean().mean()


tab = []
for (k, s), g in D_[D_.kind == 'exp'].groupby(['region', 'sign']):
    c = region_comp(g); tab.append(dict(region=k, sign=s, n_days=len(g), n_storms=g.storm.nunique(), n_box=g.box.nunique(), **c.to_dict()))
T = pd.DataFrame(tab)
T.to_csv(os.path.join(OUT, f'fc_{MODEL}_regions.csv'), index=False)
say('\n== exposure-day composites by region (K): ERA5 and forecast at lead 0..9 days')
say(T.round(2).to_string(index=False))

# pooled + storm bootstrap
EXP = D_[D_.kind == 'exp'].copy()
st_ids = EXP.storm.unique(); sidx = pd.Index(st_ids)
EXP['si'] = sidx.get_indexer(EXP.storm)
grp = EXP.groupby(['sign', 'region', 'box', 'si'])
S = grp[['obs'] + FC].sum(); N = grp.size()


def pooled(wts):
    w = wts[S.index.get_level_values('si')]
    Sw = S.mul(w, axis=0).groupby(level=['sign', 'region', 'box']).sum(); Nw = (N * w).groupby(level=['sign', 'region', 'box']).sum()
    bm = Sw.div(Nw.replace(0, np.nan), axis=0)
    return bm.groupby(level=['sign', 'region']).mean().groupby(level='sign').mean()


base = pooled(np.ones(len(sidx)))
boot = [pooled(rng.multinomial(len(sidx), np.full(len(sidx), 1 / len(sidx))).astype(float)) for _ in range(1000)]
say('\n== pooled exposure-day composite (mean over regions), 90% storm-bootstrap interval')
out = []
for s in ('warm', 'cool'):
    if s not in base.index:
        continue
    for c in ['obs'] + FC:
        bs = np.array([b.loc[s, c] for b in boot if s in b.index])
        fr = np.array([b.loc[s, c] / b.loc[s, 'obs'] for b in boot if s in b.index])
        out.append(dict(sign=s, series=c, value=base.loc[s, c], lo=np.percentile(bs, 5), hi=np.percentile(bs, 95),
                        frac_of_obs=base.loc[s, c] / base.loc[s, 'obs'], frac_lo=np.percentile(fr, 5), frac_hi=np.percentile(fr, 95)))
P = pd.DataFrame(out); P.to_csv(os.path.join(OUT, f'fc_{MODEL}_pooled.csv'), index=False)
say(P.round(3).to_string(index=False))

# forecast error on exposure vs reference days, and hot-day hits
say('\n== mean error (forecast - ERA5 anomaly) and hot-day hit rate by lead; region-mean of box means')
er = []
for kind in ('exp', 'ref'):
    g = D_[D_.kind == kind]
    for s in ('warm', 'cool'):
        h = g[g.sign == s]
        if h.empty:
            continue
        e = {d: (h[f'f{d}'] - h.obs).groupby([h.region, h.box]).mean().groupby(level=0).mean().mean() for d in LEADS_D}
        hh = h[h.hot_obs]
        hit = {d: hh[f'hot_f{d}'].groupby([hh.region, hh.box]).mean().groupby(level=0).mean().mean() for d in LEADS_D}
        er.append(dict(kind=kind, sign=s, n_days=len(h), n_hot=len(hh), **{f'err{d}': e[d] for d in LEADS_D}, **{f'hit{d}': hit[d] for d in LEADS_D}))
ER = pd.DataFrame(er); ER.to_csv(os.path.join(OUT, f'fc_{MODEL}_error_hits.csv'), index=False)
say(ER.round(3).to_string(index=False))
open(os.path.join(OUT, f'fc_{MODEL}_result.txt'), 'w').write('\n'.join(lines) + '\n')
