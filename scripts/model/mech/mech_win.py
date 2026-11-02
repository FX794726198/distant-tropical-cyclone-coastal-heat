"""A4 window fields from the A2 case files (exploratory plan plan A4 (internal), section 4: composites).

For every case: CTL minus WIDEL and CTL itself at V (14 local time), mean over the leads present, for the window fields
t2m, t850, w500, msl, u10, v10 (wlat x wlon around the region).  The 0.25-degree model ('oper') is also block-averaged to
1 degree (4 x 4), so that both models can be stored per case and composited later in any frame (region-fixed or relative
to the storm).  Per region, a native-resolution composite (mean over storms of each storm's case mean) and the share of
cases that agree in sign with it.

Usage: python mech_win.py CASEDIR MODEL OUT.npz        (MODEL = oper or small)"""
import glob, os, sys
from collections import Counter
import numpy as np, xarray as xr

D, MODEL, OUT = sys.argv[1:4]
VARS = ['t2m', 't850', 'w500', 'msl', 'u10', 'v10']
K = 4 if MODEL == 'oper' else 1


def block(a, k):
    if k == 1:
        return a
    ny, nx = a.shape[-2] // k * k, a.shape[-1] // k * k
    a = a[..., :ny, :nx]
    return a.reshape(a.shape[:-2] + (ny // k, k, nx // k, k)).mean(axis=(-3, -1))


meta, dl, cl, lats, lons, nat = [], [], [], [], [], {}
for f in sorted(glob.glob(os.path.join(D, 'case_*.nc'))):
    with xr.open_dataset(f) as ds:
        if 'WIDEL' not in ds.variant.values or 'CTL' not in ds.variant.values:
            continue
        if not all(f'win_{k}' in ds for k in VARS):
            print('missing window fields in', os.path.basename(f)); continue
        r = ds.attrs['region']
        c = np.stack([ds[f'win_{k}'].sel(variant='CTL').transpose('lead', 'wlat', 'wlon').values for k in VARS])
        w = np.stack([ds[f'win_{k}'].sel(variant='WIDEL').transpose('lead', 'wlat', 'wlon').values for k in VARS])
        d = np.nanmean(c - w, axis=1).astype('float32'); c = np.nanmean(c, axis=1).astype('float32')
        la, lo = ds.wlat.values.astype('float64'), ds.wlon.values.astype('float64')
        meta.append((int(ds.attrs['case']), r, ds.attrs['kind'], ds.attrs['sid'], str(ds.attrs.get('V', '')),
                     float(ds.attrs.get('lon_c', np.nan)), float(ds.attrs.get('lat_c', np.nan))))
        dl.append(block(d, K)); cl.append(block(c, K))
        lats.append(la[:la.size // K * K].reshape(-1, K).mean(1)); lons.append(lo[:lo.size // K * K].reshape(-1, K).mean(1))
        nat.setdefault(r, []).append((ds.attrs['sid'], d, la, lo))

shp = Counter(a.shape for a in dl).most_common(1)[0][0]
keep = [i for i, a in enumerate(dl) if a.shape == shp]
if len(keep) < len(dl):
    print(f'{len(dl) - len(keep)} cases with another window shape left out of the per-case arrays')
out = dict(case=np.array([meta[i][0] for i in keep]), region=np.array([meta[i][1] for i in keep]),
           kind=np.array([meta[i][2] for i in keep]), sid=np.array([meta[i][3] for i in keep]),
           V=np.array([meta[i][4] for i in keep]), lon_c=np.array([meta[i][5] for i in keep]),
           lat_c=np.array([meta[i][6] for i in keep]), vars=np.array(VARS),
           d=np.stack([dl[i] for i in keep]), ctl=np.stack([cl[i] for i in keep]),
           lat=np.stack([lats[i] for i in keep]), lon=np.stack([lons[i] for i in keep]))
for r, items in nat.items():
    key = Counter((len(la), len(lo), round(la[0], 3), round(lo[0], 3)) for _, _, la, lo in items).most_common(1)[0][0]
    it = [x for x in items if (len(x[2]), len(x[3]), round(x[2][0], 3), round(x[3][0], 3)) == key]
    sids = sorted({x[0] for x in it})
    per_storm = np.stack([np.nanmean(np.stack([x[1] for x in it if x[0] == s]), axis=0) for s in sids])
    comp = np.nanmean(per_storm, axis=0)
    agree = np.mean(np.sign(np.stack([x[1] for x in it])) == np.sign(comp)[None], axis=0)
    out[f'{r}__comp'] = comp.astype('float32'); out[f'{r}__agree'] = agree.astype('float16')
    out[f'{r}__lat'] = it[0][2]; out[f'{r}__lon'] = it[0][3]
    out[f'{r}__n'] = np.array([len(it), len(sids), len(items) - len(it)])
    print(f'{r}: {len(it)} cases, {len(sids)} storms, window {key[0]} x {key[1]} from ({key[2]}, {key[3]})'
          + (f', {len(items) - len(it)} cases on another window left out' if len(items) > len(it) else ''))
np.savez_compressed(OUT, **out)
print(f'saved {OUT}: {len(keep)} cases, per-case arrays {out["d"].shape} (vars {VARS}, block {K})')
