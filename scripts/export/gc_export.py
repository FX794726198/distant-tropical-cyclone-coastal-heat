"""Export compact GraphCast results for the paper figures: composites of (CTL - variant) on a 0.5-deg South China window
and per-case / per-lead station tables.

Usage: TCH=/scratch/$USER/tcheat python gc_export.py [--sets small:out_small,out_small_wide oper:out_oper_v3]
Output ($TCH/export/): gcx_maps.npz (int8 maps + scales, agreement in %), gcx_case.csv, gcx_lead.csv
Sign convention: d = CTL - variant, i.e. the effect of the storm being there.  For SHIFTN: NOTC - SHIFT (storm placed SW).
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, xarray as xr

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
GD = (113.5, 23.0)
COAST = ['59493099999', '59501099999', '59663099999', '59673099999', '59658099999']
BOX = dict(lon=(100.0, 135.0), lat=(10.0, 35.0))
FIELDS = ('t2m', 't850', 'q850', 'w500', 'w850', 'u10', 'v10', 'msl', 'tp6', 'z500')
# maps to export: (model, kind, variant, fields, with agreement map)
MAPS = [('oper', 'heat', 'WIDE', ('t2m', 't850', 'q850', 'w500', 'u10', 'v10', 'msl', 'tp6'), ('t2m', 't850', 'w500')),
        ('oper', 'heat', 'NOTC', ('t2m', 'w500', 'tp6'), ('t2m',)),
        ('oper', 'heat', 'NULLW', ('t2m',), ()), ('oper', 'heat', 'NULL', ('t2m',), ()),
        ('oper', 'heat', 'WIDEH', ('t2m',), ()), ('oper', 'heat', 'WEAK', ('t2m',), ()),
        ('oper', 'heat', 'SHIFTN', ('t2m',), ()),
        ('oper', 'placebo', 'WIDE', ('t2m', 'w500', 'u10', 'v10', 'tp6'), ('t2m',)),
        ('oper', 'placebo', 'NOTC', ('t2m',), ()),
        ('small', 'heat', 'WIDE', ('t2m',), ('t2m',)), ('small', 'heat', 'NOTC', ('t2m',), ('t2m',)),
        ('small', 'placebo', 'WIDE', ('t2m',), ()), ('small', 'placebo', 'NOTC', ('t2m',), ())]
CTLMAPS = [('oper', 'heat', ('msl', 'z500')), ('oper', 'placebo', ('msl', 'z500'))]
BIAS = [('oper', 'heat', 't2m'), ('small', 'heat', 't2m')]


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_box(a, wlat, wlon, res):
    """Crop a [..., wlat, wlon] array to BOX and block-average to `res` deg (0.25 -> 0.5); 1-deg data kept as is."""
    i = (wlat >= BOX['lat'][0] - 1e-6) & (wlat <= BOX['lat'][1] + 1e-6)
    j = (wlon >= BOX['lon'][0] - 1e-6) & (wlon <= BOX['lon'][1] + 1e-6)
    a = a[..., i, :][..., j]; la, lo = wlat[i], wlon[j]
    step = int(round(res / abs(la[1] - la[0])))
    if step > 1:
        ny, nx = (len(la) - 1) // step + 1, (len(lo) - 1) // step + 1
        out = np.full(a.shape[:-2] + (ny, nx), np.nan, np.float32)
        for y in range(ny):
            for x in range(nx):
                ys = slice(max(0, y * step - step // 2), y * step + step // 2 + 1)
                xs = slice(max(0, x * step - step // 2), x * step + step // 2 + 1)
                out[..., y, x] = np.nanmean(a[..., ys, xs], axis=(-2, -1))
        a, la, lo = out, la[::step], lo[::step]
    return a, la, lo


def case_files(dirs):
    files = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(TCH, d, 'case_*.nc'))):
            files.setdefault(int(os.path.basename(f)[5:8]), []).append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='*', default=['small:out_small,out_small_wide', 'oper:out_oper_v3'])
    ap.add_argument('--res', type=float, default=0.5)
    a = ap.parse_args()
    out = os.path.join(TCH, 'export'); os.makedirs(out, exist_ok=True)
    crow, lrow, acc, grids = [], [], {}, {}
    for s in a.sets:
        model, dirs = s.split(':'); dirs = dirs.split(',')
        for case, fl in case_files(dirs).items():
            per_var, diffs = {}, {}
            for f in fl:
                ds = xr.open_dataset(f); meta = json.load(open(f.replace('.nc', '.json')))
                kind, tid, V = ds.attrs['kind'], int(ds.attrs['tid']), ds.attrs['V']
                wlat, wlon = ds.wlat.values, ds.wlon.values
                W = {k: ds['win_' + k].values for k in FIELDS if 'win_' + k in ds}
                st = {k: ds['stn_' + k] for k in ('t2m', 't850', 'q850', 'v10', 'u10', 'tp6') if 'stn_' + k in ds}
                LO, LA = np.meshgrid(wlon, wlat)
                near = hav(LO, LA, *GD) <= 600.0
                vs = list(ds.variant.values)
                for iv, v in enumerate(vs):
                    for il, L in enumerate(ds.lead.values):
                        m = W['msl'][iv, il]
                        k = np.nanargmin(np.where(near, m, np.inf))
                        dfc = meta['leads'].get(str(int(L)), {}) if isinstance(meta['leads'].get(str(int(L))), dict) else {}
                        lrow.append(dict(model=model, case=case, kind=kind, tid=tid, V=V, variant=v, lead=int(L),
                                         T=float(st['t2m'][iv, il].mean()), T850=float(st['t850'][iv, il].mean()),
                                         msl600=float(m.flat[k]), msl600_lon=float(LO.flat[k]), msl600_lat=float(LA.flat[k]),
                                         tp6=float(st['tp6'][iv, il].mean()) if 'tp6' in st else np.nan,
                                         deficit_V=dfc.get(f'{v}_deficit_at_V', np.nan),
                                         T_truth=float(ds.truth_stn_t2m.mean())))
                    if v in per_var:
                        continue
                    em = {k: float(st[k][iv].mean()) for k in st}
                    em['onshore'] = float((st['v10'][iv].sel(station=COAST) > 0).mean())
                    em['v10c'] = float(st['v10'][iv].sel(station=COAST).mean())
                    em['u10c'] = float(st['u10'][iv].sel(station=COAST).mean())
                    em['q850'] = em.get('q850', np.nan)
                    per_var[v] = dict(model=model, case=case, kind=kind, tid=tid, V=V, variant=v, nlead=ds.sizes['lead'], **em,
                                      T_truth=float(ds.truth_stn_t2m.mean()))
                # lead-mean maps; differences within the file (its own CTL)
                emW = {k: np.nanmean(x, axis=1) for k, x in W.items()}            # [variant, wlat, wlon]
                ic = vs.index('CTL')
                for iv, v in enumerate(vs):
                    if v == 'CTL':
                        diffs.setdefault('CTL', {k: emW[k][ic] for k in emW})
                        diffs.setdefault('TRUTH', {k: ds['truth_win_' + k].values for k in emW if 'truth_win_' + k in ds})
                    else:
                        diffs.setdefault(v, {k: emW[k][ic] - emW[k][iv] for k in emW})
                if 'NOTC' in vs and 'SHIFT' in vs:
                    diffs['SHIFTN'] = {k: emW[k][vs.index('NOTC')] - emW[k][vs.index('SHIFT')] for k in emW}
            crow += list(per_var.values())
            for v, F in diffs.items():
                for k, x in F.items():
                    b, la, lo = to_box(x, wlat, wlon, a.res if model == 'oper' else 1.0)
                    grids[model] = (la, lo)
                    acc.setdefault((model, kind, v, k), []).append(b)
            print(model, case, kind, sorted(diffs), flush=True)
    C = pd.DataFrame(crow); C.to_csv(os.path.join(out, 'gcx_case.csv'), index=False, float_format='%.4g')
    L = pd.DataFrame(lrow).drop_duplicates(subset=['model', 'case', 'variant', 'lead'])
    L = L[L.variant.isin(['CTL', 'NOTC', 'WIDE', 'NULL', 'NULLW'])]
    L.to_csv(os.path.join(out, 'gcx_lead.csv'), index=False, float_format='%.4g')
    arrs, keys, scales = {}, [], []
    def put(key, x, agree=False):
        if agree:
            q = np.round(np.nan_to_num(x, nan=-1) * 100).astype(np.int8); sc = 0.01
        else:
            sc = float(np.nanmax(np.abs(x))) / 127 or 1.0
            q = np.where(np.isfinite(x), np.round(x / sc), -128).astype(np.int8)
        arrs['m%03d' % len(keys)] = q; keys.append(key); scales.append(sc)
    for model, kind, v, flds, agr in MAPS:
        for k in flds:
            st = acc.get((model, kind, v, k))
            if not st:
                print('missing', model, kind, v, k); continue
            st = np.stack(st)
            put(f'{model}|{kind}|{v}|{k}|mean', np.nanmean(st, 0))
            if k in agr:
                put(f'{model}|{kind}|{v}|{k}|agree', np.mean(st > 0, 0), agree=True)
            print(model, kind, v, k, 'n cases', st.shape[0])
    for model, kind, flds in CTLMAPS:
        for k in flds:
            st = acc.get((model, kind, 'CTL', k))
            if st: put(f'{model}|{kind}|CTL|{k}|mean', np.nanmean(np.stack(st), 0) - (np.nanmean(np.stack(st)) if k in ('msl', 'z500') else 0))
    for model, kind, k in BIAS:
        c, t = acc.get((model, kind, 'CTL', k)), acc.get((model, kind, 'TRUTH', k))
        if c and t: put(f'{model}|{kind}|CTLminusERA5|{k}|mean', np.nanmean(np.stack(c) - np.stack(t), 0))
    for m, (la, lo) in grids.items():
        arrs[f'lat_{m}'] = la.astype(np.float32); arrs[f'lon_{m}'] = lo.astype(np.float32)
    np.savez(os.path.join(out, 'gcx_maps.npz'), keys=np.array(keys), scales=np.array(scales, np.float32), **arrs)
    print('maps', len(keys), 'case rows', len(C), 'lead rows', len(L))


if __name__ == '__main__':
    main()
