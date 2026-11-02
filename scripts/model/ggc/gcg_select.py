"""Global storm-removal test (analysis A, frozen 2026-10-30): choose the GraphCast cases for every region of the global map.
Design and reading rules: analysis plan A (internal, written before the runs).  Rules in this docstring were fixed before any case was run.

Regions: the 23 warm and 18 cool regions of gl_regions.csv (gl_map.regions clusters, rebuilt here as in gl_stage2.region_keys).
Candidate days: local days of YEARS (default 2016-2021) in which at least half of the region's cells are in their storm
  season and at least half have exposure code == the region's sector (nearest storm 300-1,300 km away, in that sector).
The storm: of the storms with a NATURE-TS, >= 34 kt fix at V, the one nearest the region centroid at V.
V: the 6-hourly UTC time nearest 14:00 local solar time at the region centroid, on that local day.
Usable: fixes exist at every 6-hourly time from V-54 h to V; the storm is 550-1,300 km from the nearest region cell at V and
  >= 550 km at all those times (core radius min(800, dmin-150) >= 400 km, as in the pilot); no other TS storm within
  1,300 km of the centroid at V; a storm-free sea point for the null edit exists (checked again at run time).
Ranking: region-mean matched daily-maximum anomaly (gl_map.matched, the map's own anomaly).  Warm regions take the NWARM
  warmest usable days, cool regions the NCOOL coolest, at most 2 days per storm in a region.
Usage: python -I gcg_select.py OUT_DIR [--years 2016-2021] [--nwarm 4] [--ncool 3]
Needs $TCH/global/{era5_t2_YYYY.npz (every year of the map, for the anomalies), static_1deg.npz, ibtracs csv,
  out/gl_cells.csv, out/gl_codes.npz, out/gl_regions.csv} and gl_map.py in $GLS.
Output (OUT_DIR): gc_cases.csv, gc_tracks.csv, gc_others.csv, gcg_regions.json, gcg_candidates.csv, gcg_select_log.txt
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy import ndimage

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
GL = os.path.join(TCH, 'global')
OUT = os.path.join(GL, 'out')
GLS = os.environ.get('GLS', os.path.join(TCH, 'scripts', 'global'))
sys.path.insert(0, GLS)
from gl_map import hav, bearing, load_tracks, season_mask, harmonic_anomalies, matched, SECT  # noqa: E402

DMIN_OK, DMAX_V, OTHER_KM, PRE_H = 550.0, 1300.0, 1300.0, 54


def region_keys(df, col_ex, col_sig, sign, nlon, nlat):
    """The clusters of gl_map.regions(), in the same order: lists of (i, j)."""
    G = np.zeros((nlon, nlat), bool)
    sel = df[(sign * df[col_ex] >= 0.5) & df[col_sig]]
    G[sel.i.values, sel.j.values] = True
    lab, n = ndimage.label(np.concatenate([G, G], 0), structure=np.ones((3, 3)))
    out, seen = [], set()
    for r in range(1, n + 1):
        ii, jj = np.where(lab == r)
        if ii.min() >= nlon:
            continue
        key = tuple(sorted(set(zip(ii % nlon, jj))))
        if len(key) < 3 or key in seen:
            continue
        seen.add(key); out.append(key)
    return out


def v_time(day, lon_c):
    """6-hourly UTC time nearest 14:00 local solar time at longitude lon_c on the local day `day`."""
    h = 14.0 - lon_c / 15.0
    k = int(np.round(h / 6.0))
    return pd.Timestamp(day) + pd.Timedelta(hours=6 * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--years', default='2016-2021')
    ap.add_argument('--nwarm', type=int, default=4); ap.add_argument('--ncool', type=int, default=3)
    a = ap.parse_args()
    ys, ye = map(int, a.years.split('-'))
    os.makedirs(a.out, exist_ok=True)
    T0 = time.time()
    L = [f'gcg_select.py md5 {hashlib.md5(open(__file__, "rb").read()).hexdigest()}, years {a.years}, nwarm {a.nwarm}, ncool {a.ncool}']

    def log(s):
        L.append(s); print(s, flush=True)

    cells = pd.read_csv(os.path.join(OUT, 'gl_cells.csv'))
    for c in ('valid', 'sig_best', 'sig_worst'):
        cells[c] = cells[c].astype(str).str.lower().eq('true')
    G = np.load(os.path.join(OUT, 'gl_codes.npz'))
    code, seas_m, days = G['code'], G['seas_m'], pd.DatetimeIndex(G['days'])
    assert len(cells) == code.shape[1] and (cells.i.values == G['ci']).all() and (cells.j.values == G['cj']).all()
    st = np.load(os.path.join(GL, 'static_1deg.npz'))
    nlon, nlat = len(st['lon']), len(st['lat'])
    RG = pd.read_csv(os.path.join(OUT, 'gl_regions.csv'))
    rowof = {(i, j): r for r, (i, j) in enumerate(zip(cells.i.values, cells.j.values))}
    v = cells[cells.valid]
    REG = []
    for sign, col in ((1, 'best'), (-1, 'worst')):
        keys = region_keys(v, f'ex_{col}', f'sig_{col}', sign, nlon, nlat)
        rg = RG[RG.sign == ('warm' if sign > 0 else 'cool')].reset_index(drop=True)
        assert len(keys) == len(rg), (col, len(keys), len(rg))
        for k, key in enumerate(keys):
            rows = np.array([rowof[x] for x in key])
            assert len(rows) == rg.n_cells[k]
            sec = cells[f'sec_{col}'].values[rows]
            assert pd.Series(sec).mode().iloc[0] == rg.sector[k]
            REG.append(dict(region=f"{'W' if sign > 0 else 'C'}{k + 1}", sign='warm' if sign > 0 else 'cool', rows=rows,
                            sector=rg.sector[k], s=SECT.index(rg.sector[k]), ex_map=float(rg.ex_mean[k])))
    log(f'{len(REG)} regions rebuilt from gl_regions.csv, {time.time() - T0:.0f} s')

    # matched anomalies at the region cells (the map's definition), all years of the map
    need = np.unique(np.concatenate([r['rows'] for r in REG]))
    col = {c: n for n, c in enumerate(need)}
    X = np.full((len(days), len(need)), np.nan, np.float32)
    for y in np.unique(days.year):
        d = np.load(os.path.join(GL, f'era5_t2_{y}.npz'))
        pos = pd.DatetimeIndex(d['days']).get_indexer(days[days.year == y])
        assert (pos >= 0).all(), y
        X[days.year == y] = d['tmax'][pos][:, cells.i.values[need], cells.j.values[need]]
    y0 = int(days.year.min()); ny = int(days.year.max()) - y0 + 1
    yidx = days.year.values - y0; doy = days.dayofyear.values
    A = harmonic_anomalies(X, days); del X
    seas_d, _ = season_mask(code, days)
    B = matched(A, code[:, need], seas_d[:, need], yidx, ny, doy); del A
    log(f'anomalies at {len(need)} region cells, {time.time() - T0:.0f} s')

    # storms: TS fixes (load_tracks: 00/06/12/18 UTC, NATURE TS, >= 34 kt) and all synoptic fixes for the track tables
    ib = os.path.join(GL, 'ibtracs.since1980.list.v04r01.csv')
    tr = load_tracks(ib)
    full = pd.read_csv(ib, skiprows=[1], low_memory=False, usecols=['SID', 'NAME', 'ISO_TIME', 'LAT', 'LON'])
    full['time'] = pd.to_datetime(full.ISO_TIME)
    full = full[full.time.dt.hour.isin([0, 6, 12, 18]) & (full.time.dt.minute == 0)].copy()
    full['LAT'] = pd.to_numeric(full.LAT, errors='coerce'); full['LON'] = pd.to_numeric(full.LON, errors='coerce')
    full = full.dropna(subset=['LAT', 'LON'])
    fix = {sid: g.set_index('time')[['LON', 'LAT']] for sid, g in full.groupby('SID')}
    name = full.groupby('SID').NAME.first().to_dict()
    tsv = {t: g for t, g in tr.groupby('time')}                      # TS fixes by exact time
    mon = days.month.values; yr = days.year.values
    cand = []
    why = {}
    for R in REG:
        rows = R['rows']; rlon, rlat = cells.lon.values[rows], cells.lat.values[rows]
        cx = float(np.degrees(np.angle(np.exp(1j * np.radians(rlon)).mean()))); cy = float(rlat.mean())
        R['lon'], R['lat'] = cx, cy
        mon_ok = seas_m[:, rows].mean(1) >= 0.5
        sd = mon_ok[mon - 1]
        frac = (code[:, rows] == R['s']).mean(1)
        Bm = np.nanmean(B[:, [col[r] for r in rows]], 1)
        J = np.where(sd & (frac >= 0.5) & (yr >= ys) & (yr <= ye) & np.isfinite(Bm))[0]
        w = {}
        for j in J:
            V = v_time(days[j], cx)
            g = tsv.get(V)
            if g is None:
                w['no_ts_fix_at_V'] = w.get('no_ts_fix_at_V', 0) + 1; continue
            dcen = hav(cx, cy, g.LON.values, g.LAT.values)
            k = int(np.argmin(dcen)); sid = g.SID.values[k]
            if (dcen <= OTHER_KM).sum() > 1:
                w['second_storm'] = w.get('second_storm', 0) + 1; continue
            fx = fix[sid]
            times = [V - pd.Timedelta(hours=6 * n) for n in range(PRE_H // 6 + 1)]
            if not all(t in fx.index for t in times):
                w['missing_fixes'] = w.get('missing_fixes', 0) + 1; continue
            dm = np.array([hav(rlon, rlat, fx.loc[t, 'LON'], fx.loc[t, 'LAT']).min() for t in times])
            if not (DMIN_OK <= dm[0] <= DMAX_V):
                w['distance_at_V'] = w.get('distance_at_V', 0) + 1; continue
            if dm.min() < DMIN_OK:
                w['too_close_earlier'] = w.get('too_close_earlier', 0) + 1; continue
            cand.append(dict(region=R['region'], sign=R['sign'], sector=R['sector'], day=days[j].date(), V=V.strftime('%Y-%m-%dT%H'),
                             sid=sid, name=name.get(sid, ''), wind=float(g.wind.values[k]), dist=float(dm[0]), dmin54=float(dm.min()),
                             brg=float(bearing(cx, cy, fx.loc[V, 'LON'], fx.loc[V, 'LAT'])), frac=float(frac[j]), anom=float(Bm[j])))
        why[R['region']] = dict(days=int(len(J)), **w)
    C = pd.DataFrame(cand)
    C.to_csv(os.path.join(a.out, 'gcg_candidates.csv'), index=False, float_format='%.4g')
    log(f'{len(C)} usable candidate days in {C.region.nunique() if len(C) else 0} regions, {time.time() - T0:.0f} s')

    # selection
    sel = []
    for R in REG:
        Q = C[C.region == R['region']]
        Q = Q.sort_values('anom', ascending=(R['sign'] == 'cool'))
        n_take = a.nwarm if R['sign'] == 'warm' else a.ncool
        per = {}; got = []
        for _, r in Q.iterrows():
            if per.get(r.sid, 0) >= 2:
                continue
            per[r.sid] = per.get(r.sid, 0) + 1; got.append(r)
            if len(got) >= n_take:
                break
        sel += got
        log(f"{R['region']} {R['sign']} {R['sector']} cells {len(R['rows'])} centre ({R['lon']:.1f}, {R['lat']:.1f}): "
            f"{why[R['region']]} -> {len(Q)} usable, {len(got)} chosen" +
            (f", anomalies {', '.join(f'{x.anom:+.1f}' for x in got)} K" if got else ''))
    S = pd.DataFrame(sel).reset_index(drop=True)
    S.insert(0, 'case', np.arange(len(S)))
    S['kind'] = S.sign
    S['tid'] = pd.factorize(S.sid)[0]
    S.to_csv(os.path.join(a.out, 'gc_cases.csv'), index=False, float_format='%.4g')
    trows, orows = [], []
    for _, c in S.iterrows():
        V = pd.Timestamp(c.V); fx = fix[c.sid]
        R = next(r for r in REG if r['region'] == c.region)
        for n in range(-1, 11):
            t = V - pd.Timedelta(hours=6 * n)
            if t in fx.index:
                trows.append(dict(case=int(c.case), time=t.strftime('%Y-%m-%dT%H'), lon=float(fx.loc[t, 'LON']), lat=float(fx.loc[t, 'LAT'])))
            g = tsv.get(t)
            if g is not None:
                o = g[g.SID != c.sid]
                d = hav(R['lon'], R['lat'], o.LON.values, o.LAT.values)
                for lo_, la_ in zip(o.LON.values[d <= 3000], o.LAT.values[d <= 3000]):
                    orows.append(dict(case=int(c.case), time=t.strftime('%Y-%m-%dT%H'), lon=float(lo_), lat=float(la_)))
    pd.DataFrame(trows).to_csv(os.path.join(a.out, 'gc_tracks.csv'), index=False)
    pd.DataFrame(orows, columns=['case', 'time', 'lon', 'lat']).to_csv(os.path.join(a.out, 'gc_others.csv'), index=False)
    json.dump({R['region']: dict(sign=R['sign'], sector=R['sector'], s=R['s'], lon=R['lon'], lat=R['lat'], ex_map=R['ex_map'],
                                 cells=[[float(x), float(y)] for x, y in zip(cells.lon.values[R['rows']], cells.lat.values[R['rows']])])
               for R in REG}, open(os.path.join(a.out, 'gcg_regions.json'), 'w'))
    log(f'cases {len(S)} ({(S.kind == "warm").sum()} warm, {(S.kind == "cool").sum()} cool) from {S.sid.nunique()} storms; '
        f'regions with >= 3 cases: warm {int((S[S.kind == "warm"].groupby("region").size() >= 3).sum())}, '
        f'cool {int((S[S.kind == "cool"].groupby("region").size() >= 3).sum())}; total {time.time() - T0:.0f} s')
    open(os.path.join(a.out, 'gcg_select_log.txt'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
