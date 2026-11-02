"""Global TC-heat map, stage 2 (compute node; Supplementary Section S1): for every warm (and
cool) region of gl_map.py, the excess of the regional daily maximum temperature over circulation analogues, the 850-hPa
temperature and humidity fingerprints, and the temperature as the storm approaches; the same for Guangdong as a check.
Also an exploratory test: is a storm in a given direction warmer for the land when its winds reach
the land from inland than when they reach it from the sea?

Exposure: in the region's TC season (months in which at least half of its cells are in season), the cell-days of the
region on which the cell's nearest storm is 300-1300 km away in the region's sector and no storm is within 300 km of the
cell (the codes of gl_map.py).  For each region day with at least one such cell, the excess of every exposed cell over
the same cell on the analogue days is computed, and the region value is the mean over all exposed cell-days.
Analogues (as for the Guangdong stations): the K days with the most similar 500-hPa height anomaly (area-weighted mean
squared difference) among TC-free days (no storm within 1300 km of any cell of the region) of other seasons within
+-15 days of the calendar day; the comparison excludes grid points within 1000 km of any storm on the region day and
within 600 km of the region, inside a circle of 3000 km plus the region's radius.  All series have a harmonic seasonal
cycle and trend removed and then each season's mean.  Confidence intervals resample whole storms (the storm nearest to
the exposed cells on each day).

Usage: python gl_stage2.py [--years 1980-2021] [--nboot 1000] [--K 5]
Input ($TCH/global/): out/gl_cells.csv, out/gl_codes.npz, out/gl_regions.csv (gl_map.py), era5_t2_YYYY.npz (gl_get.py),
                      era5_c_YYYY.npz (gl_get2.py), static_1deg.npz, ibtracs.since1980.list.v04r01.csv
Output ($TCH/global/out/): gl_s2_regions.csv, gl_s2_onset.csv, gl_s2_fetch.csv, gl_s2_fetch_hist.csv, gl_s2_summary.txt
"""
import argparse, os, time
from math import erf, sqrt
import numpy as np, pandas as pd
from scipy import ndimage
from gl_map import hav, load_tracks, harmonic_anomalies, fdr, SECT, RNEAR, RFAR

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
GL = os.path.join(TCH, 'global')
OUT = os.path.join(GL, 'out')
VARS = ['tmax', 't850', 'q850', 'tmax_t850', 'tcc', 'tp24', 'u10', 'v10', 'z500']


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


def dest(lon, lat, brg, dkm):
    """Point at distance dkm along initial bearing brg (deg) from (lon, lat)."""
    p1, l1, b, d = np.radians(lat), np.radians(lon), np.radians(brg), dkm / 6371.0
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(b))
    l2 = l1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return (np.degrees(l2) + 180) % 360 - 180, np.degrees(p2)


def boot(S, N, grp, rng, nb):
    """Ratio of sums over region days, resampling whole storms; normal-theory p from the bootstrap SD."""
    ok = N > 0; S, N, grp = S[ok], N[ok], grp[ok]
    if N.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    ug, inv = np.unique(grp, return_inverse=True)
    GS = np.bincount(inv, S, len(ug)); GN = np.bincount(inv, N, len(ug))
    W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(nb)]).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        bs = (W @ GS) / (W @ GN)
    m = S.sum() / N.sum(); se = np.nanstd(bs)
    p = 2 * (1 - 0.5 * (1 + erf(abs(m) / se / sqrt(2)))) if se > 0 else np.nan
    return m, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='1980-2021')
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--gd-box', default='110,117,21,25')
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split('-'))
    rng = np.random.default_rng(31)
    T0 = time.time()

    # ---- cells, codes and regions from gl_map.py
    cells = pd.read_csv(os.path.join(OUT, 'gl_cells.csv'))
    for c in ('valid', 'sig_best', 'sig_worst'):
        cells[c] = cells[c].astype(str).str.lower().eq('true')
    G = np.load(os.path.join(OUT, 'gl_codes.npz'))
    code, seas_m = G['code'], G['seas_m']
    days = pd.DatetimeIndex(G['days'])
    assert len(cells) == code.shape[1] and (cells.i.values == G['ci']).all() and (cells.j.values == G['cj']).all()
    st = np.load(os.path.join(GL, 'static_1deg.npz'))
    lon1, lat1, lsm = st['lon'], st['lat'], st['lsm']
    nlon, nlat = len(lon1), len(lat1)
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
            REG.append(dict(region=f"{'W' if sign > 0 else 'C'}{k + 1}", sign=rg.sign[k], rows=rows, sectors=[SECT.index(rg.sector[k])],
                            sector=rg.sector[k], ex_stage1=rg.ex_mean[k]))
    b = [float(x) for x in a.gd_box.split(',')]
    gd = np.where(cells.valid & cells.lon.between(b[0], b[1]) & cells.lat.between(b[2], b[3]))[0]
    REG.append(dict(region='GD_E_SE', sign='check', rows=gd, sectors=[2, 3], sector='E+SE', ex_stage1=np.nan))
    REG.append(dict(region='GD_NE_E_SE', sign='check', rows=gd, sectors=[1, 2, 3], sector='NE+E+SE', ex_stage1=np.nan))
    REG.append(dict(region='GD_SW', sign='check', rows=gd, sectors=[5], sector='SW', ex_stage1=np.nan))
    print(f'{len(REG)} regions ({len(RG)} from gl_regions.csv + 3 Guangdong checks), {time.time() - T0:.0f} s', flush=True)

    # ---- daily maximum temperature at the region cells (1 deg) and coarse fields (2.8 deg)
    need = np.unique(np.concatenate([r['rows'] for r in REG]))
    col = {c: n for n, c in enumerate(need)}
    X = np.full((len(days), len(need)), np.nan, np.float32)
    C = {k: None for k in ('z500', 't850', 'q850', 'tcc', 'tp24', 'u10', 'v10')}
    Cl = {k: [] for k in C}; cd = []
    for y in range(y0, y1 + 1):
        d = np.load(os.path.join(GL, f'era5_t2_{y}.npz'))
        pos = pd.DatetimeIndex(d['days']).get_indexer(days[days.year == y])
        assert (pos >= 0).all(), y
        X[days.year == y] = d['tmax'][pos][:, cells.i.values[need], cells.j.values[need]]
        e = np.load(os.path.join(GL, f'era5_c_{y}.npz'))
        for k in C:
            Cl[k].append(e[k].reshape(len(e['days']), -1))
        cd.append(pd.DatetimeIndex(e['days']))
        clon, clat = e['lon'], e['lat']
    cdays = cd[0].append(cd[1:]); ci_ = cdays.get_indexer(days)
    assert (ci_ >= 0).all()
    for k in C:
        C[k] = np.concatenate(Cl[k])[ci_]
    del Cl
    CLON, CLAT = [g.ravel() for g in np.meshgrid(clon, clat, indexing='ij')]
    CLON = (CLON + 180) % 360 - 180
    A = harmonic_anomalies(X, days); del X
    CA = {k: harmonic_anomalies(C[k], days) for k in C}; del C
    print(f'fields loaded and anomalies {time.time() - T0:.0f} s', flush=True)

    tr = load_tracks(os.path.join(GL, 'ibtracs.since1980.list.v04r01.csv'))
    tr = tr[(tr.day >= days[0]) & (tr.day <= days[-1])]
    di = pd.Series(np.arange(len(days)), index=days)
    TR = {}
    for day, g in tr.groupby('day'):
        if day in di.index:
            TR[di[day]] = (g.LON.values.astype(float), g.LAT.values.astype(float), g.SID.values)
    empty = (np.zeros(0), np.zeros(0), np.array([], object))

    mon = days.month.values; yr = days.year.values; doy = days.dayofyear.values
    rows_out, ons_out = [], []
    for R in REG:
        rows = R['rows']
        if len(rows) == 0:
            continue
        rlon, rlat = cells.lon.values[rows], cells.lat.values[rows]
        cen = (float(np.degrees(np.angle(np.exp(1j * np.radians(rlon)).mean()))), float(rlat.mean()))
        rad = float(hav(cen[0], cen[1], rlon, rlat).max())
        mon_ok = seas_m[:, rows].mean(1) >= 0.5
        wrap = mon_ok[11] and mon_ok[0]
        sy = yr + (wrap & (mon >= 7))
        sd = mon_ok[mon - 1]
        cc = code[:, rows]
        ex_c = sd[:, None] & np.isin(cc, R['sectors'])            # cell-days with the storm in the region's sector
        pool = sd & (cc == -1).all(1)
        # series per cell: 1-deg Tmax and the nearest 2.8-deg point for the other variables
        pts = np.array([np.argmin(hav(CLON, CLAT, x, y_)) for x, y_ in zip(rlon, rlat)])
        Y = {'tmax': A[:, [col[r] for r in rows]].astype(float)}
        for k in ('t850', 'q850', 'tcc', 'tp24', 'u10', 'v10', 'z500'):
            Y[k] = CA[k][:, pts].astype(float)
        Y['tmax_t850'] = Y['tmax'] - Y['t850']
        for k in Y:
            y = Y[k]
            for s in np.unique(sy[sd]):
                m = sy == s
                y[m] -= y[m & sd].mean(0)
            y[~sd] = np.nan
        # matching field
        dc = hav(cen[0], cen[1], CLON, CLAT)
        dom = np.where((dc <= 3000.0 + rad) & (np.abs(CLAT) <= 80))[0]
        dreg = hav(rlon[:, None], rlat[:, None], CLON[dom][None, :], CLAT[dom][None, :]).min(0)
        base = dreg > 600.0
        Z = CA['z500'][:, dom].astype(float)
        for s in np.unique(sy[sd]):
            m = sy == s
            Z[m] -= Z[m & sd].mean(0)
        wgt = np.cos(np.radians(CLAT[dom]))
        J = np.where(ex_c.any(1))[0]
        an = np.full((len(J), a.K), -1); grp = np.empty(len(J), object); calm = []
        for n, j in enumerate(J):
            fl, fa, fs = TR.get(j, empty)
            mask = base.copy()
            if len(fl):
                mask &= hav(CLON[dom][:, None], CLAT[dom][:, None], fl[None, :], fa[None, :]).min(1) > 1000.0
            dd = np.abs(doy - doy[j]); dd = np.minimum(dd, 365 - dd)
            cand = np.where(pool & (sy != sy[j]) & (dd <= 15))[0]
            calm.append(cand)
            ce = ex_c[j]
            grp[n] = fs[np.argmin(hav(rlon[ce].mean(), rlat[ce].mean(), fl, fa))] if len(fl) else f'd{j}'
            if len(cand) < a.K or mask.sum() < 10:
                continue
            w = wgt[mask]
            r = (((Z[cand][:, mask] - Z[j, mask]) ** 2) * w).sum(1) / w.sum()
            an[n] = cand[np.argsort(r)[:a.K]]
        okj = (an >= 0).all(1)
        J, an, grp, calm = J[okj], an[okj], grp[okj], [c for c, o in zip(calm, okj) if o]
        E = ex_c[J]                                                # [day, cell] exposed cells
        half = E.mean(1) >= 0.5
        out = dict(region=R['region'], sign=R['sign'], sector=R['sector'], n_cells=len(rows), lon=cen[0], lat=cen[1], radius_km=rad,
                   months=''.join('x' if m else '.' for m in mon_ok), n_days=int(len(J)), n_cell_days=int(E.sum()),
                   n_storms=int(len(np.unique(grp))), n_days_half=int(half.sum()), ex_stage1=R['ex_stage1'])

        def pairs(y, ref, sel=None):
            """Sum and count over exposed cells of y minus the mean of y on the reference days, per region day."""
            S_ = np.zeros(len(J)); N_ = np.zeros(len(J))
            for n, j in enumerate(J):
                if sel is not None and not sel[n]:
                    continue
                d_ = y[j, E[n]] - np.nanmean(y[ref[n]][:, E[n]], 0)
                ok = np.isfinite(d_); S_[n] = d_[ok].sum(); N_[n] = ok.sum()
            return S_, N_

        for k in VARS:
            if len(J) < 10:
                break
            out.update(dict(zip([f'{k}_ex', f'{k}_lo', f'{k}_hi', f'{k}_p'], boot(*pairs(Y[k], an), grp, rng, a.nboot))))
        if len(J) >= 10:
            out.update(dict(zip(['tmax_cal', 'tmax_cal_lo', 'tmax_cal_hi', 'tmax_cal_p'], boot(*pairs(Y['tmax'], calm), grp, rng, a.nboot))))
        if half.sum() >= 10:
            out.update(dict(zip(['tmax_half', 'tmax_half_lo', 'tmax_half_hi', 'tmax_half_p'],
                                boot(*pairs(Y['tmax'], an, half), grp, rng, a.nboot))))
        rows_out.append(out)
        ymean = np.nanmean(Y['tmax'], 1)
        # approach of the storm: day 0 = first day within 1300 km of any cell of the region
        sids = set(grp)
        first, near_any = {}, np.zeros(len(days), bool)
        for j, (fl, fa, fs) in TR.items():
            dm = hav(rlon[:, None], rlat[:, None], fl[None, :], fa[None, :]).min(0)
            near_any[j] = (dm <= RFAR).any()
            for s_, dv in zip(fs, dm):
                if dv <= RFAR and (s_ not in first or j < first[s_][0]):
                    first[s_] = (j, dv)
        ent = [j for s_, (j, dv) in first.items() if s_ in sids and dv >= RNEAR and 5 <= j < len(days) - 3
               and not near_any[j - 5:j].any() and sd[j - 5:j + 4].all() and (sy[j - 5:j + 4] == sy[j]).all()]
        ent = np.array(sorted(ent), int)
        if len(ent) >= 5:
            V = np.stack([ymean[ent + lag] for lag in range(-5, 4)], 1)
            bs = np.stack([np.nanmean(V[rng.integers(0, len(ent), len(ent))], 0) for _ in range(a.nboot)])
            rise = np.nanmean(V[:, 5:7], 1) - np.nanmean(V[:, 0:3], 1)
            rb = np.array([np.nanmean(rise[rng.integers(0, len(ent), len(ent))]) for _ in range(a.nboot)])
            for n, lag in enumerate(range(-5, 4)):
                ons_out.append(dict(region=R['region'], lag=lag, n=len(ent), mean=np.nanmean(V[:, n]),
                                    lo=np.percentile(bs[:, n], 2.5), hi=np.percentile(bs[:, n], 97.5)))
            rows_out[-1].update(onset_n=len(ent), onset_rise=np.nanmean(rise), onset_rise_lo=np.percentile(rb, 2.5),
                                onset_rise_hi=np.percentile(rb, 97.5))
        print(f"{R['region']} {R['sector']}: {len(J)} days, Tmax over analogues {rows_out[-1].get('tmax_ex', np.nan):+.2f} K, "
              f"{time.time() - T0:.0f} s", flush=True)
    S2 = pd.DataFrame(rows_out)
    for s in ('warm', 'cool'):
        m = (S2.sign == s).values
        S2.loc[m, 'tmax_sig_fdr'] = fdr(S2.loc[m, 'tmax_p'].values)
    S2.to_csv(os.path.join(OUT, 'gl_s2_regions.csv'), index=False, float_format='%.4g')
    pd.DataFrame(ons_out).to_csv(os.path.join(OUT, 'gl_s2_onset.csv'), index=False, float_format='%.4g')

    # ---- exploratory: where do the storm's winds come from?  Cyclonic (tangential) wind at the cell for a storm in
    # sector s; upstream land fraction along 100-600 km against the wind (1-deg land-sea mask)
    fr = []
    vv = cells[cells.valid]
    for s in range(8):
        e_ = vv[f'ex_{SECT[s]}'].values; p_ = vv[f'p_{SECT[s]}'].values
        ok = np.isfinite(e_)
        th = np.radians(45.0 * s)
        nh = vv.lat.values >= 0
        ve = np.where(nh, np.cos(th), -np.cos(th)); vn = np.where(nh, -np.sin(th), np.sin(th))
        up = (np.degrees(np.arctan2(-ve, -vn)) + 360) % 360
        land = np.zeros(len(vv))
        for dk in (100, 200, 300, 400, 500, 600):
            x, y_ = dest(vv.lon.values, vv.lat.values, up, dk)
            ii = np.round((x % 360 - lon1[0]) / (lon1[1] - lon1[0])).astype(int) % nlon
            jj = np.clip(np.round((y_ - lat1[0]) / (lat1[1] - lat1[0])).astype(int), 0, nlat - 1)
            land += lsm[ii, jj] >= 0.5
        fr.append(pd.DataFrame(dict(lon=vv.lon.values[ok], lat=vv.lat.values[ok], sector=SECT[s], ex=e_[ok], p=p_[ok],
                                    land6=land[ok].astype(int))))
    FR = pd.concat(fr, ignore_index=True)
    FR['cls'] = np.select([FR.land6 >= 5, FR.land6 <= 1], ['from land', 'from sea'], 'mixed')
    FR['box'] = (np.floor(FR.lon / 10) * 100 + np.floor(FR.lat / 10)).astype(int)
    ub = FR.box.unique()
    rows = []
    for k, g in FR.groupby('cls'):
        rows.append(dict(cls=k, n=len(g), n_cells=g.groupby(['lon', 'lat']).ngroups, mean_ex=g.ex.mean(), median_ex=g.ex.median(),
                         frac_warm_sig=((g.p * 8 < 0.05) & (g.ex > 0)).mean(), frac_cool_sig=((g.p * 8 < 0.05) & (g.ex < 0)).mean()))
    FS = pd.DataFrame(rows)
    # land minus sea, resampling 10x10-deg boxes
    bi = {u: np.where(FR.box.values == u)[0] for u in ub}
    dl = []
    for _ in range(a.nboot):
        ix = np.concatenate([bi[u] for u in rng.choice(ub, len(ub))])
        g = FR.iloc[ix]
        dl.append(g.ex[g.cls == 'from land'].mean() - g.ex[g.cls == 'from sea'].mean())
    d0 = FR.ex[FR.cls == 'from land'].mean() - FR.ex[FR.cls == 'from sea'].mean()
    FS = pd.concat([FS, pd.DataFrame([dict(cls='land minus sea', mean_ex=d0, lo=np.nanpercentile(dl, 2.5), hi=np.nanpercentile(dl, 97.5))])])
    FS.to_csv(os.path.join(OUT, 'gl_s2_fetch.csv'), index=False, float_format='%.4g')
    bins = np.arange(-3.0, 3.01, 0.25)
    H = pd.DataFrame({k: np.histogram(np.clip(FR.ex[FR.cls == k], -2.99, 2.99), bins)[0] for k in ('from land', 'mixed', 'from sea')})
    H.insert(0, 'bin_lo', bins[:-1]); H.to_csv(os.path.join(OUT, 'gl_s2_fetch_hist.csv'), index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    keep = ['region', 'sector', 'n_cells', 'lon', 'lat', 'months', 'n_days', 'n_cell_days', 'n_storms', 'ex_stage1', 'tmax_cal', 'tmax_ex', 'tmax_lo',
            'tmax_hi', 'tmax_p', 'tmax_sig_fdr', 'tmax_half', 't850_ex', 't850_p', 'q850_ex', 'q850_p', 'tmax_t850_ex', 'tcc_ex', 'tp24_ex',
            'onset_n', 'onset_rise', 'onset_rise_lo', 'onset_rise_hi']
    w = S2[S2.sign == 'warm']
    lines = [f'gl_stage2.py {a.years}, K={a.K}, {len(S2)} regions',
             f"warm regions: {len(w)}; Tmax over analogues > 0 with FDR q=0.10: {int((w.tmax_sig_fdr.astype(bool) & (w.tmax_ex > 0)).sum())}; "
             f"T850 up (p<0.05): {int(((w.t850_p < 0.05) & (w.t850_ex > 0)).sum())}; q850 down (p<0.05): {int(((w.q850_p < 0.05) & (w.q850_ex < 0)).sum())}; "
             f"Tmax rises on approach (CI above 0): {int((w.get('onset_rise_lo', pd.Series(dtype=float)) > 0).sum())} of {int(w.get('onset_n', pd.Series(dtype=float)).notna().sum())}",
             '', S2[[c for c in keep if c in S2]].round(3).to_string(index=False), '', 'FETCH (exploratory):', FS.round(3).to_string(index=False),
             '', f'total {time.time() - T0:.0f} s']
    open(os.path.join(OUT, 'gl_s2_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
