"""Global TC-heat map, step 2 (compute node): where do land days warm when a tropical cyclone sits 300-1300 km away?
Pre-registered design: paper/全球图_分析方案_预注册.md (2026-10-29).

Anomalies: harmonic seasonal cycle + linear trend removed per cell; then, inside the cell's TC season, each year's mean
over reference days (no TC within 1300 km) is removed (interannual background, e.g. ENSO) and so is the +-15-day
calendar climatology of reference days (timing inside the season).  The excess on exposure days is therefore the
departure from calendar- and year-matched TC-free days.

Usage: python gl_map.py [--years 1980-2021] [--nboot 1000] [--placebo 5] [--box lon0,lon1,lat0,lat1] [--min-exp 100]
Input ($TCH/global/): era5_t2_YYYY.npz (gl_get.py), static_1deg.npz, ibtracs.since1980.list.v04r01.csv
Output ($TCH/global/out/): gl_cells.csv, gl_regions.csv, gl_placebo.csv, gl_summary.txt, gl_codes.npz
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy import ndimage, stats

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
GL = os.path.join(TCH, 'global')
OUT = os.path.join(GL, 'out')
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
RNEAR, RFAR = 300.0, 1300.0


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def load_tracks(path):
    df = pd.read_csv(path, skiprows=[1], low_memory=False,
                     usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'NATURE', 'BASIN'])
    for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df.ISO_TIME)
    df['wind'] = df.USA_WIND.fillna(df.WMO_WIND)
    df = df[(df.time.dt.hour.isin([0, 6, 12, 18])) & (df.time.dt.minute == 0) & (df.NATURE.str.strip() == 'TS')
            & (df.wind >= 34)].dropna(subset=['LAT', 'LON'])
    # local solar date at the storm's longitude (cells within 1300 km differ by <= ~1 h of solar time)
    df['day'] = (df.time + pd.to_timedelta(df.LON / 15.0, unit='h')).dt.normalize()
    return df[['SID', 'time', 'day', 'LAT', 'LON', 'wind', 'BASIN']].reset_index(drop=True)


def exposure_codes(days, tr, clon, clat):
    """int8 [day, cell]: -1 reference (no TC within RFAR), -2 a TC within RNEAR, 0..7 exposure with the sector of the
    nearest TC (bearing from the cell) when the nearest TC that day is RNEAR-RFAR km away."""
    code = np.full((len(days), len(clon)), -1, np.int8)
    di = pd.Series(np.arange(len(days)), index=days)
    for day, g in tr.groupby('day'):
        k = di.get(day)
        if k is None:
            continue
        d = hav(clon[:, None], clat[:, None], g.LON.values[None, :], g.LAT.values[None, :])
        q = d.argmin(1); dm = d[np.arange(len(clon)), q]
        sec = ((bearing(clon, clat, g.LON.values[q], g.LAT.values[q]) + 22.5) % 360 // 45).astype(np.int8)
        c = np.full(len(clon), -1, np.int8)
        e = (dm >= RNEAR) & (dm <= RFAR)
        c[e] = sec[e]; c[dm < RNEAR] = -2
        code[k] = c
    return code


def shift_tracks(tr, k, y0, y1):
    """Placebo: the tracks of year y + k (cyclic in y0..y1) placed on the same month-day of year y."""
    out = []
    n = y1 - y0 + 1
    for y in range(y0, y1 + 1):
        src = y0 + (y - y0 + k) % n
        g = tr[tr.day.dt.year == src].copy()
        g = g[~((g.day.dt.month == 2) & (g.day.dt.day == 29))]
        g['day'] = pd.to_datetime({'year': y, 'month': g.day.dt.month, 'day': g.day.dt.day})
        out.append(g)
    return pd.concat(out)


def harmonic_anomalies(X, days):
    t = (days.year.values - 2000.5) / 10.0
    w = 2 * np.pi * days.dayofyear.values / 365.25
    D = np.stack([np.ones_like(t), t] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)], 1)
    B = np.linalg.lstsq(D, X, rcond=None)[0]
    return (X - D @ B).astype(np.float32)


def season_mask(code, days, min_frac=0.03):
    mon = days.month.values
    seas = np.zeros((12, code.shape[1]), bool)
    for m in range(1, 13):
        seas[m - 1] = (code[mon == m] >= 0).mean(0) >= min_frac
    return seas[mon - 1], seas


def matched(A, code, seas_d, yidx, ny, doy):
    """Remove, inside the season, each year's reference-day mean and the +-15-day calendar mean of reference days."""
    ref = seas_d & (code == -1)
    B = A.copy()
    for y in range(ny):
        m = yidx == y
        n = ref[m].sum(0)
        mu = np.where(ref[m], A[m], 0).sum(0, dtype=np.float64) / np.maximum(n, 1)
        B[m] -= np.where(n > 0, mu, 0).astype(np.float32)
    S = np.zeros((366, A.shape[1])); C = np.zeros((366, A.shape[1]))
    Bz = np.where(ref, B, np.float32(0))
    for k in range(366):
        m = doy == k + 1
        S[k] = Bz[m].sum(0, dtype=np.float64); C[k] = ref[m].sum(0)
    Sw = sum(np.roll(S, r, 0) for r in range(-15, 16)); Cw = sum(np.roll(C, r, 0) for r in range(-15, 16))
    B -= (Sw / np.maximum(Cw, 1))[doy - 1].astype(np.float32)
    B[~seas_d] = np.nan
    return B


def year_sums(B, sel, yidx, ny):
    S = np.zeros((ny, B.shape[1])); N = np.zeros((ny, B.shape[1]))
    Bz = np.where(sel, B, np.float32(0))
    for y in range(ny):
        m = yidx == y
        S[y] = Bz[m].sum(0, dtype=np.float64); N[y] = sel[m].sum(0)
    return S, N


def boot_mean(S, N, W):
    """Mean of matched anomalies on selected days; year-block bootstrap (W [nboot, ny]); normal-theory p from boot SD."""
    with np.errstate(invalid='ignore', divide='ignore'):
        est = S.sum(0) / N.sum(0)
        b = (W @ S) / (W @ N)
    sd = np.nanstd(b, 0)
    lo, hi = est - 1.96 * sd, est + 1.96 * sd
    with np.errstate(invalid='ignore', divide='ignore'):
        p = 2 * stats.norm.sf(np.abs(est) / sd)
    return est, lo, hi, p


def fdr(p, q=0.10):
    p = np.asarray(p, float); ok = np.isfinite(p); sig = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return sig
    ps = np.sort(p[ok]); m = len(ps)
    below = ps <= q * np.arange(1, m + 1) / m
    if below.any():
        sig[ok] = p[ok] <= ps[np.where(below)[0].max()]
    return sig


def analyse(B, code, seas_d, yidx, ny, W, min_exp):
    """Excess maps (all sectors and per sector) for one exposure-code array."""
    r = {}
    S, N = year_sums(B, seas_d & (code >= 0), yidx, ny)
    r['ex'], r['lo'], r['hi'], r['p'] = boot_mean(S, N, W)
    r['n'] = N.sum(0)
    r['valid'] = r['n'] >= min_exp
    nc = B.shape[1]
    r['exs'] = np.full((8, nc), np.nan); r['ps'] = np.full((8, nc), np.nan); r['ns'] = np.zeros((8, nc))
    for s in range(8):
        S, N = year_sums(B, seas_d & (code == s), yidx, ny)
        e, _, _, p = boot_mean(S, N, W)
        r['ns'][s] = N.sum(0)
        g = r['ns'][s] >= min_exp / 2
        r['exs'][s, g] = e[g]; r['ps'][s, g] = p[g]
    has = np.isfinite(r['exs']).any(0); cc = np.arange(nc)
    kb = np.where(has, np.argmax(np.where(np.isfinite(r['exs']), r['exs'], -np.inf), 0), 0)
    kw = np.where(has, np.argmin(np.where(np.isfinite(r['exs']), r['exs'], np.inf), 0), 0)
    r['kb'], r['kw'] = kb, kw
    r['ex_best'] = np.where(has, r['exs'][kb, cc], np.nan); r['p_best'] = np.where(has, np.minimum(1, 8 * r['ps'][kb, cc]), np.nan)
    r['ex_worst'] = np.where(has, r['exs'][kw, cc], np.nan); r['p_worst'] = np.where(has, np.minimum(1, 8 * r['ps'][kw, cc]), np.nan)
    for k in ('', '_best', '_worst'):
        pp = r['p' + k] if k else r['p']
        r['sig' + k] = fdr(np.where(r['valid'], pp, np.nan)) & r['valid']
    return r


def regions(df, col_ex, col_sig, sign, nlon, nlat):
    """8-connected clusters (lon wraps) of significant cells with sign*excess >= 0.5 K; >= 3 cells."""
    G = np.zeros((nlon, nlat), bool)
    sel = df[(sign * df[col_ex] >= 0.5) & df[col_sig]]
    G[sel.i.values, sel.j.values] = True
    big = np.concatenate([G, G], 0)
    lab, n = ndimage.label(big, structure=np.ones((3, 3)))
    rows, seen = [], set()
    idx = df.set_index(['i', 'j'])
    for r in range(1, n + 1):
        ii, jj = np.where(lab == r)
        if ii.min() >= nlon:
            continue
        key = tuple(sorted(set(zip(ii % nlon, jj))))
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        cells = idx.loc[list(key)]
        rows.append(dict(sign='warm' if sign > 0 else 'cool', n_cells=len(key),
                         lon=float(np.degrees(np.angle(np.exp(1j * np.radians(cells.lon)).mean()))),
                         lat=float(cells.lat.mean()), lon_min=float(cells.lon.min()), lon_max=float(cells.lon.max()),
                         lat_min=float(cells.lat.min()), lat_max=float(cells.lat.max()),
                         ex_mean=float(cells[col_ex].mean()), ex_any_mean=float(cells.ex_any.mean()),
                         sector=cells[col_ex.replace('ex_', 'sec_')].mode().iloc[0],
                         hot_ratio=float(cells.hot_ratio.mean()), n_exp_median=float(cells.n_exp.median()),
                         months=cells.months.mode().iloc[0]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='1980-2021')
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--placebo', type=int, default=5)
    ap.add_argument('--box', default='')
    ap.add_argument('--min-exp', type=int, default=100)
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split('-'))
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    st = np.load(os.path.join(GL, 'static_1deg.npz'))
    lon, lat = st['lon'], st['lat']
    LONg, LATg = np.meshgrid(lon, lat, indexing='ij')
    land = (st['lsm'] >= 0.5) & (np.abs(LATg) <= 60)
    if a.box:
        b = [float(x) for x in a.box.split(',')]
        land &= (LONg >= b[0]) & (LONg <= b[1]) & (LATg >= b[2]) & (LATg <= b[3])
    ci, cj = np.where(land)
    clon, clat = (((lon[ci] + 180) % 360) - 180).astype(float), lat[cj].astype(float)

    X, X14, dl = [], [], []
    for y in range(y0, y1 + 1):
        d = np.load(os.path.join(GL, f'era5_t2_{y}.npz'))
        X.append(d['tmax'][:, ci, cj]); X14.append(d['t14'][:, ci, cj]); dl.append(pd.DatetimeIndex(d['days']))
    X = np.concatenate(X); X14 = np.concatenate(X14); days = dl[0].append(dl[1:])
    ok = np.isfinite(X).all(1) & np.isfinite(X14).all(1)
    X, X14, days = X[ok], X14[ok], days[ok]
    print(f'{len(clon)} land cells, {len(days)} days ({(~ok).sum()} dropped for missing hours), {time.time() - T0:.0f} s', flush=True)
    A = harmonic_anomalies(X, days); A14 = harmonic_anomalies(X14, days)
    del X, X14

    tr = load_tracks(os.path.join(GL, 'ibtracs.since1980.list.v04r01.csv'))
    tr = tr[(tr.day >= days[0]) & (tr.day <= days[-1])]
    code = exposure_codes(days, tr, clon, clat)
    print(f'tracks: {tr.SID.nunique()} storms, {len(tr)} fixes; exposure codes {time.time() - T0:.0f} s', flush=True)
    seas_d, seas_m = season_mask(code, days)
    yidx = days.year.values - y0; ny = y1 - y0 + 1; doy = days.dayofyear.values
    rng = np.random.default_rng(29)
    W = np.stack([np.bincount(rng.integers(0, ny, ny), minlength=ny) for _ in range(a.nboot)]).astype(float)

    B = matched(A, code, seas_d, yidx, ny, doy)
    R = analyse(B, code, seas_d, yidx, ny, W, a.min_exp)
    B14 = matched(A14, code, seas_d, yidx, ny, doy); del A14
    S, N = year_sums(B14, seas_d & (code >= 0), yidx, ny)
    ex14, _, _, p14 = boot_mean(S, N, W); del B14
    valid = R['valid']; cc = np.arange(len(clon))
    res = pd.DataFrame(dict(i=ci, j=cj, lon=clon, lat=clat, n_exp=R['n'].astype(int), ex_any=R['ex'], ex_any_lo=R['lo'],
                            ex_any_hi=R['hi'], p_any=R['p'], sig_any=R['sig'], ex_any_t14=ex14, p_any_t14=p14))
    res['months'] = [''.join('x' if seas_m[m, c] else '.' for m in range(12)) for c in range(len(clon))]
    for s in range(8):
        res[f'ex_{SECT[s]}'] = R['exs'][s]; res[f'n_{SECT[s]}'] = R['ns'][s].astype(int); res[f'p_{SECT[s]}'] = R['ps'][s]
    res['sec_best'] = np.array(SECT)[R['kb']]; res['ex_best'] = R['ex_best']; res['p_best'] = R['p_best']; res['sig_best'] = R['sig_best']
    res['sec_worst'] = np.array(SECT)[R['kw']]; res['ex_worst'] = R['ex_worst']; res['p_worst'] = R['p_worst']; res['sig_worst'] = R['sig_worst']

    # hot days: top 5% of matched season anomalies; share that are exposure days vs calendar-matched expectation
    hr = np.full(len(clon), np.nan); hd = np.full(len(clon), np.nan)
    Hy = np.zeros((ny, len(clon))); Ey = np.zeros((ny, len(clon)))
    for c in np.where(valid)[0]:
        sd = seas_d[:, c]; e = code[:, c] >= 0
        thr = np.percentile(B[sd, c], 95); H = np.where(sd & (B[:, c] >= thr))[0]
        cnt_e = np.zeros((ny, 366)); cnt_d = np.zeros((ny, 366))
        np.add.at(cnt_e, (yidx[sd], doy[sd] - 1), e[sd]); np.add.at(cnt_d, (yidx[sd], doy[sd] - 1), 1)
        we = sum(np.roll(cnt_e, r, 1) for r in range(-7, 8)); wd = sum(np.roll(cnt_d, r, 1) for r in range(-7, 8))
        yk, dk = yidx[H], doy[H] - 1
        exp_ = (we[:, dk].sum(0) - we[yk, dk]) / np.maximum(wd[:, dk].sum(0) - wd[yk, dk], 1)
        np.add.at(Hy[:, c], yk, e[H]); np.add.at(Ey[:, c], yk, exp_)
        hr[c] = e[H].mean() / max(exp_.mean(), 1e-9); hd[c] = e[H].mean() - exp_.mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        bh = (W @ Hy) / (W @ Ey)
    res['hot_ratio'] = hr; res['hot_diff'] = hd
    res['hot_ratio_lo'] = np.nanpercentile(np.where(valid, bh, np.nan), 2.5, axis=0) if valid.any() else np.nan
    res['hot_ratio_hi'] = np.nanpercentile(np.where(valid, bh, np.nan), 97.5, axis=0) if valid.any() else np.nan
    res['valid'] = valid
    res.to_csv(os.path.join(OUT, 'gl_cells.csv'), index=False)
    np.savez_compressed(os.path.join(OUT, 'gl_codes.npz'), code=code, days=days.values.astype('datetime64[D]'),
                        ci=ci, cj=cj, valid=valid, seas_m=seas_m)
    print(f'real map done {time.time() - T0:.0f} s', flush=True)

    def frac(R_, v):
        return dict(n_valid=int(v.sum()), frac_sig_any=float(R_['sig'][v].mean()) if v.any() else np.nan,
                    frac_sig_any_warm=float((R_['sig'] & (R_['ex'] > 0))[v].mean()) if v.any() else np.nan,
                    frac_sig_best=float(R_['sig_best'][v].mean()) if v.any() else np.nan,
                    frac_best_ge05_sig=float((R_['sig_best'] & (R_['ex_best'] >= 0.5))[v].mean()) if v.any() else np.nan,
                    median_ex_any=float(np.nanmedian(R_['ex'][v])) if v.any() else np.nan)
    rows = [dict(run='real', **frac(R, valid))]
    for k in range(1, a.placebo + 1):
        cp = exposure_codes(days, shift_tracks(tr, k, y0, y1), clon, clat)
        Bp = matched(A, cp, seas_d, yidx, ny, doy)
        Rp = analyse(Bp, cp, seas_d, yidx, ny, W, a.min_exp)
        rows.append(dict(run=f'shift{k}', **frac(Rp, valid & Rp['valid'])))
        del Bp
        print(f'placebo shift {k}: {time.time() - T0:.0f} s', flush=True)
    PL = pd.DataFrame(rows); PL.to_csv(os.path.join(OUT, 'gl_placebo.csv'), index=False)

    # compact 1-deg grids for the figures (int8, NaN = -128): excess in 0.05 K steps, sector 0-7, flags
    def q8(v, step):
        out = np.full((len(lon), len(lat)), -128, np.int8)
        vv = np.clip(np.round(np.asarray(v, float) / step), -127, 127)
        g = np.isfinite(vv)
        out[ci[g], cj[g]] = vv[g].astype(np.int8)
        return out
    np.savez_compressed(os.path.join(OUT, 'gl_fig.npz'), lon=lon, lat=lat,
                        ex_any=q8(np.where(valid, R['ex'], np.nan), 0.05), sig_any=q8(np.where(valid, R['sig'], np.nan), 1),
                        ex_best=q8(np.where(valid, R['ex_best'], np.nan), 0.05), sec_best=q8(np.where(valid, R['kb'], np.nan), 1),
                        sig_best=q8(np.where(valid, R['sig_best'], np.nan), 1),
                        ex_worst=q8(np.where(valid, R['ex_worst'], np.nan), 0.05), sec_worst=q8(np.where(valid, R['kw'], np.nan), 1),
                        sig_worst=q8(np.where(valid, R['sig_worst'], np.nan), 1),
                        hot_ratio=q8(np.where(valid, hr, np.nan), 0.05), n_exp=q8(np.where(valid, np.minimum(R['n'] / 10, 127), np.nan), 1),
                        ex_t14=q8(np.where(valid, ex14, np.nan), 0.05))

    RG = pd.concat([regions(res[valid], 'ex_best', 'sig_best', +1, len(lon), len(lat)),
                    regions(res[valid], 'ex_worst', 'sig_worst', -1, len(lon), len(lat))], ignore_index=True)
    RG.to_csv(os.path.join(OUT, 'gl_regions.csv'), index=False)

    # Guangdong check (pre-registered a): TC to the E or SE
    gd = valid & res.lon.between(110, 117).values & res.lat.between(21, 25).values
    S, N = year_sums(B, seas_d & ((code == 2) | (code == 3)), yidx, ny)
    e_g, _, _, p_g = boot_mean(S, N, W)
    pl = PL.iloc[1:]
    lines = [f'gl_map.py {a.years}, {len(clon)} land cells, {int(valid.sum())} with >= {a.min_exp} exposure days in season',
             f'ex_any (all sectors): median {np.nanmedian(R["ex"][valid]):+.2f} K; FDR-significant {int(R["sig"].sum())} cells '
             f'({int((R["sig"] & (R["ex"] > 0)).sum())} warm, {int((R["sig"] & (R["ex"] < 0)).sum())} cool); '
             f'14 LT version median {np.nanmedian(ex14[valid]):+.2f} K',
             f'best sector: FDR-significant {int(R["sig_best"].sum())} cells; >= +0.5 K and significant: {int((R["sig_best"] & (R["ex_best"] >= 0.5)).sum())}',
             f'worst sector: FDR-significant {int(R["sig_worst"].sum())} cells; <= -0.5 K and significant: {int((R["sig_worst"] & (R["ex_worst"] <= -0.5)).sum())}',
             '', 'PLACEBO (tracks shifted by k years):', PL.round(4).to_string(index=False),
             f"(b) placebo/real: frac_sig_any {pl.frac_sig_any.mean() / max(PL.frac_sig_any[0], 1e-9):.2f}, "
             f"frac_sig_best {pl.frac_sig_best.mean() / max(PL.frac_sig_best[0], 1e-9):.2f}, "
             f"hotspot cells {pl.frac_best_ge05_sig.mean() / max(PL.frac_best_ge05_sig[0], 1e-9):.2f} (pre-registered: <= 0.20)",
             '', f'(a) Guangdong cells (110-117E, 21-25N): {int(gd.sum())}; TC to E or SE: excess mean {np.nanmean(e_g[gd]):+.2f} K, '
             f'positive {int((e_g[gd] > 0).sum())}, positive with p<0.05 {int(((p_g < 0.05) & (e_g > 0))[gd].sum())}',
             res[gd][['lon', 'lat', 'n_exp', 'ex_any', 'ex_E', 'ex_SE', 'ex_SW', 'sec_best', 'ex_best', 'hot_ratio']].round(2).to_string(index=False),
             '', 'REGIONS (all, warm and cool):', RG.round(2).to_string(index=False), '', f'total {time.time() - T0:.0f} s']
    open(os.path.join(OUT, 'gl_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
