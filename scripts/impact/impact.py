"""A3 (frozen design: plan A3 (internal)): hot person-days per year caused by distant tropical cyclones.

For every grid point of the 23 warm regions of the global map: daily maximum temperature anomaly A (harmonic seasonal cycle
and linear trend removed, as in gl_map.py), hot day = A >= P90 of the point's storm-season days 1980-2021; exposure day =
storm-season day whose exposure code equals the region's direction; storm contribution on exposure days
delta = f * max(ex_dir, 0) with ex_dir the global-map excess of the point in the region's direction (0 if missing);
storm-caused hot day = exposure day with A >= P90 and A - delta < P90.  Person-days use GHS-POP (pop_agg.py), 2020
population for the main result.  The 18 cool regions give the hot days prevented by storms (f_cool = 0.16, and 1).

Inputs (read only): $GL/era5_t2_YYYY.npz, $GL/static_1deg.npz, $GL/out/gl_codes.npz, $GL/out/gl_cells.csv;
$IMP/gcg_regions.json, $IMP/all_oper_regions.csv (A2), $IMP/pop/pop_1deg.npz.
Output: $IMP/out/impact_result.txt (with the verdict), impact_regions.csv, impact_years.csv, impact_cells.csv.
Usage: python impact.py [--nboot 1000]
"""
import argparse, json, os, time
import numpy as np, pandas as pd

GL = os.environ.get('GL', '/scratch/$USER/tcheat/global')
IMP = os.environ.get('IMP', '/scratch/$USER/tcheat/impact')
OUT = os.path.join(IMP, 'out')
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
F_MAIN, F_LOW, F_CI = 0.50, 0.34, (0.39, 0.61)              # A2: 0.52 / 1.03 K (0.40-0.63); days not in A: 34%
F_COOL = 0.16                                                # A2 cool regions: 0.15 / 0.94 K
CONT = {'Africa': [1, 2, 3, 4], 'South Asia': [5, 6], 'East and Southeast Asia': [7, 9, 10, 12, 13],
        'Australia and New Guinea': [8, 11, 14, 15, 16], 'Americas': [17, 18, 19, 20, 21, 22, 23]}


def harmonic_anomalies(X, days):                             # identical to gl_map.py
    t = (days.year.values - 2000.5) / 10.0
    w = 2 * np.pi * days.dayofyear.values / 365.25
    D = np.stack([np.ones_like(t), t] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)], 1)
    B = np.linalg.lstsq(D, X, rcond=None)[0]
    return (X - D @ B).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nboot', type=int, default=1000); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); T0 = time.time(); log = []

    def say(s=''):
        print(s, flush=True); log.append(s)

    Z = np.load(os.path.join(GL, 'out', 'gl_codes.npz'))
    code, ci, cj, seas_m = Z['code'], Z['ci'], Z['cj'], Z['seas_m']
    days = pd.DatetimeIndex(Z['days'])
    st = np.load(os.path.join(GL, 'static_1deg.npz')); lon, lat = st['lon'].astype(float), st['lat'].astype(float)
    clon, clat = ((lon[ci] + 180) % 360) - 180, lat[cj]
    CEL = pd.read_csv(os.path.join(GL, 'out', 'gl_cells.csv'))
    assert (CEL.i.values == ci).all() and (CEL.j.values == cj).all(), 'gl_cells.csv and gl_codes.npz cell order differ'
    key = {(int(round(clon[k])), int(round(clat[k]))): k for k in range(len(ci))}

    REG = json.load(open(os.path.join(IMP, 'gcg_regions.json')))
    rows = []
    for r, d in REG.items():
        s = int(d['s']); cells = d['cells'] if isinstance(d['cells'], list) else json.loads(d['cells'])
        for lo_, la_ in cells:
            k = key[(int(round(((float(lo_) + 180) % 360) - 180)), int(round(float(la_))))]
            ex = CEL[f'ex_{SECT[s]}'].values[k]
            rows.append(dict(region=r, kind=d['sign'], s=s, cell=k, lon=clon[k], lat=clat[k], ex_dir=ex))
    RC = pd.DataFrame(rows)
    U = np.unique(RC.cell.values); pos = {c: n for n, c in enumerate(U)}
    say(f'A3 impact.py: {len(REG)} regions, {len(RC)} region cells ({RC[RC.kind == "warm"].cell.nunique()} warm, '
        f'{RC[RC.kind == "cool"].cell.nunique()} cool), {len(U)} distinct grid points; {len(days)} days '
        f'{days[0].date()}..{days[-1].date()}')

    # daily maximum temperature at the region cells, on exactly the days of gl_codes.npz
    want = set(days.values.astype('datetime64[D]').tolist()); X = []; got = []
    for y in range(days[0].year, days[-1].year + 1):
        d = np.load(os.path.join(GL, f'era5_t2_{y}.npz'))
        dd = d['days'].astype('datetime64[D]'); m = np.array([v in want for v in dd.tolist()])
        X.append(d['tmax'][m][:, ci[U], cj[U]]); got.append(dd[m])
    X = np.concatenate(X); got = np.concatenate(got)
    assert len(got) == len(days) and (got == days.values.astype('datetime64[D]')).all(), 'day mismatch'
    assert np.isfinite(X).all()
    A = harmonic_anomalies(X, days); TC = X - 273.15
    say(f'loaded Tmax {X.shape} in {time.time() - T0:.0f} s; Tmax range {TC.min():.1f}..{TC.max():.1f} C')
    mon = days.month.values; y0 = days[0].year; yidx = days.year.values - y0; ny = yidx.max() + 1; years = np.arange(y0, y0 + ny)
    SE = seas_m[mon - 1][:, U]                                  # season mask [day, U]
    P90 = np.array([np.percentile(A[SE[:, n], n], 90) for n in range(len(U))])
    P95 = np.array([np.percentile(A[SE[:, n], n], 95) for n in range(len(U))])

    # population: epochs -> years (linear; 2021 = 2020), 2020 for the main result
    PZ = np.load(os.path.join(IMP, 'pop', 'pop_1deg.npz'))
    assert np.allclose(PZ['lon'], lon) and np.allclose(PZ['lat'], lat)
    pe = PZ['pop'][:, ci[U], cj[U]]; ep = PZ['epochs']
    popy = np.stack([np.array([np.interp(y, ep, pe[:, n]) for n in range(len(U))]) for y in years])   # [year, U]
    p20 = pe[list(ep).index(2020)]

    # regional f from A2 (regions with >= 3 days; d_free / ex_map clipped to 0-1; others F_MAIN)
    A2 = pd.read_csv(os.path.join(IMP, 'all_oper_regions.csv')).set_index('region')
    A2['usable'] = A2.usable.astype(str).str.lower().eq('true')
    freg = {r: (float(np.clip(A2.loc[r, 'd_free'] / A2.loc[r, 'ex_map'], 0, 1)) if r in A2.index and A2.loc[r, 'usable'] else F_MAIN)
            for r in REG}

    FV = {'f0.50': F_MAIN, 'f0.34': F_LOW, 'f0.39': F_CI[0], 'f0.61': F_CI[1], 'fregion': None}
    yc = lambda m: np.bincount(yidx[m], minlength=ny)
    cnt = []                                                    # per region cell: per-year counts [ny]
    for r_, g in RC.groupby('region', sort=False):
        kind = g.kind.iloc[0]; s = g.s.iloc[0]
        for _, row in g.iterrows():
            n = pos[row.cell]; c = row.cell; sd = SE[:, n]
            expo = sd & (code[:, c] == s); a_ = A[:, n]; t_ = TC[:, n]; ex = row.ex_dir
            rec = dict(region=r_, kind=kind, cell=c, u=n, lon=row.lon, lat=row.lat, ex_dir=ex, pop2020=p20[n],
                       n_exp=yc(expo), n_hot=yc(sd & (a_ >= P90[n])), n_hot_exp=yc(expo & (a_ >= P90[n])),
                       n_hot95=yc(sd & (a_ >= P95[n])), n_hot35=yc(sd & (t_ >= 35.0)))
            exv = ex if np.isfinite(ex) else 0.0
            if kind == 'warm':
                for nm, f in FV.items():
                    f = freg[r_] if f is None else f; dl = f * max(exv, 0.0)
                    rec[f'att_{nm}'] = yc(expo & (a_ >= P90[n]) & (a_ - dl < P90[n]))
                dl = F_MAIN * max(exv, 0.0); rec['delta'] = dl
                rec['att95'] = yc(expo & (a_ >= P95[n]) & (a_ - dl < P95[n]))
                rec['att35'] = yc(expo & (t_ >= 35.0) & (t_ - dl < 35.0))
                rec['degday'] = dl * yc(expo)
            else:
                for nm, f in (('cool0.16', F_COOL), ('cool1.00', 1.0)):
                    dl = f * min(exv, 0.0)                     # negative
                    rec[f'prev_{nm}'] = yc(expo & (a_ < P90[n]) & (a_ - dl >= P90[n]))
                rec['delta'] = F_COOL * min(exv, 0.0)
            cnt.append(rec)
    C = pd.DataFrame(cnt)
    say(f'counts done {time.time() - T0:.0f} s')
    W_ = C[C.kind == 'warm']; Co = C[C.kind == 'cool']
    stack = lambda D, col: np.stack(D[col].values)              # [rows, ny]

    def pdays(D, col, pop=None):
        """person-days per year [ny]; pop None = 2020 population, 'tv' = year-varying"""
        M = stack(D, col)
        P = D.pop2020.values[:, None] if pop is None else popy[:, D.u.values].T
        return (M * P).sum(0)

    # hot-day denominator over distinct warm grid points
    Wu = W_.drop_duplicates('cell')
    hot_tot = pdays(Wu, 'n_hot'); hot95_tot = pdays(Wu, 'n_hot95'); hot35_tot = pdays(Wu, 'n_hot35')
    rng = np.random.default_rng(20261101); BI = rng.integers(0, ny, (a.nboot, ny))
    def boot(v): return np.percentile(v[BI].mean(1), [2.5, 97.5])
    def boot_ratio(v, w): return np.percentile(v[BI].sum(1) / w[BI].sum(1), [2.5, 97.5])

    Q = {nm: pdays(W_, f'att_{nm}') for nm in FV}
    q = Q['f0.50']; qm = q.mean(); qlo, qhi = boot(q)
    share = q.sum() / hot_tot.sum(); slo, shi = boot_ratio(q, hot_tot)
    share34 = Q['f0.34'].sum() / hot_tot.sum()
    pop_w = Wu.pop2020.sum()
    say(''); say('================ MAIN (23 warm regions, 2020 population, 1980-2021 mean per year)')
    say(f'population of the warm-region grid points (2020): {pop_w / 1e6:.0f} million in {len(Wu)} grid points')
    say(f'exposure person-days per year: {pdays(W_, "n_exp").mean() / 1e9:.2f} bn; '
        f'storm-season hot person-days (P90) per year: {hot_tot.mean() / 1e9:.2f} bn')
    say(f'Q = storm-caused hot person-days per year (f = 0.50): {qm / 1e9:.3f} bn [{qlo / 1e9:.3f}, {qhi / 1e9:.3f}] (year bootstrap)')
    say(f'share of storm-season hot person-days: {100 * share:.2f}% [{100 * slo:.2f}, {100 * shi:.2f}]')
    for nm in FV:
        say(f'  {nm}: Q {Q[nm].mean() / 1e9:.3f} bn, share {100 * Q[nm].sum() / hot_tot.sum():.2f}%')
    a95 = pdays(W_, 'att95'); a35 = pdays(W_, 'att35')
    say(f'  P95 hot days: Q {a95.mean() / 1e9:.3f} bn, share {100 * a95.sum() / hot95_tot.sum():.2f}%')
    say(f'  Tmax >= 35 C: Q {a35.mean() / 1e9:.3f} bn, share {100 * a35.sum() / max(hot35_tot.sum(), 1):.2f}% '
        f'(hot person-days >= 35 C per year {hot35_tot.mean() / 1e9:.2f} bn)')
    say(f'person-degree-days per year (f = 0.50): {pdays(W_, "degday").mean() / 1e9:.3f} bn K day')
    # people with at least one storm-caused hot day in a year (counts summed over regions per grid point)
    ge1 = 0.0
    for c, g in W_.groupby('cell'):
        ge1 += g.pop2020.iloc[0] * (np.stack(g['att_f0.50'].values).sum(0) >= 1).mean()
    say(f'people with >= 1 storm-caused hot day per year (mean over years): {ge1 / 1e6:.0f} million')
    qtv = pdays(W_, 'att_f0.50', 'tv')
    sl = lambda v: np.polyfit(years, v, 1)[0] * 10
    say(f'trend 1980-2021: year-varying population {sl(qtv) / 1e6:+.1f} M person-days per decade '
        f'({100 * sl(qtv) / qtv.mean():+.1f}% of mean); 2020 population {sl(q) / 1e6:+.1f} M per decade ({100 * sl(q) / qm:+.1f}%)')

    # regions and continents
    RR = []
    for r_, g in W_.groupby('region', sort=False):
        v = pdays(g, 'att_f0.50'); gu = g.drop_duplicates('cell')
        RR.append(dict(region=r_, kind='warm', cells=len(g), pop2020_M=gu.pop2020.sum() / 1e6, f_region=freg[r_],
                       delta_mean=g.delta.mean(), exp_days_per_yr=stack(g, 'n_exp').sum(1).mean() / ny,
                       Q_M=v.mean() / 1e6, Q_share=v.mean() / qm, Q_fregion_M=pdays(g, 'att_fregion').mean() / 1e6,
                       hot_share=v.sum() / max(pdays(gu, 'n_hot').sum(), 1)))
    for r_, g in Co.groupby('region', sort=False):
        gu = g.drop_duplicates('cell')
        RR.append(dict(region=r_, kind='cool', cells=len(g), pop2020_M=gu.pop2020.sum() / 1e6, delta_mean=g.delta.mean(),
                       exp_days_per_yr=stack(g, 'n_exp').sum(1).mean() / ny,
                       prev016_M=pdays(g, 'prev_cool0.16').mean() / 1e6, prev100_M=pdays(g, 'prev_cool1.00').mean() / 1e6))
    RR = pd.DataFrame(RR); RR.to_csv(os.path.join(OUT, 'impact_regions.csv'), index=False, float_format='%.4g')
    say(''); say('REGIONS (warm: Q in million person-days per year at f = 0.50; cool: hot person-days prevented)')
    say(RR.round(3).to_string(index=False))
    cs = {k: RR[RR.region.isin([f'W{i}' for i in v])].Q_M.sum() * 1e6 / qm for k, v in CONT.items()}
    say(''); say('continent shares of Q: ' + ', '.join(f'{k} {100 * v:.1f}%' for k, v in cs.items()))
    p16 = pdays(Co, 'prev_cool0.16'); p100 = pdays(Co, 'prev_cool1.00')
    say(f'cool regions: hot person-days prevented per year {p16.mean() / 1e9:.3f} bn (f_cool 0.16), {p100.mean() / 1e9:.3f} bn (map, f = 1); '
        f'net with Q: {(qm - p16.mean()) / 1e9:.3f} bn and {(qm - p100.mean()) / 1e9:.3f} bn')

    # verdict (design section 6)
    M1 = qm >= 1e9; M2 = share >= 0.10
    if M1:
        R_ = (Q['f0.34'].mean() >= 0.5e9) and (qlo >= 0.5e9); rnote = f'Q(f=0.34) {Q["f0.34"].mean() / 1e9:.3f} bn, Q lower {qlo / 1e9:.3f} bn (need >= 0.5)'
    elif M2:
        R_ = (share34 >= 0.05) and (slo >= 0.05); rnote = f'share(f=0.34) {100 * share34:.2f}%, share lower {100 * slo:.2f}% (need >= 5%)'
    else:
        R_ = False; rnote = 'not evaluated (no scale criterion met)'
    top = RR[RR.kind == 'warm'].Q_share.max(); n10 = sum(v >= 0.10 for v in cs.values())
    B_ = (top < 0.5) and (n10 >= 2)
    say(''); say('================ VERDICT (frozen criteria)')
    say(f'M1 Q >= 1.0 bn: {qm / 1e9:.3f} -> {"met" if M1 else "not met"}; M2 share >= 10%: {100 * share:.2f}% -> {"met" if M2 else "not met"}')
    say(f'R robustness: {rnote} -> {"met" if R_ else "not met"}')
    say(f'B breadth: largest region {100 * top:.1f}% (< 50%), continents >= 10%: {n10} (>= 2) -> {"met" if B_ else "not met"}')
    say(f'A3 {"PASS" if ((M1 or M2) and R_ and B_) else "FAIL"}')

    pd.DataFrame(dict(year=years, Q_pop2020=q, Q_popvarying=qtv, hot_pop2020=hot_tot, Q_f034=Q['f0.34'], Q_f061=Q['f0.61'],
                      Q_fregion=Q['fregion'], Q_P95=a95, Q_35C=a35, prev_cool016=p16, prev_cool100=p100,
                      pop_warm=popy[:, Wu.u.values].sum(1))).to_csv(os.path.join(OUT, 'impact_years.csv'), index=False, float_format='%.6g')
    cols = [c for c in C.columns if any(isinstance(v, np.ndarray) for v in C[c])]
    CC = C.drop(columns=cols).copy()
    for c in cols:
        CC[c + '_per_yr'] = [np.asarray(v).mean() if isinstance(v, np.ndarray) else np.nan for v in C[c]]
    CC.to_csv(os.path.join(OUT, 'impact_cells.csv'), index=False, float_format='%.5g')
    say(f'total {time.time() - T0:.0f} s')
    open(os.path.join(OUT, 'impact_result.txt'), 'w').write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
