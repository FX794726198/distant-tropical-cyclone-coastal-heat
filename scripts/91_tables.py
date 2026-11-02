"""Supplementary tables from the analysis outputs (markdown pipe tables for the build script).
Usage: python 91_tables.py OBS_DIR FIGPACK_DIR DB_DIR PARTS_DIR
Writes tabS1.md (stations), tabS2.md (observational robustness), tabS3.md (model cases), tabS4.md (GraphCast checks; descriptive since v08, no pass/fail marks)."""
import os, sys
import numpy as np, pandas as pd

obs, gcx, db, parts = sys.argv[1:5]
C = pd.read_csv(f'{gcx}/fp_case.csv', keep_default_na=False, na_values=['']); C['variant'] = C.variant.fillna('NULL')
cases = pd.read_csv(f'{db}/gc_cases.csv'); trk = pd.read_csv(f'{db}/gc_tracks.csv')
I = pd.read_csv(f'{gcx}/fp_ib.csv'); S = pd.read_csv(f'{gcx}/fp_ibstorms.csv', keep_default_na=False).set_index('s')
I['date'] = pd.Timestamp('1980-01-01') + pd.to_timedelta(I.day, 'D')


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fm(v, d=2):
    # signed value with a true minus sign, rounded half away from zero (the text uses the same rule)
    q = np.sign(v) * np.floor(abs(v) * 10 ** d + 0.5) / 10 ** d
    return f'{q:+.{d}f}'.replace('-', '−')


def ci(lo, hi, d=2):
    a, b = fm(lo, d).lstrip('+'), fm(hi, d).lstrip('+')
    return f'{a} to {b}' if lo < 0 else f'{a}–{b}'


def name_of(case, V):
    p = trk[(trk.case == case) & (trk.time == V)]
    if not len(p):
        return '–'
    q = I[I.date == pd.Timestamp(V[:10])]
    if not len(q):
        return '–'
    d = hav(p.lon.iloc[0], p.lat.iloc[0], q.lon10 / 10, q.lat10 / 10)
    j = int(np.argmin(d))
    if d.iloc[j] > 300:
        return '–'
    s = S.loc[q.s.iloc[j]]
    return f'{s["name"].capitalize()} {s.season}'


P = C.pivot_table(index=['model', 'case'], columns='variant', values='t2m')
used = C.drop_duplicates('case')[['case', 'kind', 'V']].merge(cases[['case', 'dist', 'brg']], on='case').sort_values(['kind', 'V'])
rows = ['| Case | Date (14 LT) | Storm | Distance (km) | Bearing (°) | Effect, 1° full (K) | Effect, 0.25° full (K) |', '|---|---|---|---|---|---|---|']
for kind, lab in (('heat', 'Hot days, storm to the east'), ('placebo', 'Storm to the southwest')):
    rows.append(f'| *{lab}* | | | | | | |')
    for _, r in used[used.kind == kind].iterrows():
        e1 = P.loc[('small', r.case), 'CTL'] - P.loc[('small', r.case), 'WIDE']
        e2 = P.loc[('oper', r.case), 'CTL'] - P.loc[('oper', r.case), 'WIDE']
        rows.append(f'| {r.case} | {r.V[:10]} | {name_of(r.case, r.V)} | {r.dist:,.0f} | {r.brg:.0f} | {e1:+.1f} | {e2:+.1f} |')
open(f'{parts}/tabS3.md', 'w').write('\n'.join(rows) + '\n')

# S1: stations and their east-side excess
if os.path.exists(f'{obs}/stations.csv'):
    st = pd.read_csv(f'{obs}/stations.csv', dtype={'station': str})
    gm = {}
    if os.path.exists(f'{obs}/ghcn_gsod_match.csv'):
        g = pd.read_csv(f'{obs}/ghcn_gsod_match.csv', dtype={'nearest_gsod': str})
        gm = {r.nearest_gsod: (r.ghcn, r.dist_km) for _, r in g.iterrows() if r.dist_km <= 50}
    rows = ['| Station (GSOD id) | Latitude (° N) | Longitude (° E) | Excess on east-side days (K) | 95% interval (K) | GHCN-Daily station (distance) |',
            '|---|---|---|---|---|---|']
    for _, r in st.sort_values('lon').iterrows():
        gh = gm.get(str(r.station))
        ghs = '–' if gh is None else (f'{gh[0]} (same site)' if gh[1] < 1 else f'{gh[0]} ({gh[1]:.0f} km)')
        rows.append(f'| {r.station} | {r.lat:.2f} | {r.lon:.2f} | {fm(r.excess)} | {ci(r.lo, r.hi)} | {ghs} |')
    open(f'{parts}/tabS1.md', 'w').write('\n'.join(rows) + '\n')

# S2: robustness of the observed excess
if os.path.exists(f'{obs}/robustness.csv'):
    rb = pd.read_csv(f'{obs}/robustness.csv')
    NAME = {'base: Z500, K=5, TC mask 1000 km': 'Main analysis (500-hPa height, five analogues, storms masked within 1,000 km)',
            'K=3': 'Three analogues', 'K=10': 'Ten analogues', 'TC mask 800 km': 'Storms masked within 800 km',
            'TC mask 1500 km': 'Storms masked within 1,500 km', 'Z500 + 850-hPa T, u, v': 'Analogues on 500-hPa height and 850-hPa temperature and wind',
            'pool: no TC within 1700 km': 'Analogue days with no storm within 1,700 km', '1980-2000': '1980–2000', '2001-2022': '2001–2022',
            'GHCN-Daily stations': 'GHCN-Daily stations instead of GSOD', 'east, 300-700 km': 'Storm 300–700 km away',
            'east, 700-1300 km': 'Storm 700–1,300 km away', 'east, tropical storm': 'Tropical storm (34–63 kt)',
            'east, typhoon cat 1-2': 'Typhoon, category 1–2 (64–95 kt)', 'east, typhoon cat 3-5': 'Typhoon, category 3–5 (≥96 kt)',
            'random TC-free days (null)': 'Random storm-free days (null test)'}
    rows = ['| Variant | Days | Storms | Excess (K) | 95% interval (K) |', '|---|---|---|---|---|']
    for _, r in rb.iterrows():
        if not np.isfinite(r.get('excess', np.nan)):
            continue
        ns = '–' if 'random' in r.label else f'{int(r.n_storms)}'
        rows.append(f'| {NAME.get(r.label, r.label)} | {int(r.n_days)} | {ns} | {fm(r.excess)} | {ci(r.lo, r.hi)} |')
    open(f'{parts}/tabS2.md', 'w').write('\n'.join(rows) + '\n')
print('tables written to', parts)

# S4: the criteria fixed before the GraphCast experiments, per configuration (hot cases unless stated)
REGEN = {('small', 'NOTC'): 0.18, ('small', 'WIDE'): 0.01, ('oper', 'NOTC'): 0.13, ('oper', 'WIDE'): 0.01}   # from gc_analyse.py summaries (deficit > 3 hPa)
Pt = {v: C.pivot_table(index=['model', 'kind', 'case', 'tid'], columns='variant', values=v) for v in ('t2m', 't850', 'T_truth')}
ok = lambda b: ''   # v08: checks are reported as values, without pass/fail marks
cols = [('small', 'NOTC', 'WEAK', 'NULL', '1°, core'), ('small', 'WIDE', 'WIDEH', 'NULLW', '1°, full'),
        ('oper', 'NOTC', 'WEAK', 'NULL', '0.25°, core'), ('oper', 'WIDE', 'WIDEH', 'NULLW', '0.25°, full')]
crit = [[] for _ in range(8)]
for m, v, wk, nl, lab in cols:
    t = Pt['t2m'].loc[(m, 'heat')]; t8 = Pt['t850'].loc[(m, 'heat')]; tr = Pt['T_truth'].loc[(m, 'heat')]
    tid = t.index.get_level_values('tid').values
    err = t['CTL'] - tr['CTL']; d = t['CTL'] - t[v]
    rng = np.random.default_rng(0); ug = np.unique(tid); gi = {u: np.where(tid == u)[0] for u in ug}
    bs = [d.values[np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])].mean() for _ in range(4000)]
    lo = np.percentile(bs, 2.5)
    pl = Pt['t2m'].loc[(m, 'placebo')]
    crit[0].append(f'{(err.abs() <= 1).mean():.0%} {ok((err.abs() <= 1).mean() >= 0.7)}')
    crit[1].append(f'{REGEN[(m, v)]:.0%} {ok(REGEN[(m, v)] <= 0.2)}')
    crit[2].append(f'{d.mean():+.2f} K, {(d > 0).mean():.0%} {ok(d.mean() >= 0.8 and lo > 0 and (d > 0).mean() >= 0.7)}')
    dh = (t['CTL'] - t[wk]).mean()
    crit[3].append(f'{dh:+.2f} K {ok(0 < dh < d.mean())}')
    if v == 'NOTC':
        sh = (t['NOTC'] - t['SHIFT']).mean()
        crit[4].append(f'{-sh:+.2f} K {ok(sh > 0)}')
    else:
        crit[4].append('–')
    dn = (t['CTL'] - t[nl]).mean()
    crit[5].append(f'{dn:+.2f} K {ok(abs(dn) < 0.2)}')
    dp = (pl['CTL'] - pl[v]).mean()
    crit[6].append(f'{dp:+.2f} K {ok(dp <= 0)}')
    d8 = (t8['CTL'] - t8[v]).mean()
    crit[7].append(f'{d8:+.2f} K {ok(d8 >= 0.4)}')
names = ['Cases with the control forecast within 1 K of ERA5 at the stations', 'Runs in which the removed storm regenerates',
         'Effect of the storm; share of cases with a positive effect', 'Effect with half of the storm removed',
         'Change when the storm core is moved to the southwest (core removal only)', 'Same edit at a storm-free point (null edit)',
         'Effect of the storm in 13 cases with the storm to the southwest', 'Effect of the storm on 850-hPa temperature']
rows = ['| Check | ' + ' | '.join(c[4] for c in cols) + ' |', '|---|---|---|---|---|']
for nm, cr in zip(names, crit):
    rows.append(f'| {nm} | ' + ' | '.join(c.strip() for c in cr) + ' |')
open(f'{parts}/tabS4.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
