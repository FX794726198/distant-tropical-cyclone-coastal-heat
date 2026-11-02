"""A4 GraphCast mechanism per region (exploratory), from the A2 case files.

Per case: CTL minus WIDEL at V (14 local time), mean over the region's grid points and the leads present, for t2m, t850,
q850, w500 and w850 (Pa/s, positive = more sinking with the storm), z500, msl, tp6, the onshore 10-m wind -(du, dv).n and
the alongshore wind (du, dv).(-ny, nx) with the region's seaward unit vector n (coast_normals.py).  Also the CTL values
(background with the storm present: ctl_*), the ERA5 2-m temperature at the grid points (truth_t2m) and the 2-m effect
for each lead (t2m_L24, t2m_L36, t2m_L48).  Region value: mean over storms of each storm's case mean; 95% interval from
resampling storms (2000, seed 7, regions in the fixed order W1..W23, C1..C18).  Pooled values (equal weight per region,
storms resampled within regions) for several groups of regions; correlations and a regression of the 2-m effect on the
mechanism terms across regions and cases.  Exploratory.

Usage: python mech_gc.py CASEDIR_OPER CASEDIR_SMALL A2_REGIONS_CSV COAST_NORMALS_CSV OUTDIR [MECH_ERA5_CSV]
(A2_REGIONS_CSV = all_oper_regions.csv of A2, with columns region, kind, usable, consistent, d_free; all_{model}_cases.csv
in the same folder, if present, is used to check t2m against A2's d_free)"""
import glob, os, sys
import numpy as np, pandas as pd, xarray as xr

DOPER, DSMALL, A2R, NRM, OUT = sys.argv[1:6]
ERA = sys.argv[6] if len(sys.argv) > 6 else ''   # optional mech_era5.csv for the dry/moist groups
os.makedirs(OUT, exist_ok=True)
VARS = ['t2m', 't850', 'q850', 'w500', 'w850', 'z500', 'msl', 'tp6', 'onshore', 'along']
BG = ['ctl_t2m', 'ctl_t850', 'ctl_q850', 'ctl_w500', 'ctl_tp6', 'ctl_onshore', 'ctl_along', 'truth_t2m']
ORDER = [f'W{i}' for i in range(1, 24)] + [f'C{i}' for i in range(1, 19)]
NR = pd.read_csv(NRM).set_index('region')
A2 = pd.read_csv(A2R).set_index('region')
for c in ('usable', 'consistent'):
    A2[c] = A2[c].astype(str).str.lower().eq('true')


def cases(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'case_*.nc'))):
        with xr.open_dataset(f) as ds:
            r = ds.attrs['region']; nx, ny = float(NR.loc[r, 'nx']), float(NR.loc[r, 'ny'])
            if 'WIDEL' not in ds.variant.values or 'CTL' not in ds.variant.values:
                continue
            has = lambda k: f'stn_{k}' in ds
            ctl = lambda k: ds[f'stn_{k}'].sel(variant='CTL')
            dd = lambda k: ctl(k) - ds[f'stn_{k}'].sel(variant='WIDEL')
            rec = dict(case=int(ds.attrs['case']), region=r, kind=ds.attrs['kind'], sid=ds.attrs['sid'],
                       V=str(ds.attrs.get('V', '')), nlead=int(ds.lead.size), ncell=int(ds.station.size))
            for k in ('t2m', 't850', 'q850', 'w500', 'w850', 'z500', 'msl', 'tp6'):
                rec[k] = float(dd(k).mean()) if has(k) else np.nan
            for k in ('t2m', 't850', 'q850', 'w500', 'tp6'):
                rec['ctl_' + k] = float(ctl(k).mean()) if has(k) else np.nan
            du, dv = dd('u10'), dd('v10'); cu, cv = ctl('u10'), ctl('v10')
            rec['onshore'] = float((-(du * nx + dv * ny)).mean()); rec['along'] = float((du * -ny + dv * nx).mean())
            rec['ctl_onshore'] = float((-(cu * nx + cv * ny)).mean()); rec['ctl_along'] = float((cu * -ny + cv * nx).mean())
            rec['truth_t2m'] = float(ds['truth_stn_t2m'].mean()) if 'truth_stn_t2m' in ds else np.nan
            for L in (24, 36, 48):
                rec[f't2m_L{L}'] = float(dd('t2m').sel(lead=L).mean()) if L in ds.lead.values else np.nan
            rows.append(rec)
    C = pd.DataFrame(rows)
    C['month'] = pd.to_datetime(C.V.str.slice(0, 10), errors='coerce').dt.month
    return C


def region_table(C, rng):
    out, sm = [], {}
    for r in ORDER:
        g = C[C.region == r]
        if len(g) < 3:
            continue
        s = g.groupby('sid')[VARS + BG].mean(); sm[r] = s
        rec = dict(region=r, kind=g.kind.iloc[0], n_cases=len(g), n_storms=len(s))
        idx = [rng.integers(0, len(s), len(s)) for _ in range(2000)]
        for v in VARS:
            a = s[v].values; rec[v] = float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            if np.isfinite(a).any():
                bs = [np.nanmean(a[i]) if np.isfinite(a[i]).any() else np.nan for i in idx]
                rec[v + '_lo'], rec[v + '_hi'] = np.nanpercentile(bs, [2.5, 97.5])
                rec[v + '_pos'] = float(np.mean(g[v].values > 0))   # share of cases with a positive effect
            else:
                rec[v + '_lo'] = rec[v + '_hi'] = rec[v + '_pos'] = np.nan
        for v in BG:
            rec[v] = float(np.nanmean(s[v].values)) if np.isfinite(s[v].values).any() else np.nan
        out.append(rec)
    return pd.DataFrame(out).set_index('region'), sm


def pooled(sm, regs, rng):
    est = {v: float(np.nanmean([sm[r][v].mean() for r in regs])) for v in VARS}
    bs = {v: [] for v in VARS}
    for _ in range(2000):
        pick = {r: sm[r].iloc[rng.integers(0, len(sm[r]), len(sm[r]))] for r in regs}
        for v in VARS:
            bs[v].append(np.nanmean([pick[r][v].mean() for r in regs]))
    return {v: (est[v], *np.nanpercentile(bs[v], [2.5, 97.5])) if np.isfinite(est[v]) else (np.nan,) * 3 for v in VARS}


lines = []
def say(s=''):
    print(s, flush=True); lines.append(s)

res = {}
for model, d in (('oper', DOPER), ('small', DSMALL)):
    C = cases(d); C.to_csv(os.path.join(OUT, f'mech_cases_{model}.csv'), index=False, float_format='%.5g')
    rng = np.random.default_rng(7)
    T, sm = region_table(C, rng)
    T['lat'] = [float(NR.loc[r, 'lat']) for r in T.index]
    T.to_csv(os.path.join(OUT, f'mech_regions_{model}.csv'), float_format='%.5g')
    cons = [r for r in ORDER if r in A2.index and A2.loc[r, 'kind'] == 'warm' and A2.loc[r, 'consistent'] and r in T.index]
    res[model] = (C, T, cons, sm)
    say(f'================ model {model}: {len(C)} cases, {len(T)} regions with >= 3 cases')
    ck = os.path.join(os.path.dirname(A2R), f'all_{model}_cases.csv')
    if os.path.exists(ck):
        chk = pd.read_csv(ck)
        if 'd_free' in chk:
            m = C.merge(chk[['case', 'd_free']], on='case')
            say(f'check against A2: {len(m)} cases matched, max |t2m - d_free| = {np.abs(m.t2m - m.d_free).max():.4f} K')
        if 'd850_free' in chk:
            m = C.merge(chk[['case', 'd850_free']], on='case')
            say(f'check against A2: max |t850 - d850_free| = {np.abs(m.t850 - m.d850_free).max():.4f} K')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        say(T[['kind', 'lat', 'n_cases', 'n_storms'] + VARS].round(3).to_string())
        say(''); say('background with the storm present (CTL) and ERA5 2-m temperature:')
        say(T[BG].round(3).to_string())

# ---------------------------------------------------------------- exploration
CL = pd.read_csv(ERA).set_index('region') if ERA and os.path.exists(ERA) else None
for model in ('oper', 'small'):
    C, T, cons, sm = res[model]
    rng = np.random.default_rng(11)
    say(''); say(f'================ exploration, model {model}')
    warm = [r for r in T.index if T.loc[r, 'kind'] == 'warm']; cool = [r for r in T.index if T.loc[r, 'kind'] == 'cool']
    groups = {'all usable warm': warm, 'A2-consistent warm': cons, 'warm, not consistent': [r for r in warm if r not in cons],
              'warm, |lat| < 30': [r for r in warm if abs(T.loc[r, 'lat']) < 30],
              'warm, |lat| >= 30': [r for r in warm if abs(T.loc[r, 'lat']) >= 30], 'usable cool': cool}
    if CL is not None:
        for c_ in ('dry', 'moist'):
            groups[f'warm, ERA5 {c_}'] = [r for r in warm if r in CL.index and CL.loc[r, 'class'] == c_]
    for nm, regs in groups.items():
        if not regs:
            continue
        Pg = pooled(sm, regs, rng)
        say(f'{nm} ({len(regs)}: {" ".join(regs)}):')
        say('    ' + '; '.join(f'{v} {Pg[v][0]:+.3g} [{Pg[v][1]:+.3g}, {Pg[v][2]:+.3g}]' for v in VARS if np.isfinite(Pg[v][0])))
    Tw = T.loc[warm]
    say(f'warm regions ({len(Tw)}) with w500 > 0: {int((Tw.w500 > 0).sum())}; onshore < 0: {int((Tw.onshore < 0).sum())}; '
        f't850 > 0: {int((Tw.t850 > 0).sum())}; all three: {int(((Tw.w500 > 0) & (Tw.onshore < 0) & (Tw.t850 > 0)).sum())}; '
        f'q850 < 0: {int((Tw.q850 < 0).sum())}; tp6 < 0: {int((Tw.tp6 < 0).sum())}; t2m > t850: {int((Tw.t2m > Tw.t850).sum())}')
    for lvl, D in (('regions (warm and cool)', T), ('cases (warm and cool, usable regions)', C[C.region.isin(T.index)])):
        say(f'correlation with t2m effect across {lvl}: ' +
            ', '.join(f'{v} {np.corrcoef(D[v], D.t2m)[0, 1]:+.2f}' for v in VARS[1:] if D[v].notna().all()))
    D = T.copy(); X = np.column_stack([np.ones(len(D))] + [(D[v] - D[v].mean()) / D[v].std() for v in ('w500', 'onshore', 't850')])
    b, *_ = np.linalg.lstsq(X, D.t2m.values, rcond=None)
    r2 = 1 - ((D.t2m.values - X @ b) ** 2).sum() / ((D.t2m.values - D.t2m.mean()) ** 2).sum()
    say(f'regions: t2m effect ~ standardised w500, onshore, t850: coefficients {b[1]:+.3f}, {b[2]:+.3f}, {b[3]:+.3f} K per SD, R2 {r2:.2f}')
    say('2-m effect by lead (mean over cases of warm usable regions): ' +
        ', '.join(f'{L} h {C[C.region.isin(warm)][f"t2m_L{L}"].mean():+.3f} K' for L in (24, 36, 48)))
open(os.path.join(OUT, 'mech_gc_result.txt'), 'w').write('\n'.join(lines) + '\n')
