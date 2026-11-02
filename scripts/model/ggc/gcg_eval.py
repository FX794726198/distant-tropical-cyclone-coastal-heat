"""Storm removal in GraphCast on the coasts of the global map, first set of days (analysis A; Supplementary Section S3):
effects of the edits per case, per region and pooled.

Per case: cell-mean 2-m temperature at V, averaged over the leads; d_full = CTL - WIDE, d_free = CTL - WIDEL (region cells
untouched at the initial time), d_core = CTL - NOTC, d_null = CTL - NULLW, ctl_err = CTL - ERA5.  Per region: storm-weighted
means (each storm weight 1 inside the region).
The column 'confirmed' is a descriptive label: warm regions with mean d_free >= +0.5 K, d_free > 0 in >= 3/4 of their cases
and mean d_null < half of mean d_free; cool regions with mean d_free <= -0.3 K.  Pooled values give each region equal weight,
with 95% intervals from resampling storms within regions.  The 0.25-deg operational GraphCast is the main model, the 1-deg
model a check.
d_free (region-free removal) is the main effect because it edits nothing over the region; d_full is reported beside it.
Usage: python -I gcg_eval.py TCH_ROOT [--main oper] [--nboot 2000]
Output: TCH_ROOT/out_<model>/gcg_cases.csv, gcg_regions.csv; TCH_ROOT/gcg_result.txt
"""
import argparse, glob, hashlib, json, os
import numpy as np, pandas as pd, xarray as xr

PAIRS = {'full': 'WIDE', 'free': 'WIDEL', 'core': 'NOTC', 'null': 'NULLW'}


def per_case(odir):
    rows, regen = [], []
    for f in sorted(glob.glob(os.path.join(odir, 'case_*.nc'))):
        ds = xr.open_dataset(f); meta = json.load(open(f.replace('.nc', '.json')))
        em = ds.mean('lead')
        r = dict(case=ds.attrs['case'], region=ds.attrs['region'], kind=ds.attrs['kind'], sid=ds.attrs['sid'], V=ds.attrs['V'],
                 nlead=ds.sizes['lead'], ncell=ds.sizes['station'])
        T = {v: float(em.stn_t2m.sel(variant=v).mean()) for v in ds.variant.values}
        T8 = {v: float(em.stn_t850.sel(variant=v).mean()) for v in ds.variant.values}
        r['T_CTL'] = T['CTL']; r['ctl_err'] = T['CTL'] - float(ds.truth_stn_t2m.mean())
        nulls = [m.get('NULLW', [{}])[0].get('null_centre') for L, m in meta['leads'].items() if isinstance(m, dict)]
        for k, v in PAIRS.items():
            if v in T:
                r[f'd_{k}'] = T['CTL'] - T[v]; r[f'd850_{k}'] = T8['CTL'] - T8[v]
        if 'd_null' in r and all(n is None for n in nulls):
            r['d_null'] = np.nan; r['d850_null'] = np.nan
        for L, m in meta['leads'].items():
            if isinstance(m, dict):
                for v in ('NOTC', 'WIDE', 'WIDEL'):
                    if f'{v}_deficit_at_V' in m:
                        regen.append(dict(case=r['case'], lead=int(L), variant=v, ctl=m.get('CTL_deficit_at_V'), edited=m[f'{v}_deficit_at_V']))
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(regen)


def storm_mean(df, col):
    w = 1.0 / df.groupby('sid').sid.transform('size')
    return float((df[col] * w).sum() / w.sum())


def pooled(df, col, rng, nboot, regions):
    """Equal-weight mean over regions of storm-weighted region means; bootstrap resamples storms within each region."""
    est = np.mean([storm_mean(df[df.region == g], col) for g in regions])
    bs = []
    parts = {g: {s: d for s, d in df[df.region == g].groupby('sid')} for g in regions}
    for _ in range(nboot):
        vals = []
        for g in regions:
            ss = list(parts[g]); pick = rng.integers(0, len(ss), len(ss))
            vals.append(np.mean([parts[g][ss[k]][col].mean() for k in pick]))
        bs.append(np.mean(vals))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--main', default='oper'); ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    L = [f'gcg_eval.py md5 {hashlib.md5(open(__file__, "rb").read()).hexdigest()}']
    for model in ('oper', 'small'):
        odir = os.path.join(a.root, f'out_{model}')
        if not glob.glob(os.path.join(odir, 'case_*.nc')):
            continue
        P, RG = per_case(odir)
        P.to_csv(os.path.join(odir, 'gcg_cases.csv'), index=False, float_format='%.4f')
        reg_rows = []
        for g, d in P.groupby('region'):
            r = dict(region=g, kind=d.kind.iloc[0], n_cases=len(d), n_storms=d.sid.nunique(), ctl_err=storm_mean(d, 'ctl_err'))
            for k in PAIRS:
                if f'd_{k}' in d:
                    dd = d[np.isfinite(d[f'd_{k}'])]
                    r[f'd_{k}'] = storm_mean(dd, f'd_{k}') if len(dd) else np.nan
                    r[f'pos_{k}'] = float((dd[f'd_{k}'] > 0).mean()) if len(dd) else np.nan
                    r[f'd850_{k}'] = storm_mean(dd, f'd850_{k}') if len(dd) else np.nan
            r['usable'] = r['n_cases'] >= 3
            if r['kind'] == 'warm':
                r['confirmed'] = bool(r['usable'] and r['d_free'] >= 0.5 and r['pos_free'] >= 0.75
                                      and (np.isnan(r.get('d_null', np.nan)) or r['d_null'] < 0.5 * r['d_free']))
            else:
                r['confirmed'] = bool(r['usable'] and r['d_free'] <= -0.3)
            reg_rows.append(r)
        RR = pd.DataFrame(reg_rows)
        RR.to_csv(os.path.join(odir, 'gcg_regions.csv'), index=False, float_format='%.3f')
        L += ['', f'================ model {model}: {len(P)} cases, {P.sid.nunique()} storms, {P.region.nunique()} regions',
              RR.round(2).to_string(index=False)]
        W = P[(P.kind == 'warm') & P.region.isin(RR[RR.usable & (RR.kind == 'warm')].region)]
        Cc = P[(P.kind == 'cool') & P.region.isin(RR[RR.usable & (RR.kind == 'cool')].region)]
        wr = sorted(W.region.unique()); cr = sorted(Cc.region.unique())
        res = {}
        for k in PAIRS:
            if f'd_{k}' in W and len(wr):
                Wk = W[np.isfinite(W[f'd_{k}'])]
                res[k] = pooled(Wk, f'd_{k}', rng, a.nboot, sorted(Wk.region.unique()))
        nconf = int(RR[(RR.kind == 'warm') & RR.usable].confirmed.sum()); nus = int(((RR.kind == 'warm') & RR.usable).sum())
        ncc = int(RR[(RR.kind == 'cool') & RR.usable].confirmed.sum()); ncu = int(((RR.kind == 'cool') & RR.usable).sum())
        L += ['', f'warm regions with >= 3 cases: {nus} of {int((RR.kind == "warm").sum())}; confirmed {nconf} ({nconf / max(nus, 1):.0%})',
              f'cool regions with >= 3 cases: {ncu}; mirror (d_free <= -0.3 K) {ncc} ({ncc / max(ncu, 1):.0%})']
        for k, (e, lo, hi) in res.items():
            L.append(f'pooled warm d_{k} (CTL - {PAIRS[k]}): {e:+.2f} K [{lo:+.2f}, {hi:+.2f}] (storm bootstrap within regions)')
        if len(cr):
            e, lo, hi = pooled(Cc[np.isfinite(Cc.d_free)], 'd_free', rng, a.nboot, cr)
            L.append(f'pooled cool d_free: {e:+.2f} K [{lo:+.2f}, {hi:+.2f}]')
        L.append(f'CTL - ERA5 at V, warm cases: mean {W.ctl_err.mean():+.2f} K, within 1 K in {(W.ctl_err.abs() <= 1).mean():.0%}')
        rg = RG[RG.variant == 'WIDEL'] if len(RG) else RG
        regen_share = float((rg.edited > 3).mean()) if len(rg) else np.nan
        L.append(f'regrowth (deficit > 3 hPa at V): WIDEL {regen_share:.0%} of {len(rg)} runs; ' +
                 '; '.join(f'{v} {float((RG[RG.variant == v].edited > 3).mean()):.0%}' for v in ('NOTC', 'WIDE') if (RG.variant == v).any()))
        fp = W.d850_free.mean() if 'd850_free' in W else np.nan
        L.append(f'fingerprint: 850-hPa warming by the storm (CTL - WIDEL), warm cases: {fp:+.2f} K')
    open(os.path.join(a.root, 'gcg_result.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
