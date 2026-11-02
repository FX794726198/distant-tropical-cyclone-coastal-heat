"""Storm removal in GraphCast on the coasts of the global map, second set (analysis A2: all usable exposure days 2016-2021,
at most two per storm; Supplementary Section S3).  Per-case quantities exactly as in gcg_eval.py (imported):
cell-mean 2-m temperature at V averaged over the leads; d_free = CTL - WIDEL (primary), d_full = CTL - WIDE, d_core = CTL - NOTC,
d_null = CTL - NULLW; each storm weight 1 within a region; pooled = equal-weight mean over regions; 95% intervals from resampling
storms within regions (2,000 draws, seed 7).
Usable region: >= 3 cases.  map warming of a region = ex_map in gcg_regions.json (the region-mean exposure-day anomaly of the
global map, Table S1).

Reported: cool mirror (pooled cool d_free, interval; regions with d_free < 0 and < d_null), the storm's
share of the map warming (d_free / map) and of the case-day anomaly, per-region table, hotter vs less hot days (split at the
region median anomaly), repeatability against analysis A for the overlapping cases (max |difference| of d_free).
Usage: python -I gcg_eval_all.py ROOT --main-root /scratch/$USER/tcheat/ggc
Output: ROOT/out_<model>/gcg_cases.csv, gcg_regions.csv; ROOT/gcg_all_result.txt
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gcg_eval import per_case, storm_mean, pooled, PAIRS  # noqa: E402


def region_table(P, ex):
    rows = []
    for g, d in P.groupby('region'):
        r = dict(region=g, kind=d.kind.iloc[0], n_cases=len(d), n_storms=d.sid.nunique(), n_new=int(d.new.sum()), ex_map=ex[g],
                 anom=storm_mean(d, 'anom'), ctl_err=storm_mean(d, 'ctl_err'))
        for k in PAIRS:
            dd = d[np.isfinite(d[f'd_{k}'])]
            r[f'd_{k}'] = storm_mean(dd, f'd_{k}') if len(dd) else np.nan
            r[f'pos_{k}'] = float((dd[f'd_{k}'] > 0).mean()) if len(dd) else np.nan
        r['d850_free'] = storm_mean(d, 'd850_free')
        r['usable'] = r['n_cases'] >= 3
        nul = r['d_null'] if np.isfinite(r['d_null']) else (0.0 if r['kind'] == 'warm' else 0.0)
        r['consistent'] = bool(r['usable'] and (r['d_free'] > max(0.0, nul) if r['kind'] == 'warm' else r['d_free'] < min(0.0, nul)))
        rows.append(r)
    return pd.DataFrame(rows).set_index('region')


def ratio_boot(df, num, den, regions, rng, nboot=2000):
    est = np.mean([storm_mean(df[df.region == g], num) for g in regions]) / np.mean([storm_mean(df[df.region == g], den) for g in regions])
    parts = {g: {s: d for s, d in df[df.region == g].groupby('sid')} for g in regions}; bs = []
    for _ in range(nboot):
        a_, b_ = [], []
        for g in regions:
            ss = list(parts[g]); pick = rng.integers(0, len(ss), len(ss))
            a_.append(np.mean([parts[g][ss[k]][num].mean() for k in pick])); b_.append(np.mean([parts[g][ss[k]][den].mean() for k in pick]))
        bs.append(np.mean(a_) / np.mean(b_))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--main-root', default='/scratch/$USER/tcheat/ggc'); ap.add_argument('--nboot', type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    L = [f'gcg_eval_all.py md5 {hashlib.md5(open(__file__, "rb").read()).hexdigest()}; gcg_eval.py md5 '
         f'{hashlib.md5(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcg_eval.py"), "rb").read()).hexdigest()}']
    cases = pd.read_csv(os.path.join(a.root, 'inputs', 'gc_cases.csv')).set_index('case')
    ex = {g: v['ex_map'] for g, v in json.load(open(os.path.join(a.root, 'inputs', 'gcg_regions.json'))).items()}
    mainf = os.path.join(a.main_root, 'out_oper', 'gcg_cases.csv')
    M = pd.read_csv(mainf) if os.path.exists(mainf) else pd.DataFrame(columns=['region', 'V', 'd_free'])
    seen = set(zip(M.region, M.V.astype(str)))
    L.append(f'cases {len(cases)}; analysis A cases found at {mainf}: {len(M)}')
    res = {}
    for model in ('oper', 'small'):
        odir = os.path.join(a.root, f'out_{model}')
        if not os.path.isdir(odir):
            continue
        P, RG = per_case(odir)
        if not len(P):
            continue
        P['anom'] = cases.anom.reindex(P.case).values
        P['new'] = [(g, str(v)) not in seen for g, v in zip(P.region, P.V)]
        P.to_csv(os.path.join(odir, 'gcg_cases.csv'), index=False, float_format='%.4f')
        T = region_table(P, ex); T.to_csv(os.path.join(odir, 'gcg_regions.csv'), float_format='%.3f')
        L += ['', f'================ model {model}: {len(P)} cases ({int(P.new.sum())} not run in A), {P.sid.nunique()} storms, '
                  f'{P.region.nunique()} regions', T.round(2).to_string()]
        wr = sorted(T[(T.kind == 'warm') & T.usable].index); cr = sorted(T[(T.kind == 'cool') & T.usable].index)
        W = P[P.region.isin(wr)]; C = P[P.region.isin(cr)]
        r = dict(nw=len(wr))
        if wr:
            r['free'] = pooled(W, 'd_free', rng, a.nboot, wr)
            Wn = W[np.isfinite(W.d_null)]; r['null'] = pooled(Wn, 'd_null', rng, a.nboot, sorted(Wn.region.unique()))
            r['full'] = pooled(W, 'd_full', rng, a.nboot, wr); r['core'] = pooled(W, 'd_core', rng, a.nboot, wr)
            r['map'] = float(np.mean([ex[g] for g in wr]))
            r['nconsist'] = int(T.loc[wr].consistent.sum())
            r['share_anom'] = ratio_boot(W, 'd_free', 'anom', wr, rng, a.nboot)
            Nw = W[W.new]; nr = sorted(g for g, d in Nw.groupby('region') if len(d) >= 2)
            r['new_n'] = (len(Nw[Nw.region.isin(nr)]), len(nr))
            r['new_free'] = pooled(Nw[Nw.region.isin(nr)], 'd_free', rng, a.nboot, nr) if nr else (np.nan, np.nan, np.nan)
            r['new_map'] = float(np.mean([ex[g] for g in nr])) if nr else np.nan
            hot = []
            for g in wr:
                d = W[W.region == g]; med = d.anom.median()
                hot.append((storm_mean(d[d.anom > med], 'd_free') if (d.anom > med).any() else np.nan,
                            storm_mean(d[d.anom <= med], 'd_free')))
            r['hot_split'] = (float(np.nanmean([h[0] for h in hot])), float(np.nanmean([h[1] for h in hot])))
            r['corr_anom'] = float(np.corrcoef(W.anom, W.d_free)[0, 1])
            r['err'] = (float(W.ctl_err.mean()), float((W.ctl_err.abs() <= 1).mean()))
            r['d850'] = float(storm_mean(W, 'd850_free'))
        if cr:
            r['cool'] = pooled(C, 'd_free', rng, a.nboot, cr); r['cool_map'] = float(np.mean([ex[g] for g in cr]))
            r['ncool'] = (int(T.loc[cr].consistent.sum()), len(cr))
        rg = RG[RG.variant == 'WIDEL'] if len(RG) else RG
        r['regrow'] = (float((rg.edited > 3).mean()) if len(rg) else np.nan, len(rg))
        if model == 'oper' and len(M):
            J = P.merge(M[['region', 'V', 'd_free']], on=['region', 'V'], suffixes=('', '_A'))
            r['repeat'] = (len(J), float((J.d_free - J.d_free_A).abs().max()) if len(J) else np.nan)
        res[model] = r
        L.append('')
        if wr:
            e, lo, hi = r['free']
            L += [f'usable warm regions {len(wr)}; pooled map warming {r["map"]:+.2f} K',
                  f'pooled warm d_free {e:+.2f} K [{lo:+.2f}, {hi:+.2f}] = {e / r["map"]:.0%} of the map warming '
                  f'[{lo / r["map"]:.0%}, {hi / r["map"]:.0%}]',
                  f'pooled warm d_full {r["full"][0]:+.2f} [{r["full"][1]:+.2f}, {r["full"][2]:+.2f}]; d_core {r["core"][0]:+.2f} '
                  f'[{r["core"][1]:+.2f}, {r["core"][2]:+.2f}]; d_null {r["null"][0]:+.2f} [{r["null"][1]:+.2f}, {r["null"][2]:+.2f}]',
                  f'warm regions with d_free > 0 and > d_null: {r["nconsist"]} of {len(wr)}',
                  f'share of the case-day anomaly: {r["share_anom"][0]:.0%} [{r["share_anom"][1]:.0%}, {r["share_anom"][2]:.0%}]',
                  f'cases not run in A: {r["new_n"][0]} in {r["new_n"][1]} warm regions with >= 2: pooled d_free {r["new_free"][0]:+.2f} '
                  f'[{r["new_free"][1]:+.2f}, {r["new_free"][2]:+.2f}], map warming {r["new_map"]:+.2f}',
                  f'hotter half of each region\'s days {r["hot_split"][0]:+.2f} K, less hot half {r["hot_split"][1]:+.2f} K; '
                  f'corr(case-day anomaly, d_free) {r["corr_anom"]:+.2f}',
                  f'CTL - ERA5, warm cases: {r["err"][0]:+.2f} K, within 1 K in {r["err"][1]:.0%}; 850-hPa d_free {r["d850"]:+.2f} K']
        if cr:
            L.append(f'cool mirror: pooled d_free {r["cool"][0]:+.2f} [{r["cool"][1]:+.2f}, {r["cool"][2]:+.2f}] (map {r["cool_map"]:+.2f}); '
                     f'regions with d_free < 0 and < d_null: {r["ncool"][0]} of {r["ncool"][1]}')
        L.append(f'regrowth (deficit > 3 hPa at V), WIDEL: {r["regrow"][0]:.0%} of {r["regrow"][1]} runs')
        if 'repeat' in r:
            L.append(f'repeatability vs analysis A: {r["repeat"][0]} matching cases, max |d_free difference| {r["repeat"][1]:.3f} K')
    open(os.path.join(a.root, 'gcg_all_result.txt'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
