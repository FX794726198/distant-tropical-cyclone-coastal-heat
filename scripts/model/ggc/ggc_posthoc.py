import json, sys, numpy as np, pandas as pd
U = sys.argv[1]
rng = np.random.default_rng(7)
sel = pd.read_csv(f'{U}/inputs/gc_cases.csv').set_index('case')
reg = json.load(open(f'{U}/inputs/gcg_regions.json'))
ncell = {k: len(v['cells']) for k, v in reg.items()}

def storm_mean(df, col):
    w = 1.0 / df.groupby('sid').sid.transform('size'); return float((df[col] * w).sum() / w.sum())

def pooled(df, col, regions, nboot=2000):
    est = np.mean([storm_mean(df[df.region == g], col) for g in regions])
    parts = {g: {s: d for s, d in df[df.region == g].groupby('sid')} for g in regions}; bs = []
    for _ in range(nboot):
        vals = []
        for g in regions:
            ss = list(parts[g]); pick = rng.integers(0, len(ss), len(ss))
            vals.append(np.mean([parts[g][ss[k]][col].mean() for k in pick]))
        bs.append(np.mean(vals))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def region_boot(d, col, nboot=2000):
    parts = {s: g[col].values for s, g in d.groupby('sid')}; ss = list(parts); bs = []
    for _ in range(nboot):
        pick = rng.integers(0, len(ss), len(ss)); bs.append(np.mean([parts[ss[k]].mean() for k in pick]))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

for model in ('oper', 'small'):
    P = pd.read_csv(f'{U}/out_{model}/gcg_cases.csv'); R = pd.read_csv(f'{U}/out_{model}/gcg_regions.csv').set_index('region')
    P['anom'] = sel.anom.reindex(P.case).values; P['dist'] = sel.dist.reindex(P.case).values; P['ncell'] = P.region.map(ncell)
    usable = R[(R.kind == 'warm') & R.usable.astype(str).str.lower().eq('true')].index.tolist()
    W = P[P.region.isin(usable)].copy()
    print(f'\n######## {model}: {len(W)} warm cases in {len(usable)} usable regions; CTL-ERA5 mean {W.ctl_err.mean():+.2f} K, median {W.ctl_err.median():+.2f}')
    # (a) skill split
    sk = W[W.ctl_err.abs() <= 1]; ns = W[W.ctl_err.abs() > 1]
    for nm, d in (('|err|<=1', sk), ('|err|>1', ns)):
        rg = sorted(d.region.unique()); e, lo, hi = pooled(d, 'd_free', rg)
        print(f'(a) {nm}: {len(d)} cases, {len(rg)} regions: pooled d_free {e:+.2f} [{lo:+.2f},{hi:+.2f}]; null {pooled(d, "d_null", rg)[0]:+.2f}; mean anom {d.anom.mean():+.1f}')
    print(f'(a) corr(ctl_err, d_free) across warm cases: {np.corrcoef(W.ctl_err, W.d_free)[0,1]:+.2f}; corr(anom, d_free) {np.corrcoef(W.anom, W.d_free)[0,1]:+.2f}; corr(anom, ctl_err) {np.corrcoef(W.anom, W.ctl_err)[0,1]:+.2f}')
    # cold cases: too-cold (<-1) vs ok
    cold = W[W.ctl_err < -1]; print(f'(a) cases with CTL colder than ERA5 by > 1 K: {len(cold)} ({len(cold)/len(W):.0%}), their d_free {cold.d_free.mean():+.2f} vs others {W[W.ctl_err >= -1].d_free.mean():+.2f}')
    # skilful-only region table
    rows = []
    for g, d in sk.groupby('region'):
        if len(d) < 3: continue
        m = storm_mean(d, 'd_free'); pos = (d.d_free > 0).mean(); nul = storm_mean(d, 'd_null')
        rows.append(dict(region=g, n=len(d), d_free=m, pos=pos, null=nul, conf=(m >= 0.5 and pos >= 0.75 and nul < 0.5 * m)))
    T = pd.DataFrame(rows); print(f'(a) skilful cases only, regions with >=3: {len(T)}, confirmed {int(T.conf.sum())}')
    # (b) why failed
    rb = []
    for g in usable:
        d = W[W.region == g]; lo, hi = region_boot(d, 'd_free')
        rb.append(dict(region=g, n=len(d), ncell=ncell[g], conf=str(R.loc[g, 'confirmed']).lower() == 'true', d_free=storm_mean(d, 'd_free'), lo=lo, hi=hi,
                       d_full=storm_mean(d, 'd_full'), d_null=storm_mean(d, 'd_null'), err=storm_mean(d, 'ctl_err'), dist=d.dist.mean(), anom=d.anom.mean(),
                       share=storm_mean(d, 'd_free') / d.anom.mean() if abs(d.anom.mean()) > 0.2 else np.nan))
    B = pd.DataFrame(rb).set_index('region')
    print('(b) per region:'); print(B.round(2).to_string())
    for col in ('ncell', 'err', 'dist', 'anom', 'd_null'):
        print(f'(b) {col}: confirmed mean {B[B.conf][col].mean():.2f} vs not {B[~B.conf][col].mean():.2f}')
    # (c) interval rule
    c = B[(B.lo > 0) & (B.d_free >= 0.5)]; c2 = B[(B.lo > 0)]
    print(f'(c) interval rule: mean>=0.5 & lo>0: {len(c)}/{len(B)} ({", ".join(c.index)}); lo>0 only: {len(c2)}/{len(B)}')
    # (d) share
    print(f'(d) share of anomaly explained: median over regions {B.share.median():.2f}, pooled d_free/pooled anom {B.d_free.mean()/B.anom.mean():.2f}; regions with share>=0.5: {(B.share>=0.5).sum()}')
    # (e) size
    B['trunc'] = B.d_full - B.d_free
    print(f'(e) corr(ncell, d_full-d_free) {np.corrcoef(B.ncell, B.trunc)[0,1]:+.2f}; corr(ncell, d_free) {np.corrcoef(B.ncell, B.d_free)[0,1]:+.2f}; corr(dist, d_free) {np.corrcoef(B.dist, B.d_free)[0,1]:+.2f}; corr(err, d_free) {np.corrcoef(B.err, B.d_free)[0,1]:+.2f}; corr(anom, d_free) {np.corrcoef(B.anom, B.d_free)[0,1]:+.2f}')
    B.round(3).to_csv(f'posthoc_{model}_regions.csv')
