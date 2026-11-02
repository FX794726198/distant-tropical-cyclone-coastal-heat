"""Local analysis of pack rev13b (or rev13b_small): effect of the storm for every removal definition, model and case kind,
with 95% intervals from resampling whole storms; paired differences between definitions; effect by lead time; the
control-forecast difference between the original and the revision runs (hardware check); new lows.
Usage: python -I rev_analyse.py PACKDIR DBDIR  (DBDIR holds gc_cases.csv)
"""
import os, sys
import numpy as np, pandas as pd

P, DBD = sys.argv[1], sys.argv[2]
cases = pd.read_csv(os.path.join(DBD, 'gc_cases.csv'))[['case', 'tid', 'kind']].rename(columns={'kind': 'kind_db'})


def boot(d, g, n=4000, seed=0):
    d = np.asarray(d, float); g = np.asarray(g); ok = np.isfinite(d); d, g = d[ok], g[ok]
    if len(d) == 0:
        return np.nan, np.nan, np.nan
    u = pd.Series(d).groupby(g); s, c = u.sum().values, u.count().values
    k = np.random.default_rng(seed).integers(0, len(s), (n, len(s))); m = s[k].sum(1) / c[k].sum(1)
    return d.mean(), *np.percentile(m, [2.5, 97.5])


E = pd.read_csv(os.path.join(P, 'effects_case.csv')).merge(cases, on='case', how='left')
E['model'] = E.model.fillna(E.dir.str.contains('oper').map({True: 'oper', False: 'small'}))
print('dirs and variants:'); print(E.groupby(['dir', 'model', 'kind', 'variant']).size().to_string())
rows = []
for (m, k, v), g in E.groupby(['model', 'kind', 'variant']):
    if v.startswith('CTL'):
        continue
    for col in ('d_t2m', 'd_t850', 'd_q850', 'd_msl'):
        mu, lo, hi = boot(g[col], g.tid)
        rows.append(dict(model=m, kind=k, variant=v, var=col, n=len(g), storms=g.tid.nunique(), mean=mu, lo=lo, hi=hi,
                         share_pos=float((g[col] > 0).mean()), dirs=','.join(sorted(g.dir.unique()))))
R = pd.DataFrame(rows)
print('\nEffect of the storm (control minus edited), 19-station mean, 14 LT, mean over leads; 95% CI resampling storms')
print(R[R['var'].isin(['d_t2m', 'd_t850'])].round(3).to_string(index=False))

# paired comparisons on the same cases
print('\nPaired differences of d_t2m between definitions (same model, kind, case)')
W = E.pivot_table(index=['model', 'kind', 'case', 'tid'], columns='variant', values='d_t2m').reset_index()
for m in W.model.unique():
    for k in W.kind.unique():
        g = W[(W.model == m) & (W.kind == k)]
        for a, b in (('WIDEL', 'WIDE'), ('W1000', 'WIDE'), ('WIDEL', 'NOTC'), ('W1000', 'NOTC'), ('WIDEL', 'W1000')):
            if a in g and b in g:
                d = (g[a] - g[b]).dropna()
                if len(d):
                    mu, lo, hi = boot(d, g.loc[d.index, 'tid'])
                    print(f'  {m:5s} {k:7s} {a}-{b}: {mu:+.2f} ({lo:+.2f} to {hi:+.2f}), n={len(d)}, ratio {g[a].mean() / g[b].mean():.2f}')

# control-forecast check: original runs (A100) vs revision runs (possibly V100)
print('\nControl minus ERA5 at the stations by output directory (heat cases; same cases and leads should agree if hardware does not matter)')
C = E[E.kind == 'heat'].groupby(['model', 'dir', 'case']).ctl_minus_truth_t2m.first().reset_index()
X = C.pivot_table(index=['model', 'case'], columns='dir', values='ctl_minus_truth_t2m')
print(X.describe().round(3).to_string())
for m in ('small', 'oper'):
    cols = [c for c in X.columns if (('oper' in c) == (m == 'oper')) and X.loc[m, c].notna().any()] if m in X.index.get_level_values(0) else []
    if len(cols) > 1:
        x = X.loc[m, cols].dropna()
        for c in cols[1:]:
            print(f'  {m}: {c} minus {cols[0]}: mean {(x[c] - x[cols[0]]).mean():+.3f} K, max |diff| {(x[c] - x[cols[0]]).abs().max():.3f} K over {len(x)} cases')

f = os.path.join(P, 'effects_lead.csv')
if os.path.exists(f):
    print('\nHeat-case mean by lead time'); print(pd.read_csv(f).round(2).to_string(index=False))
f = os.path.join(P, 'newlows.csv')
if os.path.exists(f):
    N = pd.read_csv(f); print('\nnewlows.csv columns:', list(N.columns), len(N)); print(N.head(20).to_string())
for m in ('small', 'oper'):
    f = os.path.join(P, f'steps_{m}.csv')
    if os.path.exists(f):
        S = pd.read_csv(f).pivot_table(index=['case', 'step_h'], columns='variant', values=['t2m', 't850', 'msl'])
        for v in ('t2m', 't850', 'msl'):
            S[(v, 'CTL-WIDE')] = S[(v, 'CTL')] - S[(v, 'WIDE')]; S[(v, 'CTL-WIDEL')] = S[(v, 'CTL')] - S[(v, 'WIDEL')]
        print(f'\nsteps {m}'); print(S[[(v, k) for v in ('t2m', 't850', 'msl') for k in ('CTL-WIDE', 'CTL-WIDEL')]].round(2).to_string())
R.to_csv(os.path.join(P, 'rev_effects_summary.csv'), index=False, float_format='%.4g')
