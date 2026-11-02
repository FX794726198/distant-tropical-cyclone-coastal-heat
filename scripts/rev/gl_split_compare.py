"""Split-sample check of the global map (independent review, 2026-10-29).

Reads $TCH/global/out (all years), out_odd and out_even (gl_map_split.py with YEARSET=odd / even) and the stage-2 tests
out_odd_test_even and out_even_test_odd (gl_stage2_split.py: regions found with one half of the years, tested with the
other half).  For every region of the all-year map: the mean excess of its cells for its storm direction in each half.
For the cells: agreement of the excess between the halves.
Usage: python gl_split_compare.py   ->  $TCH/global/split/split_regions.csv, split_cells.csv, split_summary.txt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from gl_stage2 import region_keys

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')); GL = os.path.join(TCH, 'global'); OUT = os.path.join(GL, 'split')
os.makedirs(OUT, exist_ok=True)


def load(d):
    c = pd.read_csv(os.path.join(GL, d, 'gl_cells.csv'))
    for k in ('valid', 'sig_any', 'sig_best', 'sig_worst'):
        c[k] = c[k].astype(str).str.lower().eq('true')
    return c


F, O, E = load('out'), load('out_odd'), load('out_even')
assert (F.i.values == O.i.values).all() and (F.j.values == E.j.values).all() and (O.j.values == E.j.values).all()
st = np.load(os.path.join(GL, 'static_1deg.npz')); nlon, nlat = len(st['lon']), len(st['lat'])
rowof = {(i, j): r for r, (i, j) in enumerate(zip(F.i.values, F.j.values))}
L = []
rows = []
for sign, col in ((1, 'best'), (-1, 'worst')):
    for k, key in enumerate(region_keys(F[F.valid], f'ex_{col}', f'sig_{col}', sign, nlon, nlat)):
        r = np.array([rowof[x] for x in key])
        sec = pd.Series(F[f'sec_{col}'].values[r]).mode().iloc[0]
        d = dict(region=f"{'W' if sign > 0 else 'C'}{k + 1}", sign='warm' if sign > 0 else 'cool', sector=sec, n_cells=len(r),
                 lon=float(np.degrees(np.angle(np.exp(1j * np.radians(F.lon.values[r])).mean()))), lat=float(F.lat.values[r].mean()))
        for nm, X in (('full', F), ('odd', O), ('even', E)):
            ok = X.valid.values[r]
            d[f'ex_{nm}'] = float(np.nanmean(X[f'ex_{sec}'].values[r][ok])) if ok.any() else np.nan
            d[f'n_{nm}'] = int(np.nansum(X[f'n_{sec}'].values[r][ok]))
            d[f'cells_{nm}'] = int(ok.sum())
            d[f'frac_same_sign_{nm}'] = float((sign * X[f'ex_{sec}'].values[r][ok] > 0).mean()) if ok.any() else np.nan
        rows.append(d)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'split_regions.csv'), index=False, float_format='%.4g')

both = F.valid & O.valid & E.valid
C = pd.DataFrame(dict(i=F.i, j=F.j, lon=F.lon, lat=F.lat, full_valid=F.valid, ex_any_full=F.ex_any, ex_any_odd=O.ex_any,
                      ex_any_even=E.ex_any, sec_odd=O.sec_best, ex_best_odd=O.ex_best, sig_best_odd=O.sig_best,
                      sec_even=E.sec_best, ex_best_even=E.ex_best, sig_best_even=E.sig_best))
# excess of the other half for the direction picked by this half
C['even_at_odd_sec'] = [E[f'ex_{s}'].values[n] if isinstance(s, str) else np.nan for n, s in enumerate(O.sec_best.values)]
C['odd_at_even_sec'] = [O[f'ex_{s}'].values[n] if isinstance(s, str) else np.nan for n, s in enumerate(E.sec_best.values)]
C = C[both].reset_index(drop=True)
C.to_csv(os.path.join(OUT, 'split_cells.csv'), index=False, float_format='%.3g')

lines = [f'cells valid in all three maps: {len(C)}',
         f'corr(ex_any odd, ex_any even) = {np.corrcoef(C.ex_any_odd, C.ex_any_even)[0, 1]:.3f}']
for a_, b_, s_, x_ in (('odd', 'even', 'sig_best_odd', 'even_at_odd_sec'), ('even', 'odd', 'sig_best_even', 'odd_at_even_sec')):
    m = C[s_] & (C[f'ex_best_{a_}'] >= 0.5)
    lines.append(f'cells warm and significant in {a_} years: {int(m.sum())}; in {b_} years, same direction: '
                 f'mean {C.loc[m, x_].mean():+.3f} K, share > 0 {(C.loc[m, x_] > 0).mean():.3f}')
for s in ('warm', 'cool'):
    g = R[R.sign == s]; sg = 1 if s == 'warm' else -1
    lines.append(f'{s} regions of the all-year map: {len(g)}; same sign in odd years {int((sg * g.ex_odd > 0).sum())}, '
                 f'in even years {int((sg * g.ex_even > 0).sum())}, in both {int(((sg * g.ex_odd > 0) & (sg * g.ex_even > 0)).sum())}')
for d in ('out_odd_test_even', 'out_even_test_odd'):
    f = os.path.join(GL, d, 'gl_s2_regions.csv')
    if not os.path.exists(f):
        lines.append(f'{d}: missing'); continue
    S = pd.read_csv(f); S = S[S.sign.isin(['warm', 'cool'])]
    for s in ('warm', 'cool'):
        g = S[S.sign == s]; sg = 1 if s == 'warm' else -1
        lines.append(f'{d} {s}: {len(g)} regions tested, {int(g.tmax_ex.notna().sum())} with >= 10 days; same sign '
                     f'{int((sg * g.tmax_ex > 0).sum())}; CI on that side of 0 {int(((g.tmax_lo > 0) if sg > 0 else (g.tmax_hi < 0)).sum())}; '
                     f'FDR {int(g.tmax_sig_fdr.astype(str).str.lower().eq("true").sum())}; median {g.tmax_ex.median():+.2f} K')
open(os.path.join(OUT, 'split_summary.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print(R.round(2).to_string())
