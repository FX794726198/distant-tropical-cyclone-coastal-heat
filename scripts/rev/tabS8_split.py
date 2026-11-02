"""Table S8: regions found with one half of the years and tested with the other half (split check, second review 2026-10-30).

Reads SPLIT/out_odd_test_even_s2.csv and out_even_test_odd_s2.csv (gl_stage2 on the split maps) and writes parts/tabS8.md.
q values are Benjamini-Hochberg adjusted P values within each test (half x warm/cool), the same family as the FDR flag.
Usage: python -I tabS8_split.py SPLIT PARTS
"""
import sys
import numpy as np, pandas as pd

split, parts = sys.argv[1:3]
fm = lambda v, d=2: (f'{np.sign(v) * np.floor(abs(v) * 10 ** d + 0.5) / 10 ** d:+.{d}f}').replace('-', '−')
ci = lambda lo, hi: f"{fm(lo).lstrip('+')} to {fm(hi).lstrip('+')}" if lo < 0 else f"{fm(lo).lstrip('+')}–{fm(hi).lstrip('+')}"
pv = lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'
deg = lambda x, a, b: f'{abs(x):.0f}° {a if x >= 0 else b}'


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1); return out


rows = ['| Region | Centre | Direction | Points | Days | Storms | Map change (K) | Excess, other half (K) | 95% interval (K) | *P* | *q* |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
for f, a_, b_ in (('out_odd_test_even_s2.csv', 'odd', 'even'), ('out_even_test_odd_s2.csv', 'even', 'odd')):
    d = pd.read_csv(f'{split}/{f}')
    for s in ('warm', 'cool'):
        g = d[d.sign == s].copy(); sg = 1 if s == 'warm' else -1
        g['q'] = bh(g.tmax_p.values)
        sig = g.tmax_sig_fdr.astype(str).str.lower().eq('true') & ((g.tmax_lo > 0) if sg > 0 else (g.tmax_hi < 0))
        rows.append(f'| *{s.capitalize()} regions found with the {a_} years, tested with the {b_} years* |  |  |  |  |  |  |  |  |  |  |')
        for (_, r), ok in zip(g.iterrows(), sig):
            mk = '' if ok else '^†^'
            rows.append(f'| {r.region}{mk} | {deg(r.lon, "E", "W")}, {deg(r.lat, "N", "S")} | {r.sector} | {int(r.n_cells)} | {int(r.n_days)} | '
                        f'{int(r.n_storms)} | {fm(r.ex_stage1)} | {fm(r.tmax_ex)} | {ci(r.tmax_lo, r.tmax_hi)} | {pv(r.tmax_p)} | {pv(r.q)} |')
open(f'{parts}/tabS8.md', 'w').write('\n'.join(rows) + '\n')
print(len(rows) - 2, 'rows'); print('\n'.join(rows[:8]))
