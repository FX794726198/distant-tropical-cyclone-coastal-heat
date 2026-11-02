"""Supplementary table of the warm regions of the global map (gl_map.py stage 1 and gl_stage2.py stage 2).
Usage: python 92_global_tables.py GL_DIR S2_DIR PARTS_DIR TABLE_NUMBER [COOL_TABLE_NUMBER]
Writes tabS{N}.md: one row per warm region, numbered as in Fig. 5a, and optionally the cool regions."""
import sys
import numpy as np, pandas as pd

gl, s2d, parts, num = sys.argv[1:5]
NAMES = ['Northeastern South Africa', 'Inland northern Mozambique', 'Coastal northern Mozambique', 'Madagascar', 'Southeastern India',
         'Bangladesh', 'Southern China', 'Western Australia', 'Eastern Borneo', 'Korea and northeastern China', 'Northern Australia',
         'Western Japan', 'Central Japan', 'Southern New Guinea', 'Central Queensland', 'Northeastern Queensland', 'Northwestern Mexico',
         'Eastern Mexico', 'Southeastern United States', 'Belize and Yucatán', 'Nicaragua and Honduras', 'Northeastern United States',
         'Maritime Canada']
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def months(s):
    """'xxx.......x.' style mask to runs such as 'Dec–Mar' or 'May, Oct–Dec'."""
    on = [c == 'x' for c in s]
    if all(on):
        return 'all year'
    start = next(i for i in range(12) if not on[i])          # begin scanning after a month that is off
    runs, cur = [], None
    for k in range(1, 13):
        i = (start + k) % 12
        if on[i] and cur is None:
            cur = [i, i]
        elif on[i]:
            cur[1] = i
        elif cur is not None:
            runs.append(cur); cur = None
    if cur is not None:
        runs.append(cur)
    runs.sort(key=lambda r: r[0] if r[0] <= r[1] else r[0] - 12)   # calendar order; a run across the new year comes first
    return ', '.join(MON[a] if a == b else f'{MON[a]}–{MON[b]}' for a, b in runs)


def f(v, d=2, sign=True):
    return '–' if not np.isfinite(v) else (f'{v:+.{d}f}' if sign else f'{v:.{d}f}').replace('-', '−')


RG = pd.read_csv(f'{gl}/gl_regions.csv'); W = RG[RG.sign == 'warm'].reset_index(drop=True)
assert len(W) == len(NAMES)
S = pd.read_csv(f'{s2d}/gl_s2_regions.csv'); S = S[S.sign == 'warm'].reset_index(drop=True)
assert len(S) == len(W) and (S.n_cells.values == W.n_cells.values).all()
sig = S.tmax_sig_fdr.astype(str).str.lower().isin(['true', '1', '1.0']).values & (S.tmax_lo.values > 0)   # FDR and interval above zero
def ci(lo, hi):
    # interval as in the text: 1.11–1.47, or −0.29 to 0.04 when the lower end is negative
    if not (np.isfinite(lo) and np.isfinite(hi)): return '–'
    return (f'{lo:.2f} to {hi:.2f}' if lo < 0 else f'{lo:.2f}–{hi:.2f}').replace('-', '−')


rows = ['| No. | Region (grid points) | Storm direction; months | Map warming (K) | Hot-day ratio | Days (storms) | Over analogues (K) | *T*~850~ (K) | *q*~850~ (g kg^−1^) | Rise on approach (K) |',
        '|---|---|---|---|---|---|---|---|---|---|']
for k, r in W.iterrows():
    s = S.iloc[k]
    an = f"{f(s.tmax_ex)} ({ci(s.tmax_lo, s.tmax_hi)})" + ('' if sig[k] else '^†^')
    t8 = f(s.t850_ex) + ('' if (s.t850_lo > 0 or s.t850_hi < 0) else '^‡^')
    q8 = f(s.q850_ex) + ('' if (s.q850_lo > 0 or s.q850_hi < 0) else '^‡^')
    ri = f"{f(s.get('onset_rise', np.nan))} ({ci(s.get('onset_rise_lo', np.nan), s.get('onset_rise_hi', np.nan))})" if np.isfinite(s.get('onset_rise', np.nan)) else '–'
    rows.append(f'| {k + 1} | {NAMES[k]} ({int(r.n_cells)}) | {r.sector}; {months(r.months)} | {f(r.ex_mean)} | {r.hot_ratio:.2f} | '
                f'{int(s.n_days)} ({int(s.n_storms)}) | {an} | {t8} | {q8} | {ri} |')
open(f'{parts}/tabS{num}.md', 'w').write('\n'.join(rows) + '\n')
print(f'tabS{num}.md', len(W), 'rows; over analogues significant (FDR):', int(sig.sum()))

# cool regions (optional second table)
if len(sys.argv) > 5:
    num2 = sys.argv[5]
    CNAMES = ['Central Mozambique', 'Madagascar', 'Western India', 'Eastern India', 'Myanmar and western Thailand', 'Cambodia, Laos and Thailand',
              'Northern Vietnam', 'Southern China', 'Northern Borneo', 'Northern and northwestern Australia', 'Luzon', 'Central Japan',
              'Northeastern Queensland', 'Mexico and Guatemala', 'Northeastern Mexico', 'Southeastern United States', 'Cuba',
              'Northwestern Venezuela and Colombia']
    Cg = RG[RG.sign == 'cool'].reset_index(drop=True); assert len(Cg) == len(CNAMES)
    Sc = pd.read_csv(f'{s2d}/gl_s2_regions.csv'); Sc = Sc[Sc.sign == 'cool'].reset_index(drop=True)
    assert len(Sc) == len(Cg) and (Sc.n_cells.values == Cg.n_cells.values).all()
    sgc = Sc.tmax_sig_fdr.astype(str).str.lower().isin(['true', '1', '1.0']).values & (Sc.tmax_hi.values < 0)
    mk = lambda s_, v, k=1.0, d=2: f(k * s_[v + '_ex'], d) + ('' if (s_[v + '_lo'] > 0 or s_[v + '_hi'] < 0) else '^‡^')
    rows = ['| No. | Region (grid points) | Storm direction; months | Map change (K) | Days (storms) | Over analogues (K) | *T*~850~ (K) | *q*~850~ (g kg^−1^) | Rain (mm d^−1^) | Cloud cover (%) |',
            '|---|---|---|---|---|---|---|---|---|---|']
    for k, r in Cg.iterrows():
        s_ = Sc.iloc[k]
        an = f"{f(s_.tmax_ex)} ({ci(s_.tmax_lo, s_.tmax_hi)})" + ('' if sgc[k] else '^†^')
        rows.append(f'| C{k + 1} | {CNAMES[k]} ({int(r.n_cells)}) | {r.sector}; {months(r.months)} | {f(r.ex_mean)} | {int(s_.n_days)} ({int(s_.n_storms)}) | '
                    f"{an} | {mk(s_, 't850')} | {mk(s_, 'q850')} | {mk(s_, 'tp24', 1.0, 1)} | {mk(s_, 'tcc', 100.0, 0)} |")
    open(f'{parts}/tabS{num2}.md', 'w').write('\n'.join(rows) + '\n')
    print(f'tabS{num2}.md', len(Cg), 'cool rows; significant:', int(sgc.sum()))
