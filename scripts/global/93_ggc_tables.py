"""Tables S9 and S10: GraphCast storm removal on the coasts of the global map (hottest-day set and all-day set).
v08: Table S9 is a descriptive summary (no criteria or pass/fail marks) and adds the two latitude bands of the all-day set.
Reads data/ggc (hot_*/all_* region and case tables and result files written by gcg_eval.py and gcg_eval_all.py) and writes
TABOUT/tabS9.md and TABOUT/tabS10.md (default tables_out/). Region intervals: storm bootstrap within the region (2,000 samples, seed 7), as in Fig. 5d."""
import os, re
import numpy as np, pandas as pd
D = os.environ.get('PAPER', '.'); G = f'{D}/data/ggc'
TO = os.environ.get('TABOUT', 'tables_out'); os.makedirs(TO, exist_ok=True)
NAME = {'W1': 'NE South Africa', 'W2': 'Inland N Mozambique', 'W3': 'Coastal N Mozambique', 'W4': 'Madagascar', 'W5': 'SE India',
        'W6': 'Bangladesh', 'W7': 'Southern China', 'W8': 'W Australia', 'W9': 'E Borneo', 'W10': 'Korea and NE China', 'W11': 'N Australia',
        'W12': 'W Japan', 'W13': 'Central Japan', 'W14': 'S New Guinea', 'W15': 'Central Queensland', 'W16': 'NE Queensland',
        'W17': 'NW Mexico', 'W18': 'E Mexico', 'W19': 'SE United States', 'W20': 'Belize and Yucatan', 'W21': 'Nicaragua and Honduras',
        'W22': 'NE United States', 'W23': 'Maritime Canada',
        'C1': 'Central Mozambique', 'C2': 'Madagascar', 'C3': 'Western India', 'C4': 'Eastern India', 'C5': 'Myanmar and W Thailand',
        'C6': 'Cambodia, Laos, Thailand', 'C7': 'Northern Vietnam', 'C8': 'Southern China', 'C9': 'Northern Borneo', 'C10': 'N and NW Australia',
        'C11': 'Luzon', 'C12': 'Central Japan', 'C13': 'NE Queensland', 'C14': 'Mexico and Guatemala', 'C15': 'NE Mexico',
        'C16': 'SE United States', 'C17': 'Cuba', 'C18': 'NW Venezuela and Colombia'}
num = r'([+-]?\d+\.\d+)'
def block(fname, model):
    t = open(f'{G}/{fname}').read()
    return [b for b in re.split(r'================ ', t) if b.startswith(f'model {model}')][0]
def grab(b, pat, n=3):
    m = re.search(pat, b); assert m, pat; return [float(x) for x in m.groups()[:n]]
f2 = lambda v: '0.00' if abs(v) < 0.005 else f'{v:+.2f}'.replace('-', '−')
ci = lambda m, lo, hi: f'{f2(m)} ({f2(lo)} to {f2(hi)})'
reg = lambda pre, mdl: pd.read_csv(f'{G}/{pre}_{mdl}_regions.csv').set_index('region')
T = lambda s: str(s).lower() == 'true'

# ---------------------------------------------------------------- Table S9: summary of the two sets (descriptive since v08)
import json
RJ = json.load(open(f'{G}/gcg_regions.json'))
H = {m: block('hot_result.txt', m) for m in ('oper', 'small')}; A = {m: block('all_result.txt', m) for m in ('oper', 'small')}
HR = {m: reg('hot', m) for m in ('oper', 'small')}; AR = {m: reg('all', m) for m in ('oper', 'small')}
pw = {m: grab(H[m], rf'pooled warm d_free \(CTL - WIDEL\): {num} K \[{num}, {num}\]') for m in H}
pn = {m: grab(H[m], rf'pooled warm d_null \(CTL - NULLW\): {num} K \[{num}, {num}\]') for m in H}
rg = {m: float(re.search(r'WIDEL (\d+)% of', H[m]).group(1)) for m in H}
a = {}
for m in A:
    b = A[m]
    a[m] = dict(free=grab(b, rf'pooled warm d_free {num} K \[{num}, {num}\]'), share=grab(b, rf'= (\d+)% of the map warming \[(\d+)%, (\d+)%\]'),
                mapw=grab(b, rf'pooled map warming {num} K', 1)[0], null=grab(b, rf'd_null {num} \[{num}, {num}\]'),
                cons=grab(b, r'warm regions with d_free > 0 and > d_null: (\d+) of (\d+)', 2),
                usable=grab(b, r'usable warm regions (\d+);', 1)[0],
                new=grab(b, rf'cases not run in A: (\d+) in (\d+) warm regions with >= 2: pooled d_free {num} \[{num}, {num}\], map warming {num}', 6),
                regrow=grab(b, r'regrowth \(deficit > 3 hPa at V\), WIDEL: (\d+)% of (\d+) runs', 2))


def band(m, lo_lat, hi_lat):
    """Pooled region-free effect of the usable warm regions of the all-day set in a latitude band (|lat| in [lo, hi)):
    storms weighted equally within a region, regions equally; 2,000 storm resamples within regions (seed 7)."""
    C = pd.read_csv(f'{G}/all_{m}_cases.csv'); R = AR[m]
    regs = [r for r in R.index if R.loc[r, 'kind'] == 'warm' and T(R.loc[r, 'usable']) and lo_lat <= abs(RJ[r]['lat']) < hi_lat]
    sm = {r: C[C.region == r].groupby('sid').d_free.mean().values for r in regs}
    est = np.mean([sm[r].mean() for r in regs]); mp = np.mean([R.loc[r, 'ex_map'] for r in regs])
    rng = np.random.default_rng(7)
    bs = [np.mean([sm[r][rng.integers(0, len(sm[r]), len(sm[r]))].mean() for r in regs]) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    npos = sum(1 for r in regs if R.loc[r, 'd_free'] > 0 and R.loc[r, 'd_free'] > R.loc[r, 'd_null'])
    return len(regs), npos, (est, lo, hi), mp


def cons_hot(m):
    w = HR[m][(HR[m].kind == 'warm') & HR[m].usable.map(T)]
    return int(((w.d_free > 0) & (w.d_free > w.d_null)).sum()), len(w)


o, s_ = a['oper'], a['small']
pct = lambda e, mp: f'{100 * e / mp:.0f}%'
rows = []
rows.append(('Usable warm regions (at least three days)', f'{cons_hot("oper")[1]}', f'{cons_hot("small")[1]}', f'{o["usable"]:.0f}', f'{s_["usable"]:.0f}'))
rows.append(('Pooled effect, warm regions (K)', ci(*pw['oper']), ci(*pw['small']), ci(*o['free']), ci(*s_['free'])))
rows.append((f'Share of the map warming of these regions ({f2(o["mapw"])} K)', '–', '–', f'{o["share"][0]:.0f}% ({o["share"][1]:.0f}–{o["share"][2]:.0f}%)',
             f'{s_["share"][0]:.0f}% ({s_["share"][1]:.0f}–{s_["share"][2]:.0f}%)'))
rows.append(('Warm regions with a positive effect larger than the null edit', '{} of {}'.format(*cons_hot('oper')), '{} of {}'.format(*cons_hot('small')),
             f'{o["cons"][0]:.0f} of {o["cons"][1]:.0f}', f'{s_["cons"][0]:.0f} of {s_["cons"][1]:.0f}'))
rows.append(('Pooled null edit (K)', f2(pn['oper'][0]), f2(pn['small'][0]), f2(o['null'][0]), f2(s_['null'][0])))
rows.append(('Removed storm grows back (region-free removal, share of runs)', f'{rg["oper"]:.0f}%', f'{rg["small"]:.0f}%', f'{o["regrow"][0]:.0f}%', f'{s_["regrow"][0]:.0f}%'))
nw, nws = o['new'], s_['new']
rows.append((f'Days not in the hottest-day set ({nw[0]:.0f} days, {nw[1]:.0f} regions with at least two): effect (K); share of their map warming ({f2(nw[5])} K)',
             '–', '–', f'{ci(*nw[2:5])}; {100 * nw[2] / nw[5]:.0f}%', f'{ci(*nws[2:5])}; {100 * nws[2] / nws[5]:.0f}%'))
for nm, lo_, hi_ in (('below 30°', 0, 30), ('30° and above', 30, 91)):
    bo, bs_ = band('oper', lo_, hi_), band('small', lo_, hi_)
    rows.append((f'Warm regions at latitudes {nm} ({bo[0]} regions; map warming {f2(bo[3])} K): effect (K); share; regions with a positive effect larger than the null edit',
                 '–', '–', f'{ci(*bo[2])}; {pct(bo[2][0], bo[3])}; {bo[1]} of {bo[0]}', f'{ci(*bs_[2])}; {pct(bs_[2][0], bs_[3])}; {bs_[1]} of {bs_[0]}'))
L = ['| Quantity | Hottest days, 0.25° | Hottest days, 1° | All exposure days, 0.25° | All exposure days, 1° |', '|---|---|---|---|---|']
for r in rows:
    L.append('| ' + ' | '.join(r) + ' |')
open(f'{TO}/tabS9.md', 'w').write('\n'.join(L) + '\n')

# ---------------------------------------------------------------- Table S10: region results
O, S, HO = reg('all', 'oper'), reg('all', 'small'), reg('hot', 'oper'); CO = pd.read_csv(f'{G}/all_oper_cases.csv')
rng = np.random.default_rng(7)
def rci(r):
    sm = CO[CO.region == r].groupby('sid').d_free.mean().values
    bs = [sm[rng.integers(0, len(sm), len(sm))].mean() for _ in range(2000)]
    return np.percentile(bs, [2.5, 97.5])
L = ['| No. | Region | Days (storms; new) | Map change (K) | Case-day anomaly (K) | Effect, 0.25° (K) | Effect, 1° (K) | Null edit (K) | Hottest or coolest days, 0.25° (K) | Sign of the map |',
     '|---|---|---|---|---|---|---|---|---|---|']
order = [f'W{i}' for i in range(1, 24)] + [f'C{i}' for i in range(1, 19)]
few = []
for grp_, kind in (('Warm regions', 'warm'), ('Cool regions', 'cool')):
    L.append(f'| {grp_} | | | | | | | | | |')
    for r in order:
        if r[0] != kind[0].upper(): continue
        if r not in O.index or not T(O.loc[r, 'usable']):
            few.append(f'{r} ({int(O.loc[r, "n_cases"]) if r in O.index else 0})'); continue
        o, s_ = O.loc[r], S.loc[r]; lo, hi = rci(r)
        hv = f2(HO.loc[r, 'd_free']) if r in HO.index and T(HO.loc[r, 'usable']) else '–'
        L.append(f'| {r[1:] if kind == "warm" else r} | {NAME[r]} | {int(o.n_cases)} ({int(o.n_storms)}; {int(o.n_new)}) | {f2(o.ex_map)} | {f2(o.anom)} | '
                 f'{ci(o.d_free, lo, hi)} | {f2(s_.d_free)} | {f2(o.d_null)} | {hv} | {"yes" if T(o.consistent) else "no"} |')
open(f'{TO}/tabS10.md', 'w').write('\n'.join(L) + '\n')
print('fewer than three days:', ', '.join(few))
print(open(f'{TO}/tabS9.md').read()); print(open(f'{TO}/tabS10.md').read())
