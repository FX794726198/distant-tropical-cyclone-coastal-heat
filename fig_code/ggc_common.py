"""Shared inputs for Fig. 5 (figS05_graphcast_cells.py) and Fig. S7 (figS04_graphcast_regions.py): GraphCast storm removal on the warm and cool coasts of the
global map (2016-2021). Region centroids (gcg_regions.json), short names (Tables S1 and S2), map label positions (as in Fig. 1a),
pooled values from the result files of gcg_eval.py (hottest days) and gcg_eval_all.py (all usable exposure days), region tables,
region intervals from resampling storms and grid-point results by distance from the coast (ggc_cells.py)."""
import os, re
import numpy as np, pandas as pd

GGC = os.environ.get('GGC', 'data/ggc')
GD = (113.5, 23.0)
CEN = {'W1': (30.7, -24.7), 'W2': (36.7, -12.9), 'W3': (39.4, -14.9), 'W4': (46.1, -19.8), 'W5': (79.1, 11.8), 'W6': (91.2, 23.8),
       'W7': (109.0, 26.4), 'W8': (114.2, -23.5), 'W9': (117.3, 2.0), 'W10': (125.6, 40.0), 'W11': (132.7, -19.0), 'W12': (132.4, 33.9),
       'W13': (139.1, 37.1), 'W14': (142.7, -7.6), 'W15': (146.0, -24.0), 'W16': (146.0, -17.9), 'W17': (-106.9, 25.3), 'W18': (-98.1, 21.9),
       'W19': (-84.8, 36.0), 'W20': (-88.8, 17.2), 'W21': (-84.7, 13.6), 'W22': (-73.2, 44.1), 'W23': (-64.6, 45.0),
       'C1': (37.0, -17.3), 'C2': (47.3, -18.5), 'C3': (75.5, 15.9), 'C4': (83.5, 20.9), 'C5': (97.1, 20.2), 'C6': (105.7, 13.9),
       'C7': (105.1, 23.1), 'C8': (112.7, 23.2), 'C9': (115.0, 4.3), 'C10': (126.6, -19.6), 'C11': (121.2, 16.0), 'C12': (137.0, 34.7),
       'C13': (145.2, -18.1), 'C14': (-100.6, 21.9), 'C15': (-101.6, 28.8), 'C16': (-82.3, 32.0), 'C17': (-78.0, 21.2), 'C18': (-71.3, 10.0)}
NAME = {'W1': 'NE South Africa', 'W2': 'Inland N Mozambique', 'W3': 'Coastal N Mozambique', 'W4': 'Madagascar', 'W5': 'SE India',
        'W6': 'Bangladesh', 'W7': 'Southern China', 'W8': 'W Australia', 'W9': 'E Borneo', 'W10': 'Korea and NE China', 'W11': 'N Australia',
        'W12': 'W Japan', 'W13': 'Central Japan', 'W14': 'S New Guinea', 'W15': 'Central Queensland', 'W16': 'NE Queensland',
        'W17': 'NW Mexico', 'W18': 'E Mexico', 'W19': 'SE United States', 'W20': 'Belize and Yucatan', 'W21': 'Nicaragua and Honduras',
        'W22': 'NE United States', 'W23': 'Maritime Canada',
        'C1': 'Central Mozambique', 'C2': 'Madagascar', 'C3': 'Western India', 'C4': 'Eastern India', 'C5': 'Myanmar and W Thailand',
        'C6': 'Cambodia, Laos, Thailand', 'C7': 'Northern Vietnam', 'C8': 'Southern China', 'C9': 'Northern Borneo', 'C10': 'N and NW Australia',
        'C11': 'Luzon', 'C12': 'Central Japan', 'C13': 'NE Queensland', 'C14': 'Mexico and Guatemala', 'C15': 'NE Mexico',
        'C16': 'SE United States', 'C17': 'Cuba', 'C18': 'NW Venezuela and Colombia'}
LPOS = {'W1': (35, -31), 'W2': (31, -9), 'W3': (44.5, -10), 'W4': (55, -22), 'W5': (73, 6), 'W6': (91, 16.5), 'W7': (101, 39), 'W8': (107, -25),
        'W9': (122, 3.5), 'W10': (117, 45), 'W11': (133, -31), 'W12': (131, 27.5), 'W13': (147, 40), 'W14': (143, -0.5), 'W15': (156, -27),
        'W16': (153.5, -15.5), 'W17': (-117, 17), 'W18': (-91, 26.5), 'W19': (-99, 41), 'W20': (-82.5, 21.5), 'W21': (-79, 12),
        'W22': (-85, 48.5), 'W23': (-57, 41),
        'C1': (26, -19.5), 'C4': (80, 27), 'C5': (97, 12), 'C7': (103, 31), 'C8': (113, 16.5), 'C9': (111, 9), 'C12': (141, 30),
        'C15': (-109, 33.5), 'C16': (-73, 30), 'C17': (-74, 15.5), 'C18': (-66, 13.5)}
NUM = lambda r: r[1:] if r[0] == 'W' else r      # warm regions carry their Fig. 1 numbers, cool regions C1-C18 (Table S2)


def pooled_from(fname):
    """pooled (mean, lo, hi) per model from a result file of gcg_eval.py (hot) or gcg_eval_all.py (all)"""
    txt = open(f'{GGC}/{fname}').read(); out = {}
    num = r'([+-]?\d+\.\d+)'
    for blk in re.split(r'================ model ', txt)[1:]:
        m = blk.split(':')[0].strip(); d = {}
        for key in ('free', 'full', 'core', 'null'):
            g = re.search(rf'pooled warm d_{key}(?: \(CTL - \w+\))?:? {num}(?: K)? \[{num}, {num}\]', blk) or \
                re.search(rf'd_{key} {num} \[{num}, {num}\]', blk)
            d[key] = tuple(float(x) for x in g.groups())
        g = re.search(rf'pooled (?:cool d_free:|d_free) {num} K \[{num}, {num}\]', blk) or re.search(rf'cool mirror: pooled d_free {num} \[{num}, {num}\]', blk)
        d['cool'] = tuple(float(x) for x in g.groups())
        g = re.search(rf'cases not run in A: .*?pooled d_free {num} \[{num}, {num}\], map warming {num}', blk)
        if g: d['new'] = tuple(float(x) for x in g.groups()[:3]); d['new_map'] = float(g.group(4))
        g = re.search(rf'pooled map warming {num} K', blk)
        if g: d['map'] = float(g.group(1))
        g = re.search(rf'\(map {num}\)', blk)
        if g: d['cool_map'] = float(g.group(1))
        out[m] = d
    return out


def regions(prefix, model):
    d = pd.read_csv(f'{GGC}/{prefix}_{model}_regions.csv').set_index('region')
    for c in ('usable', 'consistent', 'confirmed'):
        if c in d: d[c] = d[c].astype(str).str.lower().eq('true')
    return d


def region_ci(rng, cases, r, col='d_free', nboot=2000):
    sm = cases[cases.region == r].groupby('sid')[col].mean().values
    bs = [sm[rng.integers(0, len(sm), len(sm))].mean() for _ in range(nboot)]
    return np.percentile(bs, [2.5, 97.5])


def all_ci(rng, cases, usable):
    """intervals in a fixed region order (warm W1-W23, then cool C1-C18), as in Table S10; the first use of the generator"""
    return {r: region_ci(rng, cases, r) for r in [f'W{i}' for i in range(1, 24)] + [f'C{i}' for i in range(1, 19)] if r in usable}


EDGES = [0, 50, 150, 300, 1e9]; ELAB = ['0–50', '50–150', '150–300', '> 300']


def cell_bins(rng, model, kind, T, col='d_free', nboot=1000):
    """TC-weighted effect by distance of the grid point from the coast, with an interval from resampling region-TC pairs"""
    D = pd.read_csv(f'{GGC}/all_cells_{model}.csv')
    D = D[(D.kind == kind) & D.region.isin(T[T.usable & (T.kind == kind)].index)].copy()
    D['w'] = 1.0 / D.groupby(['region', 'sid']).sid.transform('size')
    D['b'] = pd.cut(D.d_coast, EDGES, right=False, labels=False)
    out = []
    for b in range(4):
        d = D[D.b == b]
        if len(d) == 0:
            out.append((np.nan, np.nan, np.nan, 0)); continue
        est = float((d[col] * d.w).sum() / d.w.sum())
        keys = d.groupby(['region', 'sid']).ngroup().values; nk = keys.max() + 1
        g = {k: d[col].values[keys == k] for k in range(nk)}; bs = []
        for _ in range(nboot):
            pick = rng.integers(0, nk, nk)
            bs.append(np.mean([g[k].mean() for k in pick]))
        out.append((est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(d.cell.size)))
    return out
