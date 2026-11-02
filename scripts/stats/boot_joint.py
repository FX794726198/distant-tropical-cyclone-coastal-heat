"""Sensitivity of the pooled GraphCast intervals to storms shared across regions (review 2026-11-02).
Estimand unchanged: equal-weight mean over usable regions of storm-weighted region means (gcg_eval.pooled).
Schemes: (A) storms resampled within each region independently (as in the paper);
(B) storms resampled jointly by global storm ID: one draw of storms with replacement from all storms of the group,
    each region keeps the drawn storms it contains (with multiplicity); a region with none drawn is left out of that draw;
(C) two-way: regions resampled, then storms jointly as in B.
Usage (from the repository root): python -I scripts/stats/boot_joint.py   (GGC defaults to data/ggc; writes data/stats/boot_joint.csv)"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'fig_code'))
os.environ.setdefault('GGC', 'data/ggc')
from ggc_common import CEN, regions
GGC = os.environ['GGC']; NB = 4000
O = regions('all', 'oper'); U = O[O.usable]
C = pd.read_csv(f'{GGC}/all_oper_cases.csv')
C['band'] = np.where(C.region.map(lambda r: abs(CEN[r][1])) < 30, 'low', 'mid')
C = C[C.region.isin(U.index)]
EX = U.ex_map.to_dict()

def smean(d, col):
    w = 1.0 / d.groupby('sid').sid.transform('size'); return float((d[col] * w).sum() / w.sum())

def prep(df, col):
    df = df[np.isfinite(df[col])]
    regs = sorted(df.region.unique()); sids = sorted(df.sid.unique())
    M = np.full((len(regs), len(sids)), np.nan)   # storm mean within region
    for (g, s), d in df.groupby(['region', 'sid']):
        M[regs.index(g), sids.index(s)] = d[col].mean()
    return regs, sids, M

def est(M): return float(np.nanmean(np.nanmean(M, axis=1)))

def draws(M, scheme, rng):
    R, S = M.shape; out = np.empty(NB); has = np.isfinite(M)
    for b in range(NB):
        if scheme == 'A':
            v = []
            for i in range(R):
                ss = np.where(has[i])[0]; pk = rng.choice(ss, len(ss)); v.append(M[i, pk].mean())
            out[b] = np.mean(v)
        else:
            ri = rng.integers(0, R, R) if scheme == 'C' else np.arange(R)
            cnt = np.bincount(rng.integers(0, S, S), minlength=S)
            v = []
            for i in ri:
                w = cnt * has[i]
                if w.sum() > 0: v.append(np.nansum(M[i] * w) / w.sum())
            out[b] = np.mean(v)
    return out

def group_rows(df, label, col='d_free'):
    regs, sids, M = prep(df, col); rows = []
    shared = int((np.isfinite(M).sum(0) > 1).sum())
    for sc in 'ABC':
        bs = draws(M, sc, np.random.default_rng(7))
        rows.append(dict(group=label, scheme=sc, n_regions=len(regs), n_storms=len(sids), shared_storms=shared, est=est(M),
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p_le0=float((bs <= 0).mean())))
    return rows, (regs, sids, M)

rows = []
W = C[C.kind == 'warm']; CL = C[C.kind == 'cool']
r, _ = group_rows(W, 'warm all'); rows += r
r, lowM = group_rows(W[W.band == 'low'], 'warm <30'); rows += r
r, midM = group_rows(W[W.band == 'mid'], 'warm >=30'); rows += r
r, _ = group_rows(CL, 'cool all'); rows += r
r, _ = group_rows(W, 'warm all null', 'd_null'); rows += r
# group difference low - mid, joint storm draws shared across the two groups
regsL, sL, ML = lowM; regsM, sM, MM = midM
allS = sorted(set(sL) | set(sM))
def expand(M, s):
    X = np.full((M.shape[0], len(allS)), np.nan)
    for j, x in enumerate(s): X[:, allS.index(x)] = M[:, j]
    return X
XL, XM = expand(ML, sL), expand(MM, sM)
d0 = est(XL) - est(XM)
for sc in 'ABC':
    rng = np.random.default_rng(7); out = np.empty(NB)
    for b in range(NB):
        if sc == 'A':
            vals = []
            for X in (XL, XM):
                v = []
                for i in range(X.shape[0]):
                    ss = np.where(np.isfinite(X[i]))[0]; pk = rng.choice(ss, len(ss)); v.append(X[i, pk].mean())
                vals.append(np.mean(v))
        else:
            cnt = np.bincount(rng.integers(0, len(allS), len(allS)), minlength=len(allS)); vals = []
            for X in (XL, XM):
                R = X.shape[0]; ri = rng.integers(0, R, R) if sc == 'C' else np.arange(R); v = []
                for i in ri:
                    w = cnt * np.isfinite(X[i])
                    if w.sum() > 0: v.append(np.nansum(np.nan_to_num(X[i]) * w) / w.sum())
                vals.append(np.mean(v))
        out[b] = vals[0] - vals[1]
    rows.append(dict(group='low - mid', scheme=sc, n_regions=XL.shape[0] + XM.shape[0], n_storms=len(allS), shared_storms=np.nan,
                     est=d0, lo=np.percentile(out, 2.5), hi=np.percentile(out, 97.5), p_le0=float((out <= 0).mean())))
T = pd.DataFrame(rows)
# share of the map warming with joint resampling (ratio of pooled d_free to pooled map, map fixed per region)
T.to_csv(os.environ.get('BOOTOUT', 'data/stats/boot_joint.csv'), index=False, float_format='%.4f')
pd.set_option('display.width', 200); print(T.round(3).to_string(index=False))
# storms per region overlap summary
sh = W.groupby('sid').region.nunique(); print('warm storms in >1 region:', int((sh > 1).sum()), 'of', len(sh), '; max regions per storm', sh.max())
