"""Summarise the GraphCast pilot: typhoon effect on Guangdong 14 LT temperature and pass/fail lines.

Usage: python gc_analyse.py --model small   -> $TCH/out_small/summary_NOTC.txt, per_case.csv, composite.nc
       python gc_analyse.py --model small --tag _wide --main WIDE   (WIDE / WIDEH / NULLW variants)
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, xarray as xr

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
COAST = ['59493099999', '59501099999', '59663099999', '59673099999', '59658099999']


def boot_ci(x, groups, nb=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    bs = [np.mean(x[np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])]) for _ in range(nb)]
    return np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='small')
    ap.add_argument('--tag', default=''); ap.add_argument('--main', default='NOTC', choices=['NOTC', 'WIDE'])
    a = ap.parse_args()
    odir = os.path.join(TCH, 'out_' + a.model + a.tag)
    M = a.main; WK = {'NOTC': 'WEAK', 'WIDE': 'WIDEH'}[M]; NL = {'NOTC': 'NULL', 'WIDE': 'NULLW'}[M]
    OTHERS = ('NOTC', 'WEAK', 'SHIFT', 'NULL', 'WIDE', 'WIDEH', 'NULLW')
    rows, comp = [], {}
    regen = []
    for f in sorted(glob.glob(os.path.join(odir, 'case_*.nc'))):
        ds = xr.open_dataset(f); meta = json.load(open(f.replace('.nc', '.json')))
        em = ds.mean('lead')  # lagged-ensemble mean
        r = dict(case=ds.attrs['case'], V=ds.attrs['V'], kind=ds.attrs['kind'], tid=ds.attrs['tid'], nlead=ds.sizes['lead'])
        for v in ds.variant.values:
            r[f'T_{v}'] = float(em.stn_t2m.sel(variant=v).mean())
            r[f'T850_{v}'] = float(em.stn_t850.sel(variant=v).mean())
            r[f'q850_{v}'] = float(em.stn_q850.sel(variant=v).mean())
            r[f'v10coast_{v}'] = float(em.stn_v10.sel(variant=v, station=COAST).mean())
            r[f'onshore_{v}'] = float((ds.stn_v10.sel(variant=v, station=COAST) > 0).mean())
        r['T_truth'] = float(ds.truth_stn_t2m.mean())
        # spread of CTL-NOTC across leads (member spread)
        if M in ds.variant.values:
            d = (ds.stn_t2m.sel(variant=M) - ds.stn_t2m.sel(variant='CTL')).mean('station')
            r['dMAIN_lead_sd'] = float(d.std()) if ds.sizes['lead'] > 1 else np.nan
        for L, m in meta['leads'].items():
            if isinstance(m, dict) and f'{M}_deficit_at_V' in m:
                regen.append(dict(case=r['case'], kind=r['kind'], lead=L, ctl=m.get('CTL_deficit_at_V'), notc=m[f'{M}_deficit_at_V']))
        rows.append(r)
        for v in ds.variant.values:
            for k in ('win_t2m', 'win_msl', 'win_t850', 'win_w500', 'win_q850', 'win_tp6'):
                if k in ds:
                    comp.setdefault((r['kind'], v, k), []).append(em[k].sel(variant=v).values)
        wl, wo = ds.wlat.values, ds.wlon.values
    P = pd.DataFrame(rows)
    for v in OTHERS:
        if f'T_{v}' in P:
            P[f'd{v}'] = P[f'T_{v}'] - P['T_CTL']
    if 'T_NOTC' in P and 'T_SHIFT' in P:
        P['SW_effect'] = P['T_SHIFT'] - P['T_NOTC']   # adding the storm in the SW sector
    P['ctl_err'] = P['T_CTL'] - P['T_truth']
    P.to_csv(os.path.join(odir, 'per_case.csv'), index=False)
    out = []
    for kind in ('heat', 'placebo'):
        Q = P[P.kind == kind]
        if Q.empty: continue
        g = Q.tid.values
        out.append(f'=== {kind}: {len(Q)} cases, {Q.tid.nunique()} storms, {Q.nlead.mean():.1f} leads per case')
        out.append(f'CTL minus ERA5 (station-mean 14 LT T2m): mean {Q.ctl_err.mean():+.2f} K, |err|<=1 K in {(Q.ctl_err.abs() <= 1).mean():.0%}')
        for v in OTHERS:
            k = f'd{v}'
            if k in Q:
                lo, hi = boot_ci(Q[k].values, g)
                out.append(f'{v} - CTL: {Q[k].mean():+.2f} K [{lo:+.2f}, {hi:+.2f}] (storm bootstrap), negative in {(Q[k] < 0).mean():.0%} of cases')
        if 'SW_effect' in Q:
            lo, hi = boot_ci(Q.SW_effect.values, g)
            out.append(f'SHIFT - NOTC (storm placed SW): {Q.SW_effect.mean():+.2f} K [{lo:+.2f}, {hi:+.2f}]')
        for fp in ('T850', 'q850', 'v10coast', 'onshore'):
            if f'{fp}_{M}' in Q:
                dd = Q[f'{fp}_CTL'] - Q[f'{fp}_{M}']
                lo, hi = boot_ci(dd.values, g)
                out.append(f'fingerprint CTL - {M} {fp}: {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]')
        if 'dMAIN_lead_sd' in Q:
            out.append(f'lead-to-lead SD of {M}-CTL: median {Q.dMAIN_lead_sd.median():.2f} K')
    R = pd.DataFrame(regen)
    if len(R):
        rat = (R.notc / R.ctl.where(R.ctl > 1)).median()
        out.append(f'=== regeneration: {M} runs with deficit > 3 hPa at V near the ERA5 storm centre: {(R.notc > 3).mean():.0%} of {len(R)} runs; CTL deficit median {R.ctl.median():.1f} hPa, {M} median {R.notc.median():.1f} hPa, median ratio {M}/CTL {rat:.2f}')
    H = P[P.kind == 'heat']
    if len(H) and f'd{M}' in H:
        dM = H[f'd{M}']
        lo, hi = boot_ci(dM.values, H.tid.values)
    txt = '\n'.join(out); print(txt)
    open(os.path.join(odir, f'summary_{M}.txt'), 'w').write(txt + '\n')
    if comp:
        dv = {f'{kind}_{v}_{k}': (('wlat', 'wlon'), np.nanmean(np.stack(x), 0)) for (kind, v, k), x in comp.items()}
        xr.Dataset(dv, coords={'wlat': wl, 'wlon': wo}).to_netcdf(os.path.join(odir, 'composite.nc'))


if __name__ == '__main__':
    main()
