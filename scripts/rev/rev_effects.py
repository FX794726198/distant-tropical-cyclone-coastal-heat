"""Per-case effect of the storm at the 19 stations for every variant of the given output directories (control minus
variant, 14 LT on the case day, mean over stations; per lead and averaged over leads), for t2m, t850, q850 and msl.
Usage: python rev_effects.py out_small_rev out_oper_rev out_small out_small_wide out_oper_v3  ->  $TCH/rev/effects.csv
"""
import os, sys, glob
import numpy as np, pandas as pd, xarray as xr
TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')); OUT = os.path.join(TCH, 'rev'); os.makedirs(OUT, exist_ok=True)
rows = []
for dname in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(TCH, dname, 'case_*.nc'))):
        ds = xr.open_dataset(f)
        for v in ds.variant.values:
            if v == 'CTL':
                continue
            for L in ds.lead.values:
                r = dict(dir=dname, model=ds.attrs.get('model'), case=int(ds.attrs['case']), kind=ds.attrs.get('kind'), variant=str(v), lead=int(L))
                for k in ('t2m', 't850', 'q850', 'msl'):
                    a = ds['stn_' + k].sel(variant='CTL', lead=L).values; b = ds['stn_' + k].sel(variant=v, lead=L).values
                    r['d_' + k] = float(np.nanmean(a - b))
                r['ctl_minus_truth_t2m'] = float(np.nanmean(ds['stn_t2m'].sel(variant='CTL', lead=L).values - ds['truth_stn_t2m'].values))
                rows.append(r)
        ds.close()
D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, 'effects.csv'), index=False, float_format='%.4g')
C = D.groupby(['dir', 'kind', 'variant', 'case'])[['d_t2m', 'd_t850']].mean().reset_index()
print(C.groupby(['dir', 'kind', 'variant'])[['d_t2m', 'd_t850']].agg(['mean', 'count']).round(2).to_string())
