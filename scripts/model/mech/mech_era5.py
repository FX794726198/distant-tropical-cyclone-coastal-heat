"""A4 ERA5 fingerprint of the 23 warm regions (plan plan A4 (internal); exploratory).
From the second step of the global map (gl_s2_regions.csv: excess over circulation analogues) and the seaward unit
vectors (coast_normals.py): onshore 10-m wind excess = -(u10_ex nx + v10_ex ny); class dry (q850 < 0 and rain < 0),
moist (q850 > 0) or other.
Usage: python mech_era5.py gl_s2_regions.csv coast_normals.csv OUT.csv"""
import sys
import numpy as np, pandas as pd

s2, nrm, out = sys.argv[1:4]
S = pd.read_csv(s2); N = pd.read_csv(nrm).set_index('region')
S = S[S.region.isin(N.index)].copy()
S['nx'] = S.region.map(N.nx); S['ny'] = S.region.map(N.ny); S['seaward_deg'] = S.region.map(N.seaward_deg)
S['onshore_ex'] = -(S.u10_ex * S.nx + S.v10_ex * S.ny)
S['class'] = np.where((S.q850_ex < 0) & (S.tp24_ex < 0), 'dry', np.where(S.q850_ex > 0, 'moist', 'other'))
cols = ['region', 'sign', 'sector', 'seaward_deg', 'tmax_ex', 't850_ex', 'q850_ex', 'tmax_t850_ex', 'tcc_ex', 'tp24_ex', 'onshore_ex', 'class']
S[cols].to_csv(out, index=False, float_format='%.4g')
W = S[S.sign == 'warm']
print(S[cols].round(2).to_string(index=False))
print(f'\nwarm regions: {len(W)}; dry {int((W["class"] == "dry").sum())}, moist {int((W["class"] == "moist").sum())}, '
      f'other {int((W["class"] == "other").sum())}; onshore excess < 0 in {int((W.onshore_ex < 0).sum())}; '
      f'T850 > 0 in {int((W.t850_ex > 0).sum())}; Tmax - T850 > 0 in {int((W.tmax_t850_ex > 0).sum())}')
