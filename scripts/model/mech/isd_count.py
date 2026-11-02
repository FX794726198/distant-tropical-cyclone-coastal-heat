"""A4: how many NOAA ISD stations lie in each region of the global map (1-degree region cells, centre +- 0.5 degree).
Counts stations with any data in 2016-2021 (the GraphCast case years) and stations with a long record (begin <= 1985,
end >= 2021).  Usage: python isd_count.py isd-history.csv gcg_regions.json OUT.csv"""
import json, sys
import numpy as np, pandas as pd

isd, rj, out = sys.argv[1:4]
S = pd.read_csv(isd, dtype=str)
S['lat'] = pd.to_numeric(S['LAT'], errors='coerce'); S['lon'] = pd.to_numeric(S['LON'], errors='coerce')
S['b'] = pd.to_numeric(S['BEGIN'], errors='coerce'); S['e'] = pd.to_numeric(S['END'], errors='coerce')
S = S[S.lat.notna() & S.lon.notna() & ~((S.lat == 0) & (S.lon == 0))]
S['ci'] = np.floor(S.lat + 0.5).astype(int); S['cj'] = np.floor(((S.lon + 180) % 360 - 180) + 0.5).astype(int)
recent = S[(S.e >= 20160101) & (S.b <= 20211231)]
longr = S[(S.b <= 19851231) & (S.e >= 20210101)]
key = lambda D: D.groupby(['ci', 'cj']).size()
kr, kl = key(recent), key(longr)
R = json.load(open(rj)); rows = []
for r, v in R.items():
    cells = {(int(round(la)), int(round(((lo + 180) % 360) - 180))) for lo, la in v['cells']}
    nr = sum(int(kr.get(c, 0)) for c in cells); nl = sum(int(kl.get(c, 0)) for c in cells)
    rows.append(dict(region=r, sign=v['sign'], sector=v['sector'], lon=v['lon'], lat=v['lat'], n_cells=len(cells),
                     isd_2016_2021=nr, isd_long_1985_2021=nl, per_cell_recent=nr / len(cells)))
T = pd.DataFrame(rows); T.to_csv(out, index=False, float_format='%.3g')
print(T.to_string(index=False))
