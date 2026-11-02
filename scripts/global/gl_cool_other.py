"""For each warm region of gl_map.py (same clustering as regions()), the share of its cells that are significantly cooler
(FDR) when the storm lies in the coolest sector.  Usage: python gl_cool_other.py [$TCH/global/out]  -> prints a table."""
import os, sys
import numpy as np, pandas as pd
from scipy import ndimage

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'global', 'out')
df = pd.read_csv(os.path.join(D, 'gl_cells.csv'))
df = df[df.valid.astype(str).str.lower().isin(['true', '1'])]
for c in ('sig_best', 'sig_worst'):
    df[c] = df[c].astype(str).str.lower().isin(['true', '1'])
nlon, nlat = 360, 181
G = np.zeros((nlon, nlat), bool)
sel = df[(df.ex_best >= 0.5) & df.sig_best]
G[sel.i.values, sel.j.values] = True
lab, n = ndimage.label(np.concatenate([G, G], 0), structure=np.ones((3, 3)))
idx = df.set_index(['i', 'j'])
seen, out = set(), []
for r in range(1, n + 1):
    ii, jj = np.where(lab == r)
    if ii.min() >= nlon:
        continue
    key = tuple(sorted(set(zip(ii % nlon, jj))))
    if len(key) < 3 or key in seen:
        continue
    seen.add(key)
    c = idx.loc[list(key)]
    cool = c.sig_worst & (c.ex_worst < 0)
    out.append((len(out) + 1, len(key), round(float(c.lon.mean()), 1), round(float(c.lat.mean()), 1),
                int(cool.sum()), round(float(cool.mean()), 3), int((c.sig_worst & (c.ex_worst <= -0.5)).sum())))
T = pd.DataFrame(out, columns=['k', 'n_cells', 'lon', 'lat', 'n_cool_sig', 'frac_cool_sig', 'n_cool_sig_le05'])
print(T.to_string(index=False))
for th in (0.25, 1 / 3, 0.5):
    print(f'regions with frac_cool_sig >= {th:.2f}: {int((T.frac_cool_sig >= th - 1e-12).sum())} of {len(T)}')
