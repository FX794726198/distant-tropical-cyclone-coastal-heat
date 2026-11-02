"""Coast-mean WRF maps for Fig. 2b: control minus storm-removed run (WIDEL), mean over each coast's cases, at the case time.

Input: wrfnc_maps.npz written by wrfnc_analyse.py (d02 fields, thinned; keys '{case}_{run}__{T2,U10,V10,lat,lon}').
Output: wrfnc_maps_coast.npz with, for each coast, '{coast}__lon', '{coast}__lat' (1-D, 0.15-degree grid) and
'{coast}__T2' (2-D, K, NaN outside d02), and '{coast}__wlon', '{coast}__wlat', '{coast}__du', '{coast}__dv' (points of a
0.75-degree grid, m/s).  The change in T2 is interpolated linearly to the 0.15-degree grid and stored to 0.1 K, the wind
change to 0.1 m/s, as in the figure; these are drawing values, the numbers in the text come from wrfnc_effects.csv.
The WRF grid is nearly regular, so the Delaunay triangulation behind the linear interpolation is not unique: another
SciPy/Qhull version, or the same points in another order, can change single interpolated values by up to about 1.5 K
near sharp coastal gradients (median change 0, 90th percentile 0.1 K), while the coast means are unchanged.  The file in
data/wrfnc/ is the one drawn in the manuscript (written by this procedure from the runs' maps; the control and
storm-removed runs used here did not change between the first analysis and the final one).
Usage: python -I wrfnc_coast_maps.py wrfnc_maps.npz wrfnc_maps_coast.npz"""
import sys
import numpy as np
from scipy.interpolate import griddata

Z = np.load(sys.argv[1])
CO = {'W17': [393, 398, 397], 'W3': [321, 319, 312], 'W5': [330, 332, 336], 'GD': [6, 16], 'W22': [419, 417, 426],
      'C16': [521, 522, 517]}
out = {}
for coast, cs in CO.items():
    la = Z[f'{cs[0]}_ctl__lat']; lo = Z[f'{cs[0]}_ctl__lon']
    for c in cs:
        for r in ('ctl', 'widel'):
            assert np.allclose(Z[f'{c}_{r}__lat'], la) and np.allclose(Z[f'{c}_{r}__lon'], lo), (c, r)
    dif = lambda v: np.mean([Z[f'{c}_ctl__{v}'] - Z[f'{c}_widel__{v}'] for c in cs], 0)
    D, U, V = dif('T2'), dif('U10'), dif('V10')
    pts = np.c_[lo.ravel(), la.ravel()]

    def grid(d):
        gx = np.arange(np.ceil(lo.min() / d) * d, lo.max() + 1e-9, d)
        gy = np.arange(np.ceil(la.min() / d) * d, la.max() + 1e-9, d)
        X, Y = np.meshgrid(gx, gy)
        return gx, gy, X, Y

    gx, gy, X, Y = grid(0.15)
    T = griddata(pts, D.ravel(), (X, Y), method='linear')
    q = np.where(np.isfinite(T), np.clip(np.round(T * 10), -127, 127), -128).astype(np.int8)
    out[f'{coast}__lon'] = round(float(gx[0]), 2) + 0.15 * np.arange(len(gx))
    out[f'{coast}__lat'] = round(float(gy[0]), 2) + 0.15 * np.arange(len(gy))
    out[f'{coast}__T2'] = np.where(q == -128, np.nan, q / 10.0)
    wx, wy, WX, WY = grid(0.75)
    Ug = griddata(pts, U.ravel(), (WX, WY), method='linear'); Vg = griddata(pts, V.ravel(), (WX, WY), method='linear')
    ok = np.isfinite(Ug.ravel())
    for k, v, nd in (('wlon', WX, 2), ('wlat', WY, 2), ('du', Ug, 1), ('dv', Vg, 1)):
        out[f'{coast}__{k}'] = np.round(v.ravel()[ok], nd)
    print(coast, 'grid', q.shape, 'mean change', round(float(np.nanmean(D)), 3), 'K; wind points', int(ok.sum()))
np.savez_compressed(sys.argv[2], **out)
