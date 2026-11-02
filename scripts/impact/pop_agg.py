"""A3: sum GHS-POP R2023A (30 arcsec, WGS84) into the 1-degree boxes of the global map (box = grid point +- 0.5 degree).
Usage: python pop_agg.py   (IMP=/scratch/$USER/tcheat/impact, GL=/scratch/$USER/tcheat/global)
Output: $IMP/pop/pop_1deg.npz  pop [epoch, lon, lat] (persons), epochs, lon, lat; and a check printout of world totals."""
import os, glob
import numpy as np
import rasterio
from rasterio.windows import Window

IMP = os.environ.get('IMP', '/scratch/$USER/tcheat/impact'); GL = os.environ.get('GL', '/scratch/$USER/tcheat/global')
st = np.load(os.path.join(GL, 'static_1deg.npz')); lon, lat = st['lon'].astype(float), st['lat'].astype(float)
lon180 = ((lon + 180) % 360) - 180
li = {int(round(v)): k for k, v in enumerate(lon180)}; lj = {int(round(v)): k for k, v in enumerate(lat)}
assert np.allclose(np.diff(np.sort(lon180)), 1.0) and np.allclose(np.abs(np.diff(lat)), 1.0), 'expects a 1-degree grid'
EP = [1980, 1990, 2000, 2010, 2020]
P = np.zeros((len(EP), len(lon), len(lat)))
for e, y in enumerate(EP):
    f = glob.glob(os.path.join(IMP, 'pop', f'GHS_POP_E{y}_GLOBE_R2023A_4326_30ss*.tif'))
    assert len(f) == 1, f
    with rasterio.open(f[0]) as src:
        T = src.transform; W, H = src.width, src.height
        assert abs(T.a - 1 / 120) < 1e-9 and abs(T.e + 1 / 120) < 1e-9, T
        xc = T.c + (np.arange(W) + 0.5) * T.a                           # pixel centres
        bi = np.floor(xc + 0.5).astype(int); bi = np.where(bi == 180, -180, bi)
        col_box = np.array([li[v] for v in bi])
        tot = 0.0
        for r0 in range(0, H, 1200):
            h = min(1200, H - r0)
            a = src.read(1, window=Window(0, r0, W, h)).astype(np.float64)
            a[~np.isfinite(a) | (a < 0)] = 0.0
            yc = T.f + (r0 + np.arange(h) + 0.5) * T.e
            bj = np.floor(yc + 0.5).astype(int)
            rs = np.zeros((len(lat), W))
            for v in np.unique(bj):
                if v in lj:
                    rs[lj[v]] += a[bj == v].sum(0)
            colsum = np.zeros((len(lon), len(lat)))                     # add the pixel columns into the lon boxes
            np.add.at(colsum, col_box, rs.T)
            P[e] += colsum
            tot += a.sum()
    print(y, f'raster total {tot / 1e9:.3f} bn, gridded {P[e].sum() / 1e9:.3f} bn', flush=True)
np.savez_compressed(os.path.join(IMP, 'pop', 'pop_1deg.npz'), pop=P.astype(np.float64), epochs=np.array(EP), lon=lon, lat=lat)
print('saved', os.path.join(IMP, 'pop', 'pop_1deg.npz'))
