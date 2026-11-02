"""A4: a seaward unit vector for every region of the global map.
Natural Earth 1:50m land polygons -> 0.25-degree land mask -> Gaussian smoothing (sigma 1 degree) -> minus the gradient
(per km) at the region's 1-degree grid points, summed and normalised.  Onshore wind = -(u, v) . n (positive from sea to land).
Usage: python coast_normals.py LAND_SHP gcg_regions.json OUT.csv"""
import json, sys
import numpy as np, pandas as pd
import shapefile
from shapely.geometry import shape
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

shp, rj, out = sys.argv[1:4]
RES = 0.25
geoms = [shape(s.__geo_interface__) for s in shapefile.Reader(shp).shapes()]
T = from_origin(-180, 90, RES, RES); H, W = int(180 / RES), int(360 / RES)
land = features.rasterize(((g, 1) for g in geoms), out_shape=(H, W), transform=T, fill=0, dtype='uint8').astype(float)
lat = 90 - (np.arange(H) + 0.5) * RES; lon = -180 + (np.arange(W) + 0.5) * RES
sm = ndimage.gaussian_filter(land, sigma=1.0 / RES, mode=('nearest', 'wrap'))
dy_km = RES * 111.2; dx_km = RES * 111.2 * np.cos(np.radians(lat))[:, None]
gy = -np.gradient(sm, axis=0) / dy_km          # northward derivative (row index increases southward)
gx = np.gradient(sm, axis=1) / dx_km           # eastward derivative
R = json.load(open(rj)); rows = []
for r, d in R.items():
    sx = sy = 0.0; n = 0
    for lo_, la_ in d['cells']:
        lo_ = ((float(lo_) + 180) % 360) - 180
        i = int(np.clip(np.floor((90 - float(la_)) / RES), 0, H - 1)); j = int(np.floor((lo_ + 180) / RES)) % W
        sx += gx[i, j]; sy += gy[i, j]; n += 1
    nx, ny = -sx, -sy; m = np.hypot(nx, ny)
    rows.append(dict(region=r, kind=d['sign'], n_cells=n, nx=nx / m, ny=ny / m, seaward_deg=(np.degrees(np.arctan2(nx, ny)) + 360) % 360,
                     grad_per_100km=100 * m / n, lon=d['lon'], lat=d['lat']))
pd.DataFrame(rows).to_csv(out, index=False, float_format='%.4f')
print(pd.DataFrame(rows).round(2).to_string(index=False))
