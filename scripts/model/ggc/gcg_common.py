"""Region handling for the global GraphCast storm-removal test (no model code here; imported by gcg_run.py).

For each case the region's 1-degree cells play the part of the Guangdong stations of the pilot: gc_build.STN and gc_build.GD
are replaced before the edit, so radius_for() (core radius = min(800 km, distance to the nearest cell - 150 km)) and the
WIDEL removal (outer circulation removed up to 50 km before the nearest cell) work unchanged.  The null edit is applied at
a storm-free sea point at the storm's distance from the region centroid, on the far side of the region.
"""
import json, os
import numpy as np, pandas as pd, xarray as xr
import gc_build as B
from tcfilter import gc_dist, destination

WIN_LAT, WIN_LON = 15.0, 20.0
STN_FIELDS = ('t2m', 'msl', 'u10', 'v10', 't850', 'q850', 'w500', 'w850', 'z500', 'tp6')
WIN_FIELDS = ('t2m', 't850', 'msl', 'w500', 'u10', 'v10')


def load_regions(path=None):
    return json.load(open(path or os.path.join(B.TCH, 'inputs', 'gcg_regions.json')))


def set_region(reg):
    """Point gc_build at this region: its cells become the 'stations', its centroid the 'GD' reference."""
    cells = np.array(reg['cells'], float)
    B.STN = pd.DataFrame({'lon': cells[:, 0] % 360, 'lat': cells[:, 1]},
                         index=[f'c{k:03d}' for k in range(len(cells))])
    B.GD = (reg['lon'] % 360, reg['lat'])
    return B.STN


def null_centre_for(lon0, lat0, cen, lsm, lat_s, lon_s, other_tc=()):
    """A storm-free sea point at the storm's distance from the region centroid, first on the far side of the region."""
    d0 = gc_dist(lon0, lat0, cen[0], cen[1])
    b0 = np.degrees(np.arctan2(np.sin(np.radians(lon0 - cen[0])) * np.cos(np.radians(lat0)),
                               np.cos(np.radians(cen[1])) * np.sin(np.radians(lat0))
                               - np.sin(np.radians(cen[1])) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon0 - cen[0]))))
    for db in (180, 150, -150, 120, -120, 90, -90, 60, -60):
        x, y = destination(cen[0], cen[1], b0 + db, d0)
        if not (5.0 <= abs(y) <= 60.0):
            continue
        j = int(np.argmin(np.abs(lat_s - y))); i = int(np.argmin(np.abs(lon_s - (x % 360))))
        if lsm[j, i] >= 0.5:
            continue
        if gc_dist(x, y, lon0, lat0) > 1200 and all(gc_dist(x, y, a, c) > 1000 for a, c in other_tc):
            return float(x), float(y)
    return None


def window_index(lon_vals, lat_vals, cen):
    """Indices and relative-longitude coordinate of a +-WIN_LAT, +-WIN_LON window around the centroid (wraps the dateline)."""
    dl = ((lon_vals - cen[0] + 180) % 360) - 180
    i = np.where(np.abs(dl) <= WIN_LON + 1e-6)[0]
    i = i[np.argsort(dl[i])]
    j = np.where(np.abs(lat_vals - cen[1]) <= WIN_LAT + 1e-6)[0]
    return i, j, dl[i], lat_vals[j]


def extract(p, cen):
    """Cell values and a regional window from one prediction time (dims lat, lon[, level])."""
    st = B.STN
    pts = dict(lat=xr.DataArray(st.lat.values, dims='station'), lon=xr.DataArray(st.lon.values, dims='station'))
    fields = {'t2m': p['2m_temperature'], 'msl': p['mean_sea_level_pressure'] / 100, 'u10': p['10m_u_component_of_wind'],
              'v10': p['10m_v_component_of_wind'], 't850': p['temperature'].sel(level=850),
              'q850': p['specific_humidity'].sel(level=850) * 1000, 'w500': p['vertical_velocity'].sel(level=500),
              'w850': p['vertical_velocity'].sel(level=850), 'z500': p['geopotential'].sel(level=500) / 9.80665}
    if 'total_precipitation_6hr' in p:
        fields['tp6'] = p['total_precipitation_6hr'] * 1000
    i, j, wlon, wlat = window_index(p.lon.values, p.lat.values, cen)
    out = {}
    for k, f in fields.items():
        out['stn_' + k] = f.interp(**pts).values.astype('float32')
        if k in WIN_FIELDS:
            out['win_' + k] = f.isel(lat=j, lon=i).values.astype('float32')
    return out, wlat, wlon
