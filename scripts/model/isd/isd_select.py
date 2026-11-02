"""A5 station check, step 1: choose ISD stations.

  Set R (regions): stations inside the 1-degree cells (centre +- 0.5 deg) of the 41 regions of the global map.
  Set C (all coasts): stations within 50 km of the coastline (Natural Earth 1:50m), |lat| <= 45, not more than 10 km out
                      at sea, with >= 100 exposure days (nearest tropical storm 300-1300 km away) in 1980-2021.
Both need >= 15 years in 1980-2021 with an ISD-Lite file of >= 15 kB (about 1,700 reports).  Coast normal at every station: same method as
coast_normals.py (0.25-deg land mask, Gaussian sigma 1 deg), so it describes the coast at the scale of the outer winds.

Usage: python isd_select.py META_DIR LAND_SHP gcg_regions.json IBTRACS.csv OUT_DIR
  META_DIR: isd-history.csv and isd_lite_listing.csv(.gz) from isd_list.py (both in data/inputs/isd)
"""
import json, os, sys
import numpy as np, pandas as pd
import shapefile
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely import STRtree
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

META, SHP, RJ, IBT, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
Y0, Y1 = 1980, 2021

h = pd.read_csv(os.path.join(META, 'isd-history.csv'), dtype={'USAF': str, 'WBAN': str})
h = h.rename(columns={'LAT': 'lat', 'LON': 'lon', 'STATION NAME': 'name', 'ELEV(M)': 'elev'})
h = h.dropna(subset=['lat', 'lon']); h = h[~((h.lat == 0) & (h.lon == 0))]
h['sid'] = h.USAF + '-' + h.WBAN
# coverage from the file listing of the AWS mirror (isd_list.py); the mirror's isd-inventory.csv is incomplete before
# 2020.  ISD-Lite compresses to about 8.75 bytes per report, so 15 kB ~ 1,700 reports (~4.7 a day).
lf = os.path.join(META, 'isd_lite_listing.csv'); lf = lf if os.path.exists(lf) else lf + '.gz'
inv = pd.read_csv(lf, dtype={'sid': str})
inv['sid'] = inv.sid.str.slice(0, 6) + '-' + inv.sid.str.slice(7, 12)
inv = inv.rename(columns={'year': 'YEAR'}); inv['n'] = inv.bytes / 8.75
inv = inv[(inv.YEAR >= Y0) & (inv.YEAR <= Y1)]
good = inv[inv.bytes >= 15000].groupby('sid').YEAR.nunique()
h = h[h.sid.isin(good[good >= 15].index)].copy()
h['n_years'] = h.sid.map(good)
print('stations with >= 15 good years:', len(h))

# coast geometry
geoms = [shape(s.__geo_interface__) for s in shapefile.Reader(SHP).shapes()]
tree = STRtree(geoms)
land_u = None
RES = 0.25
T = from_origin(-180, 90, RES, RES); H, W = int(180 / RES), int(360 / RES)
land = features.rasterize(((g, 1) for g in geoms), out_shape=(H, W), transform=T, fill=0, dtype='uint8').astype(float)
glat = 90 - (np.arange(H) + 0.5) * RES
sm = ndimage.gaussian_filter(land, sigma=1.0 / RES, mode=('nearest', 'wrap'))
gy = -np.gradient(sm, axis=0) / (RES * 111.2); gx = np.gradient(sm, axis=1) / (RES * 111.2 * np.cos(np.radians(glat))[:, None])

h = h[h.lat.abs() <= 46].copy()
dist, onland, nx, ny, grad = [], [], [], [], []
bnd = [g.boundary for g in geoms]; btree = STRtree(bnd)
for lo, la in zip(h.lon.values, h.lat.values):
    p = Point(lo, la)
    k = btree.nearest(p)
    d = bnd[k].distance(p) * 111.2 * np.sqrt((np.cos(np.radians(la)) ** 2 + 1) / 2)   # rough km
    hit = tree.query(p, predicate='within')
    dist.append(d); onland.append(len(hit) > 0)
    i = int(np.clip(np.floor((90 - la) / RES), 0, H - 1)); j = int(np.floor((((lo + 180) % 360)) / RES)) % W
    vx, vy = -gx[i, j], -gy[i, j]; m = np.hypot(vx, vy)
    nx.append(vx / m if m > 0 else np.nan); ny.append(vy / m if m > 0 else np.nan); grad.append(100 * m)
h['coast_km'] = dist; h['on_land'] = onland; h['nx'] = nx; h['ny'] = ny; h['grad_per_100km'] = grad

# regions
R = json.load(open(RJ))
h['region'] = ''
for r, d in R.items():
    for lo_, la_ in d['cells']:
        lo_ = ((float(lo_) + 180) % 360) - 180; la_ = float(la_)
        m = (h.lat.between(la_ - 0.5, la_ + 0.5)) & ((((h.lon - lo_ + 180) % 360) - 180).abs() <= 0.5) & (h.region == '')
        h.loc[m, 'region'] = r

# exposure days per station (nearest TS fix on the local solar day)
tr = pd.read_csv(IBT, skiprows=[1], low_memory=False, usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'NATURE'])
for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND'):
    tr[c] = pd.to_numeric(tr[c], errors='coerce')
tr['time'] = pd.to_datetime(tr.ISO_TIME); tr['wind'] = tr.USA_WIND.fillna(tr.WMO_WIND)
tr = tr[tr.time.dt.hour.isin([0, 6, 12, 18]) & (tr.time.dt.minute == 0) & (tr.NATURE.str.strip() == 'TS') & (tr.wind >= 34)
        & tr.time.dt.year.between(Y0, Y1)].dropna(subset=['LAT', 'LON'])
tr['day'] = (tr.time + pd.to_timedelta(tr.LON / 15.0, unit='h')).dt.normalize()


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


slon, slat = h.lon.values, h.lat.values
nexp = np.zeros(len(h), int)
for day, g in tr.groupby('day'):
    d = hav(slon[:, None], slat[:, None], g.LON.values[None, :], g.LAT.values[None, :]).min(1)
    nexp += (d >= 300) & (d <= 1300)
h['n_exp'] = nexp
reg = h[h.region != '']
coast = h[(h.coast_km <= 50) & (h.on_land | (h.coast_km <= 10)) & (h.lat.abs() <= 45) & (h.n_exp >= 100)]
sel = pd.concat([reg, coast]).drop_duplicates('sid')
sel['in_coast_set'] = sel.sid.isin(coast.sid)
cols = ['sid', 'USAF', 'WBAN', 'name', 'CTRY', 'lat', 'lon', 'elev', 'n_years', 'coast_km', 'on_land', 'nx', 'ny',
        'grad_per_100km', 'region', 'n_exp', 'in_coast_set']
sel[cols].to_csv(os.path.join(OUT, 'isd_stations.csv'), index=False, float_format='%.4f')
yrs = inv[inv.sid.isin(sel.sid) & (inv.bytes >= 4000)][['sid', 'YEAR', 'bytes']]
yrs.to_csv(os.path.join(OUT, 'isd_station_years.csv'), index=False)
print('region stations:', len(reg), 'in', reg.region.nunique(), 'regions; coastal set:', len(coast), '; union', len(sel),
      '; station-years to fetch', len(yrs))
print(reg.groupby('region').size().to_string())
