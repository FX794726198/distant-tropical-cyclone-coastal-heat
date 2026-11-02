"""Build GraphCast input datasets for CTL / NOTC / WEAK / SHIFT / NULL and WIDE / WIDEH / NULLW variants (no model code here)."""
import os
import numpy as np, pandas as pd, xarray as xr
from tcfilter import DiskInfill, find_centre, gc_dist, destination, plane_deficit, shift_array
from symvortex import SymRemover

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
GD = (113.5, 23.0)
SURF = ['2m_temperature', 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind',
        'total_precipitation_6hr']
ATM = ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
       'specific_humidity']
VARIANTS = ['CTL', 'NOTC', 'WEAK', 'SHIFT', 'NULL']
WIDE_VARIANTS = ['CTL', 'WIDE', 'WIDEH', 'NULLW']
WIDE_ROUT = 1500.0
STN = pd.read_csv(os.path.join(TCH, 'inputs', 'gd_stations.csv'), index_col=0)
R_MAX, R_MIN, STN_BUFFER, TAPER = 800.0, 400.0, 150.0, 150.0


def load_time(t):
    """Global fields at 0.25 deg, lat ASCENDING (-90..90), as dict of numpy arrays."""
    f = os.path.join(TCH, 'era5', pd.Timestamp(t).strftime('%Y%m%d%H') + '.nc')
    ds = xr.open_dataset(f).sortby('latitude')
    out = {v: ds[v].transpose('latitude', 'longitude').values for v in SURF}
    for v in ATM:
        out[v] = ds[v].transpose('level', 'latitude', 'longitude').values
    lat, lon, lev = ds.latitude.values, ds.longitude.values, ds.level.values
    ds.close()
    return out, lat, lon, lev


def load_static():
    s = xr.open_dataset(os.path.join(TCH, 'era5', 'static.nc')).sortby('latitude')
    return {v: s[v].transpose('latitude', 'longitude').values.astype('float32') for v in ['geopotential_at_surface', 'land_sea_mask']}


def radius_for(lon0, lat0):
    dmin = gc_dist(STN.lon.values, STN.lat.values, lon0, lat0).min()
    return min(R_MAX, dmin - STN_BUFFER), dmin


class Window:
    """Regional window around a centre with a precomputed DiskInfill."""

    def __init__(self, lat, lon, lon0, lat0, R):
        half = (R + 350.0) / 111.0
        self.j = np.where((lat >= lat0 - half) & (lat <= lat0 + half))[0]
        hx = half / max(np.cos(np.radians(lat0 + np.sign(lat0) * half)), 0.2)
        self.i = np.where((lon >= lon0 - hx) & (lon <= lon0 + hx))[0]
        LON, LAT = np.meshgrid(lon[self.i], lat[self.j])
        self.F = DiskInfill(LAT, LON, lon0, lat0, R_km=R, taper_km=TAPER)
        self.sl = (slice(self.j[0], self.j[-1] + 1), slice(self.i[0], self.i[-1] + 1))

    def disturbance_global(self, f2d):
        D = np.zeros_like(f2d)
        D[self.sl] = self.F.disturbance(f2d[self.sl])
        return D


def sym_remove(out, lat, lon, lon0, lat0, R_in, frac, info):
    """In place: remove frac x the axisymmetric outer circulation (winds, geopotential, MSLP) out to WIDE_ROUT."""
    half = (WIDE_ROUT + 150.0) / 111.0
    j = np.where((lat >= lat0 - half) & (lat <= lat0 + half))[0]
    hx = half / max(np.cos(np.radians(min(abs(lat0) + half, 80.0))), 0.2)
    i = np.where((lon >= lon0 - hx) & (lon <= lon0 + hx))[0]
    sl = (slice(j[0], j[-1] + 1), slice(i[0], i[-1] + 1))
    LON, LAT = np.meshgrid(lon[i], lat[j])
    S = SymRemover(LAT, LON, lon0, lat0, R_in=R_in, R_out=WIDE_ROUT)
    k850 = 10  # index of 850 hPa in the 13 WB2 levels (50 ... 850, 925, 1000)
    vt = lambda: S.vt_profile(out['u_component_of_wind'][k850][sl], out['v_component_of_wind'][k850][sl])
    b = vt(); i6, i10 = int(600 / S.dr), int(1000 / S.dr)
    info['vt850_600_1000_before'] = (float(b[i6]), float(b[i10]))
    info['msl_anom_in_R_before'] = float(S.profile(out['mean_sea_level_pressure'][sl] / 100)[:int(R_in / S.dr)].mean()
                                         - (out['mean_sea_level_pressure'][sl][S.ring] / 100).mean())
    pairs = [('10m_u_component_of_wind', '10m_v_component_of_wind', None)] + \
            [('u_component_of_wind', 'v_component_of_wind', k) for k in range(out['u_component_of_wind'].shape[0])]
    for un, vn, k in pairs:
        u = out[un][sl] if k is None else out[un][k][sl]
        v = out[vn][sl] if k is None else out[vn][k][sl]
        du, dv = S.wind_part(u, v)
        if k is None:
            out[un][sl] = u - frac * du; out[vn][sl] = v - frac * dv
        else:
            out[un][k][sl] = u - frac * du; out[vn][k][sl] = v - frac * dv
    out['mean_sea_level_pressure'][sl] -= frac * S.scalar_part(out['mean_sea_level_pressure'][sl])
    for k in range(out['geopotential'].shape[0]):
        out['geopotential'][k][sl] -= frac * S.scalar_part(out['geopotential'][k][sl])
    a = vt()
    info['vt850_600_1000_after'] = (float(a[i6]), float(a[i10]))
    info['msl_anom_in_R_after'] = float(S.profile(out['mean_sea_level_pressure'][sl] / 100)[:int(R_in / S.dr)].mean()
                                        - (out['mean_sea_level_pressure'][sl][S.ring] / 100).mean())


def modify(fields, lat, lon, centre, kind, null_centre=None, shift_centre=None):
    """Return a modified copy of one time's fields."""
    if kind in ('WIDE', 'WIDEH', 'NULLW'):
        # core removed by the disk infill (as NOTC / WEAK / NULL), then the outer axisymmetric circulation
        base = {'WIDE': 'NOTC', 'WIDEH': 'WEAK', 'NULLW': 'NULL'}[kind]
        out, info = modify(fields, lat, lon, centre, base, null_centre=null_centre)
        if kind == 'NULLW' and null_centre is None:
            return out, info
        lon0, lat0 = null_centre if kind == 'NULLW' else centre
        sym_remove(out, lat, lon, lon0, lat0, info['R'], 0.5 if kind == 'WIDEH' else 1.0, info)
        return out, info
    out = {k: v.copy() for k, v in fields.items()}
    if kind == 'CTL':
        return out, {}
    lon0, lat0 = centre
    R, _ = radius_for(lon0, lat0)
    info = {'R': R}
    if kind == 'NULL':
        if null_centre is None:
            return out, {'R': R, 'null': 'no TC-free point found; NULL = CTL'}
        lon0, lat0 = null_centre
    W = Window(lat, lon, lon0, lat0, R)
    if kind == 'SHIFT':
        dj = int(round((shift_centre[1] - lat0) / (lat[1] - lat[0])))
        di = int(round((shift_centre[0] - lon0) / (lon[1] - lon[0])))
    amp = 0.5 if kind == 'WEAK' else 0.0

    def one(f2d):
        D = W.disturbance_global(f2d)
        g = f2d - (1 - amp) * D
        if kind == 'SHIFT':
            g = g + shift_array(D, dj, di)
        return g

    for v in SURF:
        out[v] = one(fields[v])
    out['total_precipitation_6hr'] = np.maximum(out['total_precipitation_6hr'], 0)
    for v in ATM:
        out[v] = np.stack([one(fields[v][k]) for k in range(fields[v].shape[0])])
    out['specific_humidity'] = np.maximum(out['specific_humidity'], 0)
    msl = fields['mean_sea_level_pressure'] / 100
    LON, LAT = np.meshgrid(lon[W.i], lat[W.j])
    info['deficit_before'] = plane_deficit(msl[W.sl], LAT, LON, lon0, lat0, min(R, 600))
    info['deficit_after'] = plane_deficit(out['mean_sea_level_pressure'][W.sl] / 100, LAT, LON, lon0, lat0, min(R, 600))
    return out, info


def null_centre_for(lon0, lat0, other_tc=()):
    """A TC-free ocean point at the same distance from Guangdong as the storm."""
    d0 = gc_dist(lon0, lat0, *GD)
    for b in (120, 140, 100, 160, 80, 60):
        x, y = destination(GD[0], GD[1], b, d0)
        far = gc_dist(x, y, lon0, lat0) > 1200 and all(gc_dist(x, y, a, c) > 1000 for a, c in other_tc)
        if far and y > 8:
            return float(x), float(y)
    return None


def shift_centre_for(lon0, lat0):
    """Same storm moved to the SW of Guangdong (bearing 225 deg), 500-800 km away."""
    d0 = float(np.clip(gc_dist(lon0, lat0, *GD), 500, 800))
    x, y = destination(GD[0], GD[1], 225, d0)
    return float(x), float(y)


def to_dataset(times, flist, lat, lon, lev, static, n_targets, sub=1):
    """GraphCast-style dataset: 2 input times + n_targets NaN target times."""
    sl = (slice(None, None, sub), slice(None, None, sub))
    la, lo = lat[::sub].astype('float32'), lon[::sub].astype('float32')
    nt = 2 + n_targets
    dt = pd.to_datetime([times[0] + pd.Timedelta(hours=6 * k) for k in range(nt)])
    data = {}
    for v in SURF:
        arr = np.full((1, nt, len(la), len(lo)), np.nan, 'float32')
        for k in range(2):
            arr[0, k] = flist[k][v][sl]
        data[v] = (('batch', 'time', 'lat', 'lon'), arr)
    for v in ATM:
        arr = np.full((1, nt, len(lev), len(la), len(lo)), np.nan, 'float32')
        for k in range(2):
            arr[0, k] = flist[k][v][(slice(None),) + sl]
        data[v] = (('batch', 'time', 'level', 'lat', 'lon'), arr)
    for v, a in static.items():
        data[v] = (('lat', 'lon'), a[sl])
    ds = xr.Dataset(data, coords={'lat': la, 'lon': lo, 'level': lev.astype('int32'),
                                  'time': pd.to_timedelta(np.arange(nt) * 6, unit='h').values.astype('timedelta64[ns]'),
                                  'datetime': (('batch', 'time'), dt.values.astype('datetime64[ns]')[None, :])})
    return ds
