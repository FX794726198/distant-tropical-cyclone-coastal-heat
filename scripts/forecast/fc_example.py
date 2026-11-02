"""Forecast example for Fig. 4 (NC v10, exploratory): observed 2-m temperature anomaly on one storm-heat day and the
HRES forecasts of it at several lead times, with the observed and forecast storm positions.

Anomaly of a field = field minus the mean of the same product at the same valid hour (and, for forecasts, the same lead)
over days within +-WIN days of the case's day of year in 2016-2022 (every STEP days), excluding the case year +-0 days
window only for the case day itself.  0.25-degree WeatherBench2 data (HRES 2016-2022, ERA5).
Usage: python -I fc_example.py VALID(YYYY-MM-DDTHH, UTC, 00/06/12/18) SID LON0 LON1 LAT0 LAT1 IBTRACS OUT [LEADS ...]
"""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import gcsfs, zarr, xarray as xr

VALID = pd.Timestamp(sys.argv[1]); SID = sys.argv[2]
LON0, LON1, LAT0, LAT1 = map(float, sys.argv[3:7]); IBT, OUT = sys.argv[7], sys.argv[8]
LEADS = [int(x) for x in sys.argv[9:]] or [1, 3, 5, 7]
WIN, STEP = 15, 2
H = 'weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
E = 'weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
fs = gcsfs.GCSFileSystem(token='anon'); os.makedirs(OUT, exist_ok=True)


def sel(xd):
    lon = xd.longitude.values; lat = xd.latitude.values
    lo = np.where((lon >= LON0 % 360) & (lon <= LON1 % 360))[0] if LON0 % 360 <= LON1 % 360 else None
    la = np.where((lat >= LAT0) & (lat <= LAT1))[0]
    la = la[np.argsort(lat[la])]                                     # ascending latitude for both products
    return lo, la, lon[lo], lat[la]


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def clim_dates(t):
    doy = t.dayofyear; out = []
    for y in range(2016, 2023):
        c = pd.Timestamp(year=y, month=1, day=1, hour=t.hour) + pd.Timedelta(days=doy - 1)
        for k in range(-WIN, WIN + 1, STEP):
            d = c + pd.Timedelta(days=k)
            if d != t:
                out.append(d)
    return out


T0 = time.time()
# ---- ERA5 (truth)
xe = xr.open_zarr(fs.get_mapper(E), consolidated=True); ge = zarr.open_group(fs.get_mapper(E), mode='r')
lo, la, LON, LAT = sel(xe); te = pd.Series(np.arange(xe.sizes['time']), index=pd.DatetimeIndex(xe.time.values))
dims_e = xe['2m_temperature'].dims                                     # (time, longitude, latitude) or (time, latitude, longitude)


def era(var, t):
    a = np.asarray(ge[var][int(te[t])])
    a = a if dims_e[1] == 'longitude' else a.T
    return a[np.ix_(lo, la)]


with ThreadPoolExecutor(24) as ex:
    cl = [d for d in clim_dates(VALID) if d in te.index]
    obs = {v: era(v, VALID) for v in ('2m_temperature', 'mean_sea_level_pressure')}
    obs_clim = np.mean(list(ex.map(lambda d: era('2m_temperature', d), cl)), 0)
print(f'ERA5 done {time.time() - T0:.0f} s ({len(cl)} climatology days)', flush=True)

# ---- HRES forecasts
xh = xr.open_zarr(fs.get_mapper(H), consolidated=True); gh = zarr.open_group(fs.get_mapper(H), mode='r')
th = pd.Series(np.arange(xh.sizes['time']), index=pd.DatetimeIndex(xh.time.values))
lv = xh.prediction_timedelta.values
lh = (lv / np.timedelta64(1, 'h')).astype(int) if lv.dtype.kind == 'm' else lv.astype(int)
lpos = pd.Series(np.arange(len(lh)), index=lh); dims_h = xh['2m_temperature'].dims   # (time, lead, lon, lat) or (..., lat, lon)
lo, la, LONh, LATh = sel(xh); assert np.allclose(LONh, LON) and np.allclose(LATh, LAT), 'HRES and ERA5 grids differ'


def hres(var, init, L):
    a = np.asarray(gh[var][int(th[init]), int(lpos[L])])
    a = a if dims_h[2] == 'longitude' else a.T
    return a[np.ix_(lo, la)]


fc = {}
for d in LEADS:
    L = 24 * d + VALID.hour % 12; init = VALID - pd.Timedelta(hours=L)
    if init not in th.index or L not in lpos.index:
        print('lead', d, 'not available'); continue
    cl = [x - pd.Timedelta(hours=L) for x in clim_dates(VALID)]; cl = [x for x in cl if x in th.index]
    with ThreadPoolExecutor(24) as ex:
        c = np.mean(list(ex.map(lambda i: hres('2m_temperature', i, L), cl)), 0)
    t2 = hres('2m_temperature', init, L); ps = hres('mean_sea_level_pressure', init, L)
    fc[d] = dict(anom=t2 - c, mslp=ps / 100.0, init=str(init), lead_h=L)
    print(f'lead {d} d (init {init}, {L} h) done {time.time() - T0:.0f} s', flush=True)

# ---- storm positions: observed track and forecast centre (lowest MSLP within 600 km of the observed position)
tr = pd.read_csv(IBT, skiprows=[1], low_memory=False, usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND'])
tr = tr[tr.SID == SID].copy(); tr['time'] = pd.to_datetime(tr.ISO_TIME)
for c in ('LAT', 'LON'):
    tr[c] = pd.to_numeric(tr[c], errors='coerce')
tr = tr[tr.time.dt.minute == 0]
ob = tr[tr.time == VALID][['LON', 'LAT']].values
LO2, LA2 = np.meshgrid(((LON + 180) % 360) - 180, LAT, indexing='ij')
pos = {}
if len(ob):
    olon, olat = ob[0]
    for d, f in fc.items():
        m = hav(LO2, LA2, olon, olat) < 600
        k = np.argmin(np.where(m, f['mslp'], np.inf)); i, j = np.unravel_index(k, m.shape)
        pos[d] = (float(LO2[i, j]), float(LA2[i, j]), float(hav(LO2[i, j], LA2[i, j], olon, olat)))
        print(f'lead {d}: forecast centre {pos[d][0]:.1f},{pos[d][1]:.1f}, error {pos[d][2]:.0f} km')
track = tr[(tr.time >= VALID - pd.Timedelta(days=3)) & (tr.time <= VALID + pd.Timedelta(days=2))][['time', 'LON', 'LAT']]
np.savez_compressed(os.path.join(OUT, f'fc_example_{VALID:%Y%m%d%H}_{SID}.npz'), lon=LON, lat=LAT,
                    obs_anom=obs['2m_temperature'] - obs_clim, obs_mslp=obs['mean_sea_level_pressure'] / 100.0,
                    leads=np.array(list(fc)), fc_anom=np.stack([fc[d]['anom'] for d in fc]), fc_mslp=np.stack([fc[d]['mslp'] for d in fc]),
                    fc_pos=np.array([pos.get(d, (np.nan,) * 3) for d in fc]), track_lon=track.LON.values, track_lat=track.LAT.values,
                    track_time=track.time.values.astype('datetime64[h]').astype(str), valid=str(VALID), sid=SID)
print(f'saved; total {time.time() - T0:.0f} s')
