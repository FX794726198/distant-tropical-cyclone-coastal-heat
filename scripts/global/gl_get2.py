"""Global TC-heat map, stage 2 data (login node): daily means of ERA5 500-hPa geopotential height, 850-hPa temperature and
specific humidity, total cloud cover and 10-m wind, and the 24-h precipitation, on the 2.8125-deg WeatherBench2 grid
(6-hourly), 1980-2021, one file per year.  The day is the local solar day as in gl_get.py, to the nearest 6 h: the mean
of the four 6-hourly values that start at local midnight, and the precipitation accumulated over the 24 h that end at
the next local midnight.

Usage: python gl_get2.py [--years 1980-2021]
Output ($TCH/global/): era5_c_YYYY.npz (z500 m, t850 K, q850 g/kg, tcc 0-1, tp24 mm, u10, v10 m/s; float32
                       [day, lon, lat]; days; lon; lat)
"""
import argparse, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, xarray as xr

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
OUT = os.path.join(TCH, 'global')
ZARR = 'gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_conservative.zarr'
VARS = dict(z500=('geopotential', 500, 1 / 9.80665), t850=('temperature', 850, 1.0), q850=('specific_humidity', 850, 1000.0),
            tcc=('total_cloud_cover', None, 1.0), tp24=('total_precipitation_24hr', None, 1000.0),
            u10=('10m_u_component_of_wind', None, 1.0), v10=('10m_v_component_of_wind', None, 1.0))


def local_days(H, lon, nd, accum=False):
    """H: 6-hourly [time, lon, lat] from (year-1)-12-31 00 UTC.  Local day d at a longitude with UTC offset 6k h starts at
    index 4 (d + 1) - k.  Mean of the 4 values from there, or for an accumulation the value 4 steps later."""
    k = np.round((((lon + 180) % 360) - 180) / 90).astype(int)
    out = np.empty((nd,) + H.shape[1:], np.float32)
    for kv in np.unique(k):
        i = np.where(k == kv)[0]
        s0 = 4 - kv
        if accum:
            out[:, i] = H[s0 + 4:s0 + 4 + 4 * nd:4][:, i]
        else:
            out[:, i] = H[s0:s0 + 4 * nd][:, i].reshape(nd, 4, len(i), H.shape[2]).mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='1980-2021')
    ap.add_argument('--threads', type=int, default=7)
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split('-'))
    os.makedirs(OUT, exist_ok=True)
    ds = xr.open_zarr(ZARR, storage_options={'token': 'anon'}, chunks=None)
    lon, lat = ds.longitude.values, ds.latitude.values
    tlast = pd.Timestamp(ds.time.values[-1])
    for year in range(y0, y1 + 1):
        f = os.path.join(OUT, f'era5_c_{year}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        days = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D'); nd = len(days)
        want = pd.date_range(f'{year - 1}-12-31 00:00', periods=4 * (nd + 3), freq='6h')
        have = want[want <= tlast]

        def get(item):
            name, (var, lev, fac) = item
            da = ds[var]
            if lev is not None:
                da = da.sel(level=lev)
            v = da.sel(time=slice(have[0], have[-1])).transpose('time', 'longitude', 'latitude').values.astype(np.float32) * np.float32(fac)
            assert v.shape[0] == len(have), (name, v.shape)
            H = np.full((len(want),) + v.shape[1:], np.nan, np.float32); H[:len(have)] = v
            return name, local_days(H, lon, nd, accum=(name == 'tp24'))

        with ThreadPoolExecutor(a.threads) as ex:
            res = dict(ex.map(get, VARS.items()))
        np.savez(f + '.tmp.npz', days=days.values.astype('datetime64[D]'), lon=lon, lat=lat, **res)
        os.rename(f + '.tmp.npz', f)
        print(f'{year}: {time.time() - t0:.0f} s, NaN days {int(np.isnan(res["z500"]).any(axis=(1, 2)).sum())}', flush=True)


if __name__ == '__main__':
    main()
