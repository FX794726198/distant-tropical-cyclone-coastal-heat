"""Global TC-heat map, step 1 (login node): ERA5 hourly 2 m temperature (WeatherBench2, 1 deg, 1959-2021) reduced to the
daily maximum over the local solar day (UTC + lon/15 h, rounded) and the local 14:00 value, one file per year; plus the
1-deg land-sea mask / surface geopotential and IBTrACS v04r01 (since 1980).

Usage: python gl_get.py [--years 1980-2021] [--skip-ibtracs]
Output ($TCH/global/): era5_t2_YYYY.npz (tmax, t14 float32 [day, lon, lat]; days; lon; lat), static_1deg.npz,
                       ibtracs.since1980.list.v04r01.csv
"""
import argparse, os, time, urllib.request
import numpy as np, pandas as pd, xarray as xr

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
OUT = os.path.join(TCH, 'global')
ZARR = 'gs://weatherbench2/datasets/era5/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr'
IBT = ('https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/'
       'v04r01/access/csv/ibtracs.since1980.list.v04r01.csv')


def local_day_stats(H, lon, year):
    """H: hourly [time, lon, lat] from (year-1)-12-31 12:00 UTC to (year+1)-01-01 11:00 UTC (NaN where missing).
    Returns days, tmax, t14 [day, lon, lat] for the local solar days of `year`."""
    days = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
    nd = len(days)
    assert H.shape[0] == 24 * nd + 24, H.shape
    s = np.round((((lon + 180) % 360) - 180) / 15).astype(int)   # local solar time = UTC + s hours
    tmax = np.empty((nd,) + H.shape[1:], np.float32)
    t14 = np.empty_like(tmax)
    for sv in np.unique(s):
        i = np.where(s == sv)[0]
        start = 12 - sv                                            # index of local midnight of 1 January
        blk = H[start:start + 24 * nd, i, :].reshape(nd, 24, len(i), H.shape[2])
        tmax[:, i] = blk.max(1)                                    # NaN if any hour is missing
        t14[:, i] = blk[:, 14]
    return days, tmax, t14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', default='1980-2021')
    ap.add_argument('--skip-ibtracs', action='store_true')
    a = ap.parse_args()
    y0, y1 = map(int, a.years.split('-'))
    os.makedirs(OUT, exist_ok=True)
    ds = xr.open_zarr(ZARR, storage_options={'token': 'anon'})
    lon, lat = ds.longitude.values, ds.latitude.values
    st = os.path.join(OUT, 'static_1deg.npz')
    if not os.path.exists(st):
        np.savez(st, lsm=ds['land_sea_mask'].values.astype(np.float32),
                 zs=ds['geopotential_at_surface'].values.astype(np.float32), lon=lon, lat=lat)
    if not a.skip_ibtracs:
        f = os.path.join(OUT, os.path.basename(IBT))
        if not os.path.exists(f):
            urllib.request.urlretrieve(IBT, f + '.part'); os.rename(f + '.part', f)
            print('IBTrACS saved', os.path.getsize(f) // 2**20, 'MB', flush=True)
    tlast = pd.Timestamp(ds.time.values[-1])
    for year in range(y0, y1 + 1):
        f = os.path.join(OUT, f'era5_t2_{year}.npz')
        if os.path.exists(f):
            continue
        t0 = time.time()
        want = pd.date_range(f'{year - 1}-12-31 12:00', f'{year + 1}-01-01 11:00', freq='h')
        have = want[want <= tlast]
        v = ds['2m_temperature'].sel(time=slice(have[0], have[-1])).values.astype(np.float32)
        assert v.shape[0] == len(have)
        H = np.full((len(want),) + v.shape[1:], np.nan, np.float32)
        H[:len(have)] = v
        del v
        days, tmax, t14 = local_day_stats(H, lon, year)
        np.savez(f + '.tmp.npz', tmax=tmax, t14=t14, days=days.values.astype('datetime64[D]'), lon=lon, lat=lat)
        os.rename(f + '.tmp.npz', f)
        print(f'{year}: {time.time() - t0:.0f} s, NaN days {int(np.isnan(tmax).any(axis=(1, 2)).sum())}', flush=True)


if __name__ == '__main__':
    main()
