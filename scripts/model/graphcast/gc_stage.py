"""Stage global ERA5 0.25 deg / 13-level fields (WeatherBench2 copy) for every time the pilot needs.

Usage: python gc_stage.py [--workers 8] [--cases 0-60]
Writes  $TCH/era5/YYYYmmddHH.nc  (float32, lat descending as in the store) and $TCH/era5/static.nc
"""
import argparse, os, threading, concurrent.futures as cf
import numpy as np, pandas as pd, xarray as xr

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
URL = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
SURF = ['2m_temperature', 'mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind',
        'total_precipitation_6hr']
ATM = ['temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
       'specific_humidity']
LEADS = [24, 30, 36, 42, 48]


def needed_times(cases):
    T = set()
    for V in pd.to_datetime(cases.V):
        T.add(V)
        for L in LEADS:
            T.add(V - pd.Timedelta(hours=L)); T.add(V - pd.Timedelta(hours=L + 6))
    return sorted(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cases', default=None, help='e.g. 0-60')
    a = ap.parse_args()
    cases = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv'))
    if a.cases:
        lo, hi = map(int, a.cases.split('-')); cases = cases[(cases.case >= lo) & (cases.case <= hi)]
    os.makedirs(os.path.join(TCH, 'era5'), exist_ok=True)
    ds = xr.open_zarr(URL, storage_options={'token': 'anon'}, consolidated=True)
    st = os.path.join(TCH, 'era5', 'static.nc')
    if not os.path.exists(st):
        ds[['geopotential_at_surface', 'land_sea_mask']].load().astype('float32').to_netcdf(st)
    times = needed_times(cases)
    todo = [t for t in times if not os.path.exists(os.path.join(TCH, 'era5', t.strftime('%Y%m%d%H') + '.nc'))]
    print(len(times), 'times needed,', len(todo), 'to download', flush=True)

    lock = threading.Lock()  # netCDF/HDF5 writes are not thread-safe

    def get(t):
        f = os.path.join(TCH, 'era5', t.strftime('%Y%m%d%H') + '.nc')
        x = ds[SURF + ATM].sel(time=t).load().astype('float32')
        with lock:
            x.to_netcdf(f + '.part'); os.replace(f + '.part', f)
        return t

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for k, t in enumerate(ex.map(get, todo)):
            if k % 10 == 0: print(k + 1, '/', len(todo), t, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
