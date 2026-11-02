"""Monthly Nino-3.4 SST anomaly (5S-5N, 170W-120W) from WeatherBench2 ERA5 1.5 deg, 1979-2022; base 1991-2020."""
import time, numpy as np, pandas as pd, gcsfs, zarr, xarray as xr
from concurrent.futures import ThreadPoolExecutor
fs = gcsfs.GCSFileSystem(token='anon')
p = 'weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
xd = xr.open_zarr(fs.get_mapper(p), consolidated=True); g = zarr.open_group(fs.get_mapper(p), mode='r')
t = pd.DatetimeIndex(xd.time.values); lon = xd.longitude.values; lat = xd.latitude.values
io = np.where((lon >= 190) & (lon <= 240))[0]; jo = np.where((lat >= -5) & (lat <= 5))[0]
w = np.cos(np.radians(lat[jo]))
z = g['sea_surface_temperature']; print(z.shape, z.chunks, lon[io][[0, -1]], lat[jo][[0, -1]])
i0 = np.searchsorted(t, pd.Timestamp('1979-01-01')); i1 = np.searchsorted(t, pd.Timestamp('2023-01-01'))
starts = list(range(i0 - i0 % 8, i1, 8))
def rd(s):
    a = np.asarray(z[s:s + 8, io[0]:io[-1] + 1, jo[0]:jo[-1] + 1])
    return s, (np.nanmean(a * w[None, None, :], axis=(1, 2)) / w.mean())
T0 = time.time(); out = {}
with ThreadPoolExecutor(32) as ex:
    for s, v in ex.map(rd, starts):
        for k, x in enumerate(v):
            out[t[s + k]] = x
S = pd.Series(out).sort_index(); S = S[(S.index >= '1979-01-01') & (S.index < '2023-01-01')]
M = S.resample('MS').mean()
clim = M['1991':'2020'].groupby(M['1991':'2020'].index.month).mean()
A = M - clim.reindex(M.index.month).values
pd.DataFrame(dict(sst=M, nino34=A)).to_csv('nino34_era5_monthly.csv', float_format='%.4f')
print(f'{time.time() - T0:.0f} s', A['2015':'2016'].round(2).values[:14])
