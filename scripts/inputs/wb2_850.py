import numpy as np, xarray as xr, time, pandas as pd, sys
from concurrent.futures import ThreadPoolExecutor
URL='gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
ds=xr.open_zarr(URL,storage_options={'token':'anon'})
Z=np.load('era5/z500_uv.npz',allow_pickle=True)
zlat=Z['lat']; zlon=Z['lon']
lat=ds.latitude.values; lon=ds.longitude.values
am=(lat>=zlat.min()-0.01)&(lat<=zlat.max()+0.01); lm=(lon>=zlon.min()-0.01)&(lon<=zlon.max()+0.01)
ai=np.where(am)[0]; li=np.where(lm)[0]
t=ds.time.values; tt=pd.DatetimeIndex(t)
sel=np.where((tt.year>=1979)&(tt.year<=2022)&(tt.month>=6)&(tt.month<=10))[0]
starts=sorted(set((s//8)*8 for s in sel))
lev=list(ds.level.values).index(850)
out={}
for var in ['temperature','u_component_of_wind','v_component_of_wind','specific_humidity']:
    sub=ds[var].isel(level=lev, longitude=slice(li.min(),li.max()+1), latitude=slice(ai.min(),ai.max()+1))
    def get(s):
        for k in range(6):
            try: return s, sub.isel(time=slice(s,s+8)).values.astype(np.float32)
            except Exception: time.sleep(3)
        raise RuntimeError(s)
    t0=time.time(); res={}
    with ThreadPoolExecutor(32) as ex:
        for s,v in ex.map(get,starts): res[s]=v
    arr=np.concatenate([res[s] for s in starts]); tim=np.concatenate([t[s:s+8] for s in starts])
    ti=pd.DatetimeIndex(tim); k=(ti.year>=1979)&(ti.year<=2022)&(ti.month>=6)&(ti.month<=10)
    arr=arr[k]; ti=ti[k]
    # dims: (time, lon, lat) or (time, lat, lon)?
    dims=sub.dims
    daily=pd.DataFrame(arr.reshape(len(ti),-1)).groupby(ti.normalize()).mean()
    out[var]=daily.values.astype(np.float32); days=daily.index.values
    print(var, dims, arr.shape, time.time()-t0, flush=True)
lo=lon[li.min():li.max()+1]; la=lat[ai.min():ai.max()+1]
np.savez_compressed('era5/wb2_850.npz', days=days, lon=lo, lat=la, dims=np.array(dims), **out)
print('done')
