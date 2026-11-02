import xarray as xr, numpy as np, pandas as pd, concurrent.futures as cf, time
URL='gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
_ds=None
def ds():
    global _ds
    if _ds is None: _ds=xr.open_zarr(URL,storage_options={'token':'anon'})
    return _ds
def extract(var, t0, t1, lat_idx, lon_idx, level=None, nthreads=24):
    """return times, array (time, npts) at given index arrays (lat_idx, lon_idx) for 6-hourly steps"""
    d=ds()[var]
    tt=ds().time.values
    i0=np.searchsorted(tt,np.datetime64(t0)); i1=np.searchsorted(tt,np.datetime64(t1))
    i0=(i0//8)*8
    blocks=[(a,min(a+8,i1)) for a in range(i0,i1,8)]
    if level is not None:
        li=int(np.where(ds().level.values==level)[0][0])
    def get(b):
        for k in range(5):
            try:
                if level is not None:
                    x=d.isel(time=slice(*b),level=li).values
                else:
                    x=d.isel(time=slice(*b)).values
                return x[:,lon_idx,lat_idx]
            except Exception as e:
                time.sleep(2)
        raise RuntimeError(b)
    with cf.ThreadPoolExecutor(nthreads) as ex:
        out=list(ex.map(get,blocks))
    return tt[i0:i1], np.concatenate(out,0)
def idx(lats, lons):
    la=ds().latitude.values; lo=ds().longitude.values
    lats=np.atleast_1d(lats); lons=np.atleast_1d(lons)%360
    li=np.array([np.argmin(abs(la-x)) for x in lats]); oi=np.array([np.argmin(abs((lo-x+180)%360-180)) for x in lons])
    return li,oi
