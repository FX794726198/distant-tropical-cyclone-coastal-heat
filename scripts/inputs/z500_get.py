import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wb2, numpy as np, pandas as pd, time
t=time.time()
ds=wb2.ds()
la=ds.latitude.values; lo=ds.longitude.values
Li=np.where((la>=5)&(la<=45))[0]; Oi=np.where((lo>=90)&(lo<=160))[0]
L,O=np.meshgrid(Li,Oi,indexing='ij'); L=L.ravel(); O=O.ravel()
out={}
for var,lev in [('geopotential',500),('10m_u_component_of_wind',None),('10m_v_component_of_wind',None)]:
    T=[];X=[]
    for y in range(1979,2023):
        tt,x=wb2.extract(var,f'{y}-06-01',f'{y}-11-01',L,O,level=lev,nthreads=32)
        T.append(tt);X.append(x.astype(np.float32))
    tt=np.concatenate(T); x=np.concatenate(X)
    df=pd.DataFrame(x,index=pd.to_datetime(tt))
    df=df[(df.index.month>=6)&(df.index.month<=10)].resample('D').mean().dropna(how='all')
    out[var]=df.values.astype(np.float32); days=df.index.values
    print(var,x.shape,round(time.time()-t),flush=True)
np.savez_compressed('era5/z500_uv.npz',z=out['geopotential'],u=out['10m_u_component_of_wind'],v=out['10m_v_component_of_wind'],days=days,lat=la[L],lon=lo[O],nlat=len(Li),nlon=len(Oi))
print('done',round(time.time()-t))
