import os, sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
d=sys.argv[1]; out=sys.argv[2]; y0,y1=int(sys.argv[3]),int(sys.argv[4])
days=pd.date_range(f'{y0}-01-01',f'{y1}-12-31')
st=pd.read_fwf(os.path.join(d,'ghcnd-stations.txt'),colspecs=[(0,11),(12,20),(21,30),(31,37),(41,71)],names=['id','lat','lon','elev','name'])
files=sorted(os.listdir(os.path.join(d,'st')))
def load(f):
    df=pd.read_csv(os.path.join(d,'st',f),usecols=[0,1,2,3,5],dtype={'ID':str,'DATE':str,'ELEMENT':str,'DATA_VALUE':float,'Q_FLAG':str})
    df=df[df.ELEMENT.isin(['TMAX','TMIN','PRCP'])&df.Q_FLAG.isna()]
    df['DATE']=pd.to_datetime(df.DATE,format='%Y%m%d')
    df=df[(df.DATE>=days[0])&(df.DATE<=days[-1])]
    res={}
    for e in ['TMAX','TMIN','PRCP']:
        s=df[df.ELEMENT==e].set_index('DATE').DATA_VALUE/10.0
        s=s[~s.index.duplicated()]
        res[e]=s.reindex(days).values.astype(np.float32)
    return f[:-4],res
with ProcessPoolExecutor(4) as ex:
    R=list(ex.map(load,files))
ids=[r[0] for r in R]
arr={e:np.stack([r[1][e] for r in R]) for e in ['TMAX','TMIN','PRCP']}
meta=st.set_index('id').loc[ids]
np.savez_compressed(out,ids=np.array(ids),lat=meta.lat.values,lon=meta.lon.values,elev=meta.elev.values,days=days.values,**arr)
print(len(ids), {e:np.isfinite(arr[e]).mean() for e in arr})
