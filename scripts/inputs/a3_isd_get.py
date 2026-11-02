import urllib.request, os, concurrent.futures as cf, pandas as pd, io
st=['59493099999','59501099999','59663099999','59673099999','59658099999']
out='raw/isd'
def get(a):
    s,y=a; fn=f'{out}/{y}_{s}.csv'
    if os.path.exists(fn): return fn
    try:
        b=urllib.request.urlopen(f'https://noaa-global-hourly-pds.s3.amazonaws.com/{y}/{s}.csv',timeout=90).read()
        open(fn,'wb').write(b); return fn
    except Exception as e: return None
jobs=[(s,y) for s in st for y in range(1979,2023)]
with cf.ThreadPoolExecutor(24) as ex: res=list(ex.map(get,jobs))
dfs=[]
for f in res:
    if not f: continue
    try:
        d=pd.read_csv(f,usecols=['STATION','DATE','WND','TMP','DEW'],dtype=str)
    except Exception: continue
    d['DATE']=pd.to_datetime(d.DATE); d=d[d.DATE.dt.month.between(6,10)]
    w=d.WND.str.split(',',expand=True); d['wd']=pd.to_numeric(w[0],errors='coerce'); d['wtype']=w[2]; d['ws']=pd.to_numeric(w[3],errors='coerce')/10
    d.loc[(d.wd==999)|(w[1].isin(['2','3','6','7'])),'wd']=float('nan'); d.loc[d.ws>90,'ws']=float('nan')
    for c in ['TMP','DEW']:
        t=d[c].str.split(',',expand=True); v=pd.to_numeric(t[0],errors='coerce')/10; v[v>90]=float('nan'); d[c]=v
    dfs.append(d[['STATION','DATE','wd','wtype','ws','TMP','DEW']])
D=pd.concat(dfs); D.to_parquet('data/inputs/isd_gd.parquet'); print(D.shape, D.STATION.value_counts())
