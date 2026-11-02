import sys, os, io, concurrent.futures as cf, urllib.request, pandas as pd
def fetch(st, y, outdir):
    fn=f"{outdir}/{y}_{st}.csv"
    if os.path.exists(fn): return fn
    url=f"https://noaa-gsod-pds.s3.amazonaws.com/{y}/{st}.csv"
    try:
        b=urllib.request.urlopen(url,timeout=60).read()
        open(fn,'wb').write(b); return fn
    except Exception as e:
        return None
def load(stations, years, outdir, nthreads=32):
    os.makedirs(outdir,exist_ok=True)
    jobs=[(s,y) for s in stations for y in years]
    with cf.ThreadPoolExecutor(nthreads) as ex:
        res=list(ex.map(lambda a: fetch(a[0],a[1],outdir), jobs))
    dfs=[]
    for f in res:
        if f:
            try:
                d=pd.read_csv(f,usecols=['STATION','DATE','LATITUDE','LONGITUDE','ELEVATION','NAME','TEMP','DEWP','WDSP','MXSPD','MAX','MIN','PRCP','PRCP_ATTRIBUTES','SLP'],dtype={'STATION':str})
                dfs.append(d)
            except Exception as e: pass
    d=pd.concat(dfs,ignore_index=True)
    d['DATE']=pd.to_datetime(d['DATE'])
    for c,m in [('MAX',9999.9),('MIN',9999.9),('TEMP',9999.9),('DEWP',9999.9),('PRCP',99.99),('WDSP',999.9),('MXSPD',999.9),('SLP',9999.9)]:
        d.loc[d[c]>=m,c]=float('nan')
    for c in ['MAX','MIN','TEMP','DEWP']: d[c]=(d[c]-32)*5/9
    d['PRCP']=d['PRCP']*25.4
    return d
