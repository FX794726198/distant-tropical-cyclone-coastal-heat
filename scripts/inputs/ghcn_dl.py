import sys, os, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
outdir=sys.argv[1]; lon0,lon1,lat0,lat1=map(float,sys.argv[2:6]); el=sys.argv[6]
inv=pd.read_fwf(os.path.join(outdir,'ghcnd-inventory.txt'),colspecs=[(0,11),(12,20),(21,30),(31,35),(36,40),(41,45)],names=['id','lat','lon','el','y0','y1'])
tx=inv[(inv.el==el)&(inv.lon>=lon0)&(inv.lon<=lon1)&(inv.lat>=lat0)&(inv.lat<=lat1)&(inv.y0<=1985)&(inv.y1>=2015)]
ids=sorted(set(tx.id))
os.makedirs(os.path.join(outdir,'st'),exist_ok=True)
def get(i):
    fn=os.path.join(outdir,'st',i+'.csv')
    if os.path.exists(fn): return
    r=requests.get(f'https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/{i}.csv',timeout=120)
    if r.status_code==200: open(fn,'wb').write(r.content)
with ThreadPoolExecutor(16) as ex: list(ex.map(get,ids))
print(len(ids))
