"""IBTrACS subset for the Guangdong observational analysis: 06 UTC fixes in 90-160E, 0-45N, June-October, 1980 onwards,
storms with a wind >= 25 kt at that fix (USA_WIND or WMO_WIND).  Compact integer arrays for transfer.

Usage: python ib_export.py IBTRACS_CSV OUT_NPZ
"""
import sys
import numpy as np, pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, skiprows=[1], low_memory=False,
                 usecols=['SID', 'SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'USA_PRES', 'WMO_PRES', 'NATURE', 'BASIN'],
                 keep_default_na=False)
for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'USA_PRES', 'WMO_PRES'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['time'] = pd.to_datetime(df.ISO_TIME)
df = df[(df.time.dt.hour == 6) & (df.time.dt.minute == 0) & df.time.dt.month.between(6, 10)]
df = df[df.LON.between(90, 160) & df.LAT.between(0, 45)]
w = df.USA_WIND.fillna(-1).clip(-1, 254); ww = df.WMO_WIND.fillna(-1).clip(-1, 254)
df = df[(w >= 25) | (ww >= 25)]
sids = np.array(sorted(df.SID.unique()))
meta = df.groupby('SID').agg(name=('NAME', 'first'), season=('SEASON', 'first'))
sid_i = np.searchsorted(sids, df.SID.values)
nat = {'TS': 0, 'ET': 1, 'DS': 2, 'MX': 3, 'SS': 4, 'NR': 5}
np.savez(dst, sid=sid_i.astype(np.int16), day=((df.time.dt.normalize() - pd.Timestamp('1980-01-01')).dt.days).values.astype(np.int16),
         lat10=np.round(df.LAT.values * 10).astype(np.int16), lon10=np.round(df.LON.values * 10).astype(np.int16),
         usa_wind=df.USA_WIND.fillna(255).values.astype(np.uint8), wmo_wind=df.WMO_WIND.fillna(255).values.astype(np.uint8),
         pres=df.WMO_PRES.fillna(df.USA_PRES).fillna(0).values.astype(np.int16),
         nature=df.NATURE.map(nat).fillna(9).values.astype(np.int8),
         sids=sids.astype('U13'), names=meta.loc[sids, 'name'].values.astype('U24'), seasons=meta.loc[sids, 'season'].values.astype(np.int16))
print(len(df), 'fixes', len(sids), 'storms', df.time.min(), df.time.max())
