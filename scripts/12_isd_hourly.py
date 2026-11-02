"""Hourly temperature, dew point and wind at the 19 stations (ISD-lite, NOAA) -> station-day table in local time (UTC+8).

Per station and local day: temperature, dew point, relative humidity and wet-bulb temperature (Stull 2011) at 14 LT,
their 12-17 LT means, the wet-bulb temperature at the hour of the highest hourly temperature, the daily maximum of the
hourly wet-bulb temperature, and the wind at 02 and 14 LT (for the cross-shore component at the coastal stations).
Usage: python -I 12_isd_hourly.py ISD_DIR OUT.parquet   (ISD_DIR holds USAF-WBAN-YEAR.gz files)
"""
import sys, os, glob
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
cols = ['y', 'm', 'd', 'h', 'T', 'Td', 'slp', 'wd', 'ws', 'sky', 'p1', 'p6']


def stull_tw(T, RH):
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


es = lambda T: 6.112 * np.exp(17.67 * T / (T + 243.5))
frames = []
for f in sorted(glob.glob(os.path.join(src, '*.gz'))):
    b = os.path.basename(f).split('-')
    sid = b[0] + b[1]
    d = pd.read_csv(f, sep=r'\s+', header=None, names=cols, na_values=[-9999], compression='gzip')
    if d.empty:
        continue
    d['T'] /= 10.0; d['Td'] /= 10.0; d['ws'] /= 10.0
    bad = (d['T'] < -10) | (d['T'] > 45) | (d['Td'] > d['T'] + 0.5) | (d['Td'] < -40)
    d.loc[bad, ['T', 'Td']] = np.nan
    t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h)) + pd.Timedelta('8h')       # local time
    d['day'] = t.dt.normalize(); d['lh'] = t.dt.hour
    d = d[d.day.dt.month.between(6, 10)]
    RH = np.clip(100 * es(d['Td']) / es(d['T']), 1, 100)
    d['RH'] = RH; d['Tw'] = stull_tw(d['T'], RH)
    d.loc[d.wd == 0, 'wd'] = np.nan                          # calm or variable: no direction
    d['STATION'] = sid
    frames.append(d[['STATION', 'day', 'lh', 'T', 'Td', 'RH', 'Tw', 'wd', 'ws']])
H = pd.concat(frames, ignore_index=True)
g = H.groupby(['STATION', 'day'])
day = pd.DataFrame({'n_obs': g['T'].count()})
at = lambda h, v: H[H.lh == h].set_index(['STATION', 'day'])[v].groupby(level=[0, 1]).first()
for v in ('T', 'Td', 'RH', 'Tw'):
    day[v + '14'] = at(14, v)
pm = H[H.lh.between(12, 17)].groupby(['STATION', 'day'])
for v in ('T', 'Td', 'RH', 'Tw'):
    m = pm[v].mean(); n = pm[v].count()
    day[v + '_pm'] = m.where(n >= 2)
ok = H.dropna(subset=['T'])
ix = ok.groupby(['STATION', 'day'])['T'].idxmax()
day['Tmax_h'] = ok.loc[ix].set_index(['STATION', 'day'])['T']
day['Tw_atTmax'] = ok.loc[ix].set_index(['STATION', 'day'])['Tw']
day['hour_Tmax'] = ok.loc[ix].set_index(['STATION', 'day'])['lh']
# daily values need at least four reports including 14 LT (stations reporting 4, 8 or 24 times a day); with 8 or more
# reports the stricter versions *_8 are also kept
nT = ok.groupby(['STATION', 'day'])['Tw'].count(); has14 = day['Tw14'].notna().reindex(nT.index).fillna(False)
twmax = ok.groupby(['STATION', 'day'])['Tw'].max()
day['Tw_max'] = twmax.where((nT >= 4) & has14); day['Tw_max_8'] = twmax.where(nT >= 8)
day['Tw_atTmax_8'] = day['Tw_atTmax'].where(nT >= 8)
day['Tmax_h'] = day['Tmax_h'].where((nT >= 4) & has14); day['Tw_atTmax'] = day['Tw_atTmax'].where((nT >= 4) & has14)
for h in (2, 14):
    for v in ('wd', 'ws'):
        day[f'{v}{h:02d}'] = at(h, v)
day = day.reset_index()
day.to_parquet(out, index=False)
print(day.shape, day.STATION.nunique(), day.day.min(), day.day.max())
print(day.describe().T[['count', 'mean']].round(2).to_string())
