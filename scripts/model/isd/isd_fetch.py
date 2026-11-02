"""A5 station check, step 2: download ISD-Lite (NOAA AWS mirror) for the selected stations, 1980-2021, and reduce each
station to local-solar-day values (local solar time = UTC + round(lon / 15) h, as for the global map):

  T_pm, Td_pm, slp_pm : means of the hourly reports at 12-16 LT
  u_pm, v_pm, ws_pm   : mean wind vector and mean speed at 12-16 LT (calm = zero vector; no direction = left out)
  n_pm                : number of 12-16 LT reports with a temperature
  Tmax, n_day         : highest hourly temperature and number of reports with a temperature (Tmax kept only when
                        n_day >= 8 and n_pm >= 1)
Quality limits: -40 <= T <= 55 C, Td <= T + 0.5, Td >= -60, ws <= 60 m/s.  Raw files are not kept.
Usage: python isd_fetch.py isd_stations.csv isd_station_years.csv OUT_DIR [N_PROC]
"""
import io, os, sys, gzip, time, urllib.request
from multiprocessing import Pool
import numpy as np, pandas as pd

B = 'https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/'
COLS = ['y', 'm', 'd', 'h', 'T', 'Td', 'slp', 'wd', 'ws', 'sky', 'p1', 'p6']


def get(url):
    for k in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            if getattr(e, 'code', None) == 404:
                return None
            time.sleep(2 ** k)
    return None


def one(args):
    sid, lon, years, out = args
    f = os.path.join(out, sid + '.parquet')
    if os.path.exists(f):
        return sid, -1
    fr = []
    for y in years:
        b = get(f'{B}{y}/{sid}-{y}.gz')
        if not b:
            continue
        try:
            d = pd.read_csv(io.BytesIO(gzip.decompress(b)), sep=r'\s+', header=None, names=COLS, na_values=[-9999])
        except Exception:
            continue
        fr.append(d)
    if not fr:
        pd.DataFrame().to_parquet(f); return sid, 0
    d = pd.concat(fr, ignore_index=True)
    d['T'] /= 10.0; d['Td'] /= 10.0; d['slp'] /= 10.0; d['ws'] /= 10.0
    d.loc[(d['T'] < -40) | (d['T'] > 55), 'T'] = np.nan
    d.loc[(d['Td'] > d['T'] + 0.5) | (d['Td'] < -60) | d['T'].isna(), 'Td'] = np.nan
    d.loc[(d.ws < 0) | (d.ws > 60), 'ws'] = np.nan
    calm = d.ws == 0
    okd = d.wd.between(1, 360)
    rad = np.radians(d.wd.where(okd))
    d['u'] = np.where(calm, 0.0, -d.ws * np.sin(rad)); d['v'] = np.where(calm, 0.0, -d.ws * np.cos(rad))
    s = int(np.round((((lon + 180) % 360) - 180) / 15))
    t = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h)) + pd.Timedelta(hours=s)
    d['day'] = t.dt.normalize(); d['lh'] = t.dt.hour
    g = d.groupby('day')
    D = pd.DataFrame({'n_day': g['T'].count(), 'Tmax': g['T'].max()})
    p = d[d.lh.between(12, 16)].groupby('day')
    for v, n in (('T', 'T_pm'), ('Td', 'Td_pm'), ('slp', 'slp_pm'), ('u', 'u_pm'), ('v', 'v_pm'), ('ws', 'ws_pm')):
        D[n] = p[v].mean()
    D['n_pm'] = p['T'].count()
    D['n_pm'] = D.n_pm.fillna(0).astype(np.int8); D['n_day'] = D.n_day.astype(np.int8)
    D.loc[(D.n_day < 8) | (D.n_pm < 1), 'Tmax'] = np.nan
    D = D[(D.n_pm > 0) | D.Tmax.notna()].astype({c: np.float32 for c in ('Tmax', 'T_pm', 'Td_pm', 'slp_pm', 'u_pm', 'v_pm', 'ws_pm')})
    D.reset_index().to_parquet(f, index=False)
    return sid, len(D)


if __name__ == '__main__':
    st = pd.read_csv(sys.argv[1], dtype={'USAF': str, 'WBAN': str}); yr = pd.read_csv(sys.argv[2])
    out = sys.argv[3]; npr = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    os.makedirs(out, exist_ok=True)
    ys = yr.groupby('sid').YEAR.apply(sorted)
    jobs = [(r.sid, r.lon, ys.get(r.sid, []), out) for r in st.itertuples()]
    t0 = time.time(); n = 0
    with Pool(npr) as pool:
        for sid, k in pool.imap_unordered(one, jobs):
            n += 1
            if n % 50 == 0:
                print(f'{n}/{len(jobs)} stations, {time.time() - t0:.0f} s', flush=True)
    print('done', n, f'{time.time() - t0:.0f} s')
