"""How well do the WRF control runs of the six coasts reproduce the case day? (second text review, 2026-11-02)

For each of the 17 control runs, over the same region (non-Guangdong: land points of the region's 1-degree cells, cell
means averaged; Guangdong: the 19 stations) and the same daytime window (08-18 local solar time of the case day, local
time = UTC + round(lon / 15) h, as wrfnc_analyse.py):
  WRF control (wrfnc_runs.csv / wrfnc_hourly.csv, final runs): Tmax (per-point maximum, region mean), daytime mean
    onshore 10-m wind, short-wave down at the surface, cloud (column maximum), 2-m specific humidity, rain over the
    window and over the local day;
  ERA5 (hourly, 0.25 deg, ARCO-ERA5 on Google Cloud, read-only): the same quantities on the ERA5 grid (total cloud cover
    for cloud);
  stations (ISD-Lite, NOAA AWS mirror): every station inside the region's boxes (Guangdong: the 19 stations) with
    >= 4 temperature reports in the window: Tmax, daytime onshore wind, and rain reported in the day (6-h or 1-h totals);
    the same station points in ERA5 (nearest grid point) and, at the case time V, in the WRF d02 map (nearest point of
    the thinned map) for T2.
  storm at V: WRF control d01 sea-level pressure minimum within 400 km of the IBTrACS position (wrfnc_tracks.csv) and its
    distance from it; ERA5 mean-sea-level-pressure minimum in the same search; IBTrACS wind.
Exploratory: no thresholds; the table is for judging which control runs represent the day well.
Usage: python3 wrf_ctl_check.py OUTDIR [CASE ...]
Inputs (env): WRFNC_OUT (wrfnc_out/all), WRFNC_IN (wrfnc_inputs), GDST (paper/data/db/gd_stations.csv),
ISDHIST (isd-history.csv), MAPS (wrfnc_maps.npz)."""
import gzip, io, os, sys, time, urllib.request
import numpy as np, pandas as pd, xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = sys.argv[1]; os.makedirs(OUTD, exist_ok=True)
WO = os.environ.get('WRFNC_OUT', os.path.join(HERE, '..', '..', '..', 'data', 'wrfnc'))
WI = os.environ.get('WRFNC_IN', os.path.join(HERE, '..', '..', '..', 'data', 'wrfnc_inputs'))
GDST = os.environ['GDST']; HIST = os.environ['ISDHIST']; MAPS = os.environ['MAPS']
R = pd.read_csv(f'{WO}/wrfnc_runs.csv'); R['V'] = pd.to_datetime(R.V)
H = pd.read_csv(f'{WO}/wrfnc_hourly.csv'); H['time'] = pd.to_datetime(H.time)
TR = pd.read_csv(f'{WI}/wrfnc_tracks.csv'); TR['time'] = pd.to_datetime(TR.time)
NORM = pd.read_csv(f'{WI}/coast_normals.csv').set_index('region')
MP = np.load(MAPS)
ISD = pd.read_csv(HIST, dtype=str)
ISD['lat'] = pd.to_numeric(ISD.LAT, errors='coerce'); ISD['lon'] = pd.to_numeric(ISD.LON, errors='coerce')
ISD['b'] = pd.to_datetime(ISD.BEGIN, format='%Y%m%d', errors='coerce'); ISD['e'] = pd.to_datetime(ISD.END, format='%Y%m%d', errors='coerce')
ISD['sid'] = ISD.USAF + '-' + ISD.WBAN
ERA = xr.open_zarr('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3', chunks=None,
                   storage_options=dict(token='anon'))
CASES = [int(c) for c in sys.argv[2:]] or sorted(R[R.run == 'ctl'].case.unique())


def km(lon, lat, lon0, lat0):
    p1, p2 = np.radians(lat), np.radians(lat0); dl = np.radians(lon0 - lon)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def era(var, times, la0, la1, lo0, lo1):
    """ERA5 field(s) over a box; longitudes in -180..180 in, ERA5 0..360 inside."""
    lo0_, lo1_ = lo0 % 360, lo1 % 360
    da = ERA[var].sel(time=times, latitude=slice(la1, la0))
    if lo0_ <= lo1_:
        da = da.sel(longitude=slice(lo0_, lo1_))
    else:
        da = xr.concat([da.sel(longitude=slice(lo0_, 360)), da.sel(longitude=slice(0, lo1_))], 'longitude')
    for k in range(4):
        try:
            v = da.values; break
        except Exception:
            time.sleep(2 ** k)
    lon = da.longitude.values; lon = np.where(lon > 180, lon - 360, lon)
    return v, da.latitude.values, lon


def isd_year(sid, year):
    url = f'https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/{year}/{sid}-{year}.gz'
    for k in range(4):
        try:
            b = urllib.request.urlopen(url, timeout=60).read(); break
        except Exception as e:
            if getattr(e, 'code', None) == 404:
                return None
            time.sleep(2 ** k)
    else:
        return None
    d = pd.read_csv(io.BytesIO(gzip.decompress(b)), sep=r'\s+', header=None,
                    names=['y', 'm', 'd', 'h', 'T', 'Td', 'slp', 'wd', 'ws', 'sky', 'p1', 'p6']).replace(-9999, np.nan)
    d['time'] = pd.to_datetime(dict(year=d.y, month=d.m, day=d.d, hour=d.h))
    d['T'] /= 10; d['ws'] /= 10; d['p1'] /= 10; d['p6'] /= 10
    d.loc[(d['T'] < -40) | (d['T'] > 55), 'T'] = np.nan
    return d


rows, srows = [], []
for case in CASES:
    r = R[(R.case == case) & (R.run == 'ctl')].iloc[0]; coast = r.coast; V = r.V
    if coast == 'GD':
        nx, ny, lonc = 0.0, -1.0, 113.5
        st = pd.read_csv(GDST); st['sid'] = st.STATION.astype(str).str[:6] + '-' + st.STATION.astype(str).str[6:]
        pts = st[['lon', 'lat']].values; boxes = None
    else:
        n = NORM.loc[coast]; nx, ny, lonc = float(n.nx), float(n.ny), float(n.lon)
        boxes = pd.read_csv(f'{WI}/cells_{coast}.csv')[['lon', 'lat']].values
    off = pd.Timedelta(hours=float(np.round(lonc / 15.0)))
    day = (V + off).normalize(); w0 = day + pd.Timedelta('8h') - off; w1 = day + pd.Timedelta('18h') - off
    hrs = pd.date_range(w0, w1, freq='h'); dhrs = pd.date_range(day - off + pd.Timedelta('1h'), day - off + pd.Timedelta('24h'), freq='h')
    # ---- WRF control
    h = H[(H.case == case) & (H.run == 'ctl')].set_index('time')
    def acc(t):
        return float(h.rainacc.asof(t)) if t >= h.index[0] else np.nan
    wrf = dict(Tmax=r.Tmax - 273.15, onshore=r.onshore_day, SW=r.SWDOWN_day, cloud=r.CLDMAX_day, Q2=r.Q2_day,
               rain_win=acc(w1) - acc(w0), rain_day=(acc(dhrs[-1]) - acc(dhrs[0] - pd.Timedelta('1h'))) if dhrs[-1] <= h.index[-1] else np.nan)
    # ---- ERA5 region
    if boxes is not None:
        la0, la1 = boxes[:, 1].min() - 0.5, boxes[:, 1].max() + 0.5; lo0, lo1 = boxes[:, 0].min() - 0.5, boxes[:, 0].max() + 0.5
    else:
        la0, la1 = pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5; lo0, lo1 = pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5
    lsm, la, lo = era('land_sea_mask', [hrs[0]], la0, la1, lo0, lo1); lsm = lsm[0]
    LO, LA = np.meshgrid(lo, la)
    if boxes is not None:
        masks = []
        for bl, bb in boxes:
            m = (np.abs(LA - bb) <= 0.5 + 1e-6) & (np.abs(LO - bl) <= 0.5 + 1e-6)
            masks.append(m & (lsm > 0.5) if (m & (lsm > 0.5)).any() else m)
    else:
        masks = []
        for pl, pb in pts:
            m = np.zeros(LA.shape, bool); j, i = np.unravel_index(np.argmin((LA - pb) ** 2 + (LO - pl) ** 2), LA.shape); m[j, i] = True
            masks.append(m)
    def reg(a):
        return float(np.nanmean([np.nanmean(a[m]) for m in masks]))
    T2, *_ = era('2m_temperature', hrs, la0, la1, lo0, lo1)
    U, *_ = era('10m_u_component_of_wind', hrs, la0, la1, lo0, lo1); Vv, *_ = era('10m_v_component_of_wind', hrs, la0, la1, lo0, lo1)
    TD, *_ = era('2m_dewpoint_temperature', hrs, la0, la1, lo0, lo1); SP, *_ = era('surface_pressure', hrs, la0, la1, lo0, lo1)
    SS, *_ = era('surface_solar_radiation_downwards', hrs, la0, la1, lo0, lo1)
    TCC, *_ = era('total_cloud_cover', hrs, la0, la1, lo0, lo1)
    TP, *_ = era('total_precipitation', dhrs, la0, la1, lo0, lo1)
    e = 6.112 * np.exp(17.67 * (TD - 273.15) / (TD - 273.15 + 243.5)); q = 1000 * 0.622 * e / (SP / 100 - 0.378 * e)
    inw = np.isin(dhrs, hrs[1:])
    e5 = dict(Tmax=reg(T2.max(0)) - 273.15, onshore=float(np.mean([reg(-(U[k] * nx + Vv[k] * ny)) for k in range(len(hrs))])),
              SW=float(np.mean([reg(SS[k]) for k in range(len(hrs))])) / 3600, cloud=float(np.mean([reg(TCC[k]) for k in range(len(hrs))])),
              Q2=float(np.mean([reg(q[k]) for k in range(len(hrs))])),
              rain_win=1000 * reg(TP[inw].sum(0)), rain_day=1000 * reg(TP.sum(0)))
    # ---- storm at V
    tr = TR[TR.case == case].set_index('time').sort_index()
    tV = tr.reindex(tr.index.union([V])).interpolate('time').loc[V]
    slp = MP[f'{case}_ctl_d01__SLP']; sla = MP[f'{case}_ctl_d01__lat']; slo = MP[f'{case}_ctl_d01__lon']
    dd = km(slo, sla, tV.lon, tV.lat); mm = np.where(dd <= 400, slp, np.nan)
    if np.isfinite(mm).any():
        j = np.nanargmin(mm); wmin, wpos = float(mm.flat[j]), float(dd.flat[j])
    else:
        wmin, wpos = np.nan, np.nan
    ms, mla, mlo = era('mean_sea_level_pressure', [V], tV.lat - 4.5, tV.lat + 4.5, tV.lon - 4.5, tV.lon + 4.5)
    MLO, MLA = np.meshgrid(mlo, mla); de = km(MLO, MLA, tV.lon, tV.lat); me = np.where(de <= 400, ms[0] / 100, np.nan)
    je = np.nanargmin(me); emin, epos = float(me.flat[je]), float(de.flat[je])
    # ---- stations
    if boxes is not None:
        S = ISD[(ISD.b <= day) & (ISD.e >= day)].copy()
        inside = np.zeros(len(S), bool)
        for bl, bb in boxes:
            inside |= (np.abs(S.lat.values - bb) <= 0.5) & (np.abs(S.lon.values - bl) <= 0.5)
        S = S[inside][['sid', 'lat', 'lon']]
    else:
        S = st[['sid', 'lat', 'lon']]
    wla, wlo, wT = MP[f'{case}_ctl__lat'], MP[f'{case}_ctl__lon'], MP[f'{case}_ctl__T2']
    for s in S.itertuples():
        yrs = sorted({(day - off).year, (day - off + pd.Timedelta('24h')).year})
        d = pd.concat([x for x in (isd_year(s.sid, y) for y in yrs) if x is not None] or [pd.DataFrame()])
        if d.empty:
            continue
        d = d.set_index('time')
        win = d.loc[(d.index >= w0) & (d.index <= w1)]
        nT = int(win['T'].notna().sum())
        if nT < 4:
            continue
        wd, ws = win.wd, win.ws
        uu = -ws * np.sin(np.radians(wd)); vv = -ws * np.cos(np.radians(wd))
        on = (-(uu * nx + vv * ny)).where(ws > 0, 0.0).where(wd.notna() | (ws == 0))
        dday = d.loc[(d.index > dhrs[0] - pd.Timedelta('1h')) & (d.index <= dhrs[-1])]
        p6 = dday.p6.dropna(); p1 = dday.p1.dropna()
        rain = float(p6[p6.index.hour % 6 == 0].sum()) if len(p6) else (float(p1.sum()) if len(p1) >= 12 else np.nan)
        j, i = np.unravel_index(np.argmin((LA - s.lat) ** 2 + (LO - s.lon) ** 2), LA.shape)
        tV_obs = win['T'].get(V, np.nan) if V in win.index else (d['T'].get(V, np.nan) if V in d.index else np.nan)
        kk = np.unravel_index(np.argmin((wla - s.lat) ** 2 + ((wlo - s.lon) * np.cos(np.radians(s.lat))) ** 2), wla.shape)
        dwr = km(wlo[kk], wla[kk], s.lon, s.lat)
        tV_e5, *_ = era('2m_temperature', [V], s.lat - 0.2, s.lat + 0.2, s.lon - 0.2, s.lon + 0.2)
        srows.append(dict(case=case, coast=coast, sid=s.sid, lat=s.lat, lon=s.lon, n_T=nT, Tmax_obs=float(win['T'].max()),
                          Tmax_e5=float(T2[:, j, i].max() - 273.15), onshore_obs=float(on.mean()) if on.notna().sum() >= 3 else np.nan,
                          onshore_e5=float(np.mean(-(U[:, j, i] * nx + Vv[:, j, i] * ny))), rain_obs=rain,
                          rain_e5=float(1000 * TP[:, j, i].sum()), TV_obs=tV_obs, TV_e5=float(np.nanmean(tV_e5) - 273.15),
                          TV_wrf=float(wT[kk] - 273.15) if dwr < 15 else np.nan))
    SD = pd.DataFrame([x for x in srows if x['case'] == case])
    row = dict(case=case, coast=coast, name=r['name'], V=V, n_sta=len(SD),
               **{f'{k}_wrf': v for k, v in wrf.items()}, **{f'{k}_e5': v for k, v in e5.items()},
               storm_wind_kt=float(tV.wind), storm_slp_wrf=wmin, storm_dist_wrf=wpos, storm_slp_e5=emin, storm_dist_e5=epos)
    if len(SD):
        row.update(Tmax_obs=SD.Tmax_obs.mean(), Tmax_e5sta=SD.Tmax_e5.mean(), onshore_obs=SD.onshore_obs.mean(),
                   onshore_e5sta=SD.onshore_e5.mean(), rain_obs=SD.rain_obs.mean(), rain_e5sta=SD.rain_e5.mean(),
                   n_rain_obs=int(SD.rain_obs.notna().sum()), TV_obs=SD.TV_obs.mean(), TV_wrf=SD.TV_wrf.mean(), TV_e5sta=SD.TV_e5.mean(),
                   n_TV=int((SD.TV_obs.notna() & SD.TV_wrf.notna()).sum()),
                   TV_wrf_minus_obs=float((SD.TV_wrf - SD.TV_obs).mean()), TV_e5_minus_obs=float((SD.TV_e5 - SD.TV_obs).mean()))
    rows.append(row); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(f'{OUTD}/ctl_check_cases.csv', index=False, float_format='%.3f')
    pd.DataFrame(srows).to_csv(f'{OUTD}/ctl_check_stations.csv', index=False, float_format='%.3f')
