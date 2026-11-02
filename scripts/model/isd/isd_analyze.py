"""A5 station check, step 3 (exploratory; nothing here is a pass/fail test).

Station days are treated exactly like the 1-degree cells of the global map (gl_map.py):
  exposure code per local solar day: nearest tropical-storm fix (IBTrACS, NATURE TS, >= 34 kt, 6-hourly) 300-1300 km
  away -> sector 0..7 of its bearing; -1 reference (none within 1300 km); -2 one within 300 km.
  anomalies: 3 harmonics + linear trend removed; inside the season, each year's mean of reference days and the
  +-15-day calendar mean of reference days removed (departure from year- and calendar-matched storm-free days).
Variables: Tmax (highest hourly report, >= 8 reports), T_pm, Td_pm (12-16 LT means), onshore wind at 12-16 LT
  = -(u, v) . n with n the station's seaward normal (Natural Earth, sigma 1 deg, isd_select.py), ws_pm, slp_pm.

(1) Regions: stations inside the 41 regions, season = the region's months on the map, exposure days with the storm in
    the region's sector.  Region value = mean over the 1-degree cells holding stations (as on the map) of the mean over
    their station-days; 90% interval from 1,000 bootstrap draws of storms.  Compared with the ERA5 value of the map (ex_map, Tmax).
(2) All coasts: stations within 50 km of the coast (isd_select.py), station's own season (months with >= 3% exposure
    days), every exposure day whatever the sector.  For each day the idealized outer wind of the nearest storm (as in
    geometry.py: tangential + 0.4 x inflow) gives an implied onshore component at the station.  Days are binned by it;
    station-days are averaged within 2.5-degree boxes and boxes count equally; 90% intervals from 1,000 bootstrap draws
    of years (all stations and storms of a year move together).  Also per-station least-squares slopes.

Usage: python isd_analyze.py SEL_DIR DAILY_DIR IBTRACS.csv gcg_regions.json gl_regions.csv OUT_DIR
"""
import json, os, sys
import numpy as np, pandas as pd

SEL, DLY, IBT, RJ, GLR, OUT = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
Y0, Y1 = 1980, 2021
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
VARS = ['Tmax', 'T_pm', 'Td_pm', 'on_pm', 'ws_pm', 'slp_pm']
rng = np.random.default_rng(1)
lines = []


def say(s=''):
    print(s, flush=True); lines.append(s)


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


st = pd.read_csv(os.path.join(SEL, 'isd_stations.csv'), dtype={'USAF': str, 'WBAN': str, 'region': str})
st['region'] = st.region.fillna('')
st = st[[os.path.exists(os.path.join(DLY, s + '.parquet')) for s in st.sid]].reset_index(drop=True)
days = pd.date_range(f'{Y0}-01-01', f'{Y1}-12-31', freq='D'); ND, NS = len(days), len(st)
slon, slat = st.lon.values, st.lat.values

# ---- nearest storm fix per station and day
tr = pd.read_csv(IBT, skiprows=[1], low_memory=False, usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'NATURE'])
for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND'):
    tr[c] = pd.to_numeric(tr[c], errors='coerce')
tr['time'] = pd.to_datetime(tr.ISO_TIME); tr['wind'] = tr.USA_WIND.fillna(tr.WMO_WIND)
tr = tr[tr.time.dt.hour.isin([0, 6, 12, 18]) & (tr.time.dt.minute == 0) & (tr.NATURE.str.strip() == 'TS') & (tr.wind >= 34)].dropna(subset=['LAT', 'LON'])
tr['day'] = (tr.time + pd.to_timedelta(tr.LON / 15.0, unit='h')).dt.normalize()
tr = tr[tr.day.dt.year.between(Y0, Y1)]
sids = pd.Index(tr.SID.unique()); tr['k'] = sids.get_indexer(tr.SID)
DM = np.full((ND, NS), np.inf, np.float32); FLO = np.full((ND, NS), np.nan, np.float32)
FLA = np.full((ND, NS), np.nan, np.float32); FK = np.full((ND, NS), -1, np.int32)
di = pd.Series(np.arange(ND), index=days)
for day, g in tr.groupby('day'):
    i = di[day]
    d = hav(slon[:, None], slat[:, None], g.LON.values[None, :], g.LAT.values[None, :])
    q = d.argmin(1); DM[i] = d[np.arange(NS), q]; FLO[i] = g.LON.values[q]; FLA[i] = g.LAT.values[q]; FK[i] = g.k.values[q]
say(f'{NS} stations with data, {len(sids)} storms, {ND} days')

# ---- regions
R = json.load(open(RJ)); G = pd.read_csv(GLR)
for k, v in R.items():
    j = ((G.lon - v['lon']).abs() + (G.lat - v['lat']).abs()).idxmin()
    v['months'] = G.loc[j, 'months']


def anomalies(X, code, seas_d):
    """X [day] -> matched anomalies (NaN outside the season or without data)."""
    ok = np.isfinite(X)
    if ok.sum() < 3000:
        return np.full(ND, np.nan)
    t = (days.year.values - 2000.5) / 10.0; w = 2 * np.pi * days.dayofyear.values / 365.25
    D = np.stack([np.ones_like(t), t] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)], 1)
    B = np.linalg.lstsq(D[ok], X[ok], rcond=None)[0]
    A = X - D @ B
    ref = seas_d & (code == -1) & ok
    yi = days.year.values - Y0
    for y in range(Y1 - Y0 + 1):
        m = yi == y; n = ref[m].sum()
        if n > 0:
            A[m] -= A[m][ref[m]].mean()
    doy = days.dayofyear.values
    S = np.bincount(doy[ref] - 1, A[ref], minlength=366); C = np.bincount(doy[ref] - 1, minlength=366).astype(float)
    Sw = sum(np.roll(S, r) for r in range(-15, 16)); Cw = sum(np.roll(C, r) for r in range(-15, 16))
    A = A - (Sw / np.maximum(Cw, 1))[doy - 1]
    A[~seas_d | ~ok] = np.nan
    return A


def implied(lon0, lat0, flo, fla, nx, ny, a=0.4, flip=False):
    r = np.stack([((flo - lon0 + 180) % 360 - 180) * np.cos(np.radians(lat0)), fla - lat0]); r = r / np.hypot(*r)
    t = np.stack([r[1], -r[0]]) if (lat0 >= 0) != flip else np.stack([-r[1], r[0]])
    V = t + a * r
    return -(V[0] * nx + V[1] * ny) / np.hypot(*V)


mon = days.month.values
reg_rows, coast_rows, slopes = [], [], []
fr, fc = os.path.join(OUT, 'isd_region_days.parquet'), os.path.join(OUT, 'isd_coast_days.parquet')
if os.environ.get('ISD_REUSE') and os.path.exists(fr) and os.path.exists(fc):
    RG, CS = pd.read_parquet(fr), pd.read_parquet(fc)
else:
    for s in range(NS):
        x = st.iloc[s]
        D = pd.read_parquet(os.path.join(DLY, x.sid + '.parquet'))
        if D.empty:
            continue
        D = D.set_index('day').reindex(days)
        D['on_pm'] = -(D.u_pm * x.nx + D.v_pm * x.ny)
        dm = DM[:, s]
        sec = ((bearing(x.lon, x.lat, FLO[:, s], FLA[:, s]) + 22.5) % 360 // 45)
        code = np.where(dm < 300, -2, np.where(dm <= 1300, sec, -1)).astype(int)
        # (1) region stations, the region's months and sector
        if x.region:
            v = R[x.region]; sd = np.array([v['months'][m - 1] == 'x' for m in mon])
            e = sd & (code == SECT.index(v['sector']))
            if e.sum() >= 10:
                rec = {'sid': x.sid, 'region': x.region, 'day': days[e], 'storm': FK[e, s]}
                for var in VARS:
                    rec[var] = anomalies(D[var].values.astype(float), code, sd)[e]
                reg_rows.append(pd.DataFrame(rec))
        # (2) coastal stations, own season, all sectors
        if x.in_coast_set:
            frac = np.array([(code[mon == m] >= 0).mean() for m in range(1, 13)])
            sd = (frac >= 0.03)[mon - 1]
            e = sd & (code >= 0)
            if e.sum() >= 50:
                rec = {'sid': x.sid, 'day': days[e], 'storm': FK[e, s], 'dist': dm[e], 'sector': code[e],
                       'imp': implied(x.lon, x.lat, FLO[e, s].astype(float), FLA[e, s].astype(float), x.nx, x.ny),
                       'imp0': implied(x.lon, x.lat, FLO[e, s].astype(float), FLA[e, s].astype(float), x.nx, x.ny, a=0.0),
                       'imp1': implied(x.lon, x.lat, FLO[e, s].astype(float), FLA[e, s].astype(float), x.nx, x.ny, a=1.0),
                       'imp_flip': implied(x.lon, x.lat, FLO[e, s].astype(float), FLA[e, s].astype(float), x.nx, x.ny, flip=True)}
                for var in VARS:
                    rec[var] = anomalies(D[var].values.astype(float), code, sd)[e]
                coast_rows.append(pd.DataFrame(rec))
        if s % 200 == 0:
            print(s, NS, flush=True)

    RG = pd.concat(reg_rows, ignore_index=True); CS = pd.concat(coast_rows, ignore_index=True)
    RG.to_parquet(os.path.join(OUT, 'isd_region_days.parquet'), index=False)
    CS.to_parquet(os.path.join(OUT, 'isd_coast_days.parquet'), index=False)

say(f'region station-days {len(RG)} ({RG.sid.nunique()} stations, {RG.region.nunique()} regions); '
    f'coastal station-days {len(CS)} ({CS.sid.nunique()} stations)')

# ---- (1) region table
nb = 1000; rows = []
for r, g in RG.groupby('region'):
    v = R[r]; out = dict(region=r, sign=v['sign'], sector=v['sector'], lat=v['lat'], lon=v['lon'], ex_map=v['ex_map'],
                         n_st=g.sid.nunique(), n_storm=g.storm.nunique(), n_days=len(g))
    g = g.merge(st[['sid', 'lon', 'lat']], on='sid')
    cell = pd.Index((np.round(g.lon) * 1000 + np.round(g.lat)).unique()); ci = cell.get_indexer(np.round(g.lon) * 1000 + np.round(g.lat))
    sto = pd.Index(g.storm.unique()); si = sto.get_indexer(g.storm); out['n_cells'] = len(cell)
    W = np.stack([np.bincount(rng.integers(0, len(sto), len(sto)), minlength=len(sto)) for _ in range(nb)]).astype(float)
    for var in VARS:
        ok = g[var].notna().values
        S = np.zeros((len(cell), len(sto))); C = np.zeros_like(S)
        np.add.at(S, (ci[ok], si[ok]), g[var].values[ok]); np.add.at(C, (ci[ok], si[ok]), 1)
        if C.sum() == 0:
            out[var] = np.nan; continue
        with np.errstate(invalid='ignore', divide='ignore'):
            out[var] = np.nanmean(np.where(C.sum(1) > 0, S.sum(1) / C.sum(1), np.nan))     # cells count equally, as on the map
            B = np.nanmean(np.where((C @ W.T) > 0, (S @ W.T) / (C @ W.T), np.nan), 0)         # storms resampled
        out[var + '_lo'], out[var + '_hi'] = np.nanpercentile(B, [5, 95])
    rows.append(out)
RT = pd.DataFrame(rows); RT['band'] = np.where(RT.lat.abs() < 30, 'low', 'mid')
geo = pd.read_csv(os.path.join(os.path.dirname(RJ), '..', 'mech', 'geometry_map_regions.csv'))
RT = RT.merge(geo[['region', 'imp04']], on='region', how='left')
RT.to_csv(os.path.join(OUT, 'isd_regions.csv'), index=False, float_format='%.4g')
say('\n== (1) regions: storm in the region sector, region season; stations vs ERA5 map (Tmax)')
say(f'{len(RT)} regions with >= 10 exposure days at a station; usable (>= 20 storms): {(RT.n_storm >= 20).sum()}')
U = RT[RT.n_storm >= 20]
for b, g in U.groupby('band'):
    sg = (np.sign(g.Tmax) == np.where(g.sign == 'warm', 1, -1)); sp = (np.sign(g.T_pm) == np.where(g.sign == 'warm', 1, -1))
    say(f'  {b}: {len(g)} regions; station sign as on the map: Tmax {int(sg.sum())}/{int(g.Tmax.notna().sum())}, T_pm {int(sp.sum())}/{len(g)}; corr(station Tmax, map) '
        f'{g.Tmax.corr(g.ex_map):+.2f}; mean station/map ratio (warm) {g[g.sign == "warm"].Tmax.mean() / g[g.sign == "warm"].ex_map.mean():.2f}')
    for k, h in g.groupby('sign'):
        say(f'    {k}: n {len(h)}, Tmax {h.Tmax.mean():+.2f} K (map {h.ex_map.mean():+.2f}), T_pm {h.T_pm.mean():+.2f}, '
            f'Td_pm {h.Td_pm.mean():+.2f}, onshore {h.on_pm.mean():+.2f} m/s (negative in {int((h.on_pm < 0).sum())}), '
            f'slp {h.slp_pm.mean():+.2f} hPa')
say(f'  corr(station onshore change, implied outer wind): <30 {U[U.band == "low"].on_pm.corr(U[U.band == "low"].imp04):+.2f}, '
    f'all {U.on_pm.corr(U.imp04):+.2f}; corr(station Tmax, implied): <30 {U[U.band == "low"].Tmax.corr(U[U.band == "low"].imp04):+.2f}')
say(RT.sort_values(['sign', 'region']).round(2)[['region', 'sign', 'sector', 'lat', 'n_st', 'n_cells', 'n_storm', 'ex_map', 'Tmax', 'Tmax_lo',
                                                  'Tmax_hi', 'T_pm', 'Td_pm', 'on_pm', 'imp04']].to_string(index=False))

# ---- (2) all coasts, binned by the implied outer wind
CS = CS.merge(st[['sid', 'lat', 'lon', 'region']], on='sid')
CS['band'] = np.where(CS.lat.abs() < 30, 'low', 'mid')
CS['box'] = (np.floor(CS.lon / 2.5)).astype(int) * 1000 + np.floor(CS.lat / 2.5).astype(int)
CS['year'] = CS.day.dt.year - Y0
EDGES = np.array([-1.001, -0.6, -0.2, 0.2, 0.6, 1.001]); CEN = ['-1 to -0.6', '-0.6 to -0.2', '-0.2 to 0.2', '0.2 to 0.6', '0.6 to 1']
CS['bin'] = np.digitize(CS.imp, EDGES) - 1


def binned(df, var, nb=1000):
    """box-weighted bin means with a year-block bootstrap. Returns est [bin], lo, hi."""
    d = df.dropna(subset=[var, 'imp'])
    bx = pd.Index(d.box.unique()); bi = bx.get_indexer(d.box)
    S = np.zeros((len(bx), 5, Y1 - Y0 + 1)); C = np.zeros_like(S)
    np.add.at(S, (bi, d.bin.values, d.year.values), d[var].values); np.add.at(C, (bi, d.bin.values, d.year.values), 1)

    def est(w):
        s = (S * w).sum(2); c = (C * w).sum(2)
        with np.errstate(invalid='ignore', divide='ignore'):
            m = np.where(c >= 5, s / c, np.nan)
        return np.nanmean(m, 0)
    e = est(np.ones(Y1 - Y0 + 1))
    W = np.stack([np.bincount(rng.integers(0, Y1 - Y0 + 1, Y1 - Y0 + 1), minlength=Y1 - Y0 + 1) for _ in range(nb)])
    B = np.stack([est(w) for w in W])
    return e, np.nanpercentile(B, 5, 0), np.nanpercentile(B, 95, 0), len(bx)


say('\n== (2) all coasts within 50 km: every exposure day, binned by the implied onshore component of the outer wind')
say('   (negative = the idealized outer circulation blows off the land); box-weighted means, 90% year-bootstrap intervals')
BR = []
for name, sub in (('lat < 30', CS[CS.band == 'low']), ('lat 30-45', CS[CS.band == 'mid']),
                  ('lat < 30, outside the 41 regions', CS[(CS.band == 'low') & (CS.region == '')]),
                  ('lat < 30, Southern Hemisphere', CS[(CS.band == 'low') & (CS.lat < 0)]),
                  ('lat < 30, 300-700 km', CS[(CS.band == 'low') & (CS.dist <= 700)]),
                  ('lat < 30, 700-1300 km', CS[(CS.band == 'low') & (CS.dist > 700)])):
    say(f'-- {name}: {sub.sid.nunique()} stations, {sub.box.nunique()} boxes, {len(sub)} station-days')
    for var in ('Tmax', 'T_pm', 'Td_pm', 'on_pm'):
        e, lo, hi, nbx = binned(sub, var)
        say(f'   {var:6s} ' + '  '.join(f'{CEN[i]}: {e[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]' for i in range(5)))
        for i in range(5):
            BR.append(dict(subset=name, var=var, bin=CEN[i], est=e[i], lo=lo[i], hi=hi[i]))
pd.DataFrame(BR).to_csv(os.path.join(OUT, 'isd_coast_bins.csv'), index=False, float_format='%.4g')

# coast orientation beyond storm direction: slope of the anomaly on the implied component after removing the mean of
# each storm-bearing sector (8 sectors; within a sector the implied component differs between stations only through
# how their coasts face), station-day least squares, 90% year-bootstrap interval
def fe_slope(df, var, x='imp', nb=1000, within=True):
    d = df.dropna(subset=[var, x]).copy()
    d['w'] = 1.0 / d.groupby('box')[var].transform('count')          # every 2.5-degree box counts equally
    g = d.sector if within else pd.Series(0, index=d.index)
    for c in (var, x):
        d[c + '_dm'] = d[c] - (d.w * d[c]).groupby(g).transform('sum') / d.w.groupby(g).transform('sum')
    yr = d.year.values; X = d[x + '_dm'].values; Y = d[var + '_dm'].values; w = d.w.values
    Sxy = np.bincount(yr, w * X * Y, minlength=Y1 - Y0 + 1); Sxx = np.bincount(yr, w * X * X, minlength=Y1 - Y0 + 1)
    e = Sxy.sum() / Sxx.sum()
    W = np.stack([np.bincount(rng.integers(0, Y1 - Y0 + 1, Y1 - Y0 + 1), minlength=Y1 - Y0 + 1) for _ in range(nb)])
    B = (W @ Sxy) / (W @ Sxx)
    return e, *np.percentile(B, [5, 95])


say('\n== slope on the implied component (K or m/s per unit; -1 = straight off the land, +1 = straight onto it); boxes weighted equally')
FE = []
for name, sub in (('lat < 30', CS[CS.band == 'low']), ('lat 30-45', CS[CS.band == 'mid']),
                  ('lat < 30, Southern Hemisphere', CS[(CS.band == 'low') & (CS.lat < 0)])):
    for var in ('Tmax', 'Td_pm', 'on_pm'):
        o = dict(subset=name, var=var)
        for x, lab in (('imp', 'a=0.4'), ('imp0', 'a=0'), ('imp1', 'a=1'), ('imp_flip', 'wrong rotation')):
            o['all ' + lab] = fe_slope(sub, var, x, nb=1, within=False)[0]
        o['within-sector a=0.4'], o['ws_lo'], o['ws_hi'] = fe_slope(sub, var)
        o['within-sector wrong rotation'] = fe_slope(sub, var, 'imp_flip', nb=1)[0]
        FE.append(o)
        say(f"  {name:30s} {var:6s} " + ', '.join(f'{k} {v:+.2f}' for k, v in o.items() if k not in ('subset', 'var')))
pd.DataFrame(FE).to_csv(os.path.join(OUT, 'isd_coast_slopes.csv'), index=False, float_format='%.4g')

# per-station slopes of Tmax and onshore on the implied component
for sid, g in CS.groupby('sid'):
    o = {'sid': sid}
    for var in ('Tmax', 'T_pm', 'Td_pm', 'on_pm'):
        h = g.dropna(subset=[var, 'imp'])
        o[var + '_slope'] = np.polyfit(h.imp, h[var], 1)[0] if len(h) >= 50 and h.imp.std() > 0.2 else np.nan
    o['n'] = len(g); slopes.append(o)
SL = pd.DataFrame(slopes).merge(st[['sid', 'name', 'CTRY', 'lat', 'lon', 'region', 'coast_km']], on='sid')
SL.to_csv(os.path.join(OUT, 'isd_station_slopes.csv'), index=False, float_format='%.4g')
say('\n== per-station slopes (K or m/s per unit of the implied onshore component; negative Tmax slope = warmer when the outer wind blows off the land)')
for b, g in SL.groupby(np.where(SL.lat.abs() < 30, 'lat < 30', 'lat 30-45')):
    t = g.dropna(subset=['Tmax_slope']); o = g.dropna(subset=['on_pm_slope'])
    say(f'  {b}: Tmax slope median {t.Tmax_slope.median():+.2f} K, negative at {int((t.Tmax_slope < 0).sum())}/{len(t)} stations; '
        f'onshore slope median {o.on_pm_slope.median():+.2f} m/s, positive at {int((o.on_pm_slope > 0).sum())}/{len(o)}; '
        f'Td slope median {g.Td_pm_slope.median():+.2f} K')
open(os.path.join(OUT, 'isd_result.txt'), 'w').write('\n'.join(lines) + '\n')
