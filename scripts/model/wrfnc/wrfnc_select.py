"""Multi-coast WRF storm-removal runs (NC upgrade, 2026-11-01): pick the cases, write the tracks and the WRF domains.

Cases come from the GraphCast all-exposure-day set (A2, paper/data/ggc/all_gc_cases.csv), so WRF and GraphCast can be
compared on the same days.  Rule (written before any WRF run, applied with the observed map anomaly only, never with a
model result): per coast, the exposure days of three different storms with the largest map anomaly in the coast's own
direction (warm coasts: largest; cool coasts: most negative), storm at least 600 km from the region centre on the case
day, and not one of the two-storm days flagged in geometry_flagged_cases.csv.  Guangdong Megi and Mangkhut (the clean
pilot pairs) are rerun with the WIDEL edit so that all coasts use the same counterfactual.

Usage: python -I wrfnc_select.py PAPER_DATA_DIR IBTRACS_CSV OUT_DIR
Writes OUT_DIR/{wrfnc_cases.csv, wrfnc_tracks.csv, wrfnc_others.csv, cells_<coast>.csv, domains.json, select_log.txt}.
"""
import json, os, sys
import numpy as np, pandas as pd
from pyproj import Proj

DATA, IBT, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
COASTS = ['W17', 'W3', 'C16', 'W22']          # Mexico Pacific, northern Mozambique, US Southeast (cool), US Northeast
OPTIONAL = ['W5']                             # south-east India, run only if the queue allows
N_PER, DMIN_KM = 3, 600.0
START_H, RUN_H = 54, 72                       # as the Guangdong runs: start 54 h before the case time, 72-h runs
RE = 6370.0                                   # WRF sphere
D01 = dict(nx=391, ny=311, dx=12.0)            # as Guangdong
D02_DX, RATIO, MARGIN02 = 3.0, 4, 150.0       # d02 = region box + 150 km on every side
EDGE_MIN = 700.0                              # every storm position >= 700 km inside d01 (grow d01 if not)
GD_STN = None

log = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)


def gc_dist(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * RE * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


reg = json.load(open(os.path.join(DATA, 'ggc', 'gcg_regions.json')))
cases = pd.read_csv(os.path.join(DATA, 'ggc', 'all_gc_cases.csv'))
flag = set(pd.read_csv(os.path.join(DATA, 'mech', 'geometry_flagged_cases.csv')).case)
rows = []
for coast in COASTS + OPTIONAL:
    s = cases[(cases.region == coast) & (cases.dist >= DMIN_KM) & (~cases.case.isin(flag))].copy()
    warm = reg[coast]['sign'] == 'warm'
    s = s[s.anom > 0] if warm else s[s.anom < 0]
    s = s.sort_values('anom', ascending=not warm).drop_duplicates('sid')
    pick = s.head(N_PER)
    say(f'{coast} ({reg[coast]["sign"]}, storms {reg[coast]["sector"]}): {len(s)} eligible storms; picked',
        ', '.join(f'{r.case} {r["name"]} {r.V} anom {r.anom:+.2f} dist {r.dist:.0f}' for _, r in pick.iterrows()))
    for _, r in pick.iterrows():
        rows.append(dict(case=int(r.case), coast=coast, sign=reg[coast]['sign'], optional=coast in OPTIONAL,
                         V=r.V, sid=r.sid, name=r['name'], dist=r.dist, anom=r.anom, edit='widel'))
# Guangdong clean pilot pairs, rerun with WIDEL (old case numbers 6 and 16; region = the Guangdong stations)
for c, V, sid, name in ((6, '2016-09-27T06', '2016265N10150', 'MEGI'), (16, '2018-09-15T06', '2018250N12170', 'MANGKHUT')):
    rows.append(dict(case=c, coast='GD', sign='warm', optional=False, V=V, sid=sid, name=name, dist=np.nan, anom=np.nan,
                     edit='widel'))
C = pd.DataFrame(rows)
C['start'] = [(pd.Timestamp(v) - pd.Timedelta(hours=START_H)).strftime('%Y-%m-%d_%H') for v in C.V]
C['end'] = [(pd.Timestamp(v) - pd.Timedelta(hours=START_H - RUN_H)).strftime('%Y-%m-%d_%H') for v in C.V]

# tracks: 3-hourly positions of the case storm (main track, interpolated) and of every other storm in the window
ib = pd.read_csv(IBT, skiprows=[1], usecols=['SID', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'TRACK_TYPE', 'USA_WIND', 'WMO_WIND'],
                 keep_default_na=False, low_memory=False)
ib = ib[ib.TRACK_TYPE == 'main']
ib['time'] = pd.to_datetime(ib.ISO_TIME); ib['lat'] = ib.LAT.astype(float); ib['lon'] = ib.LON.astype(float)
ib['wind'] = pd.to_numeric(ib.USA_WIND.replace(' ', np.nan), errors='coerce').fillna(
    pd.to_numeric(ib.WMO_WIND.replace(' ', np.nan), errors='coerce'))
tr, ot = [], []
for _, c in C.iterrows():
    t0 = pd.Timestamp(c.start.replace('_', ' ') + ':00') - pd.Timedelta('6h')
    t1 = pd.Timestamp(c.end.replace('_', ' ') + ':00') + pd.Timedelta('6h')
    g = ib[(ib.SID == c.sid) & (ib.time >= t0 - pd.Timedelta('12h')) & (ib.time <= t1 + pd.Timedelta('12h'))].set_index('time')
    g = g[~g.index.duplicated()]
    grid = pd.date_range(t0, t1, freq='3h')
    lon = np.unwrap(np.radians(g.lon.values)); g = g.assign(lonu=np.degrees(lon))
    gi = g[['lat', 'lonu', 'wind']].reindex(g.index.union(grid)).interpolate('time', limit_area='inside').reindex(grid)
    for t, r in gi.iterrows():
        if np.isfinite(r.lat):
            tr.append(dict(case=c.case, time=t.strftime('%Y-%m-%dT%H'), lon=((r.lonu + 180) % 360) - 180, lat=r.lat, wind=r.wind))
    o = ib[(ib.SID != c.sid) & (ib.time >= t0) & (ib.time <= t1)]
    for _, r in o.iterrows():
        ot.append(dict(case=c.case, sid=r.SID, name=r.NAME, time=r.time.strftime('%Y-%m-%dT%H'), lon=r.lon, lat=r.lat, wind=r.wind))
T = pd.DataFrame(tr); O = pd.DataFrame(ot)

# region cells (WRF "stations"); Guangdong uses its stations file on DelftBlue (inputs/gd_stations.csv)
for coast in C.coast.unique():
    if coast == 'GD':
        continue
    cl = np.array(reg[coast]['cells'], float)
    pd.DataFrame({'lon': ((cl[:, 0] + 180) % 360) - 180, 'lat': cl[:, 1]},
                 index=[f'c{k:03d}' for k in range(len(cl))]).rename_axis('cell').to_csv(os.path.join(OUT, f'cells_{coast}.csv'))


def make_proj(lat0, lon0):
    if abs(lat0) < 28:
        return dict(map_proj='mercator', truelat1=float(np.clip(lat0, -25, 25)), stand_lon=lon0), \
            Proj(proj='merc', lat_ts=float(np.clip(lat0, -25, 25)), lon_0=lon0, R=RE * 1000)
    t1, t2 = (30.0, 60.0) if lat0 > 0 else (-30.0, -60.0)
    return dict(map_proj='lambert', truelat1=t1, truelat2=t2, stand_lon=lon0), \
        Proj(proj='lcc', lat_1=t1, lat_2=t2, lat_0=lat0, lon_0=lon0, R=RE * 1000)


# domains: d01 centred between the region and the storm positions; check every storm position is >= 700 km inside d01
dom = {}
for coast in [x for x in C.coast.unique() if x != 'GD']:
    cl = np.array(reg[coast]['cells'], float); cl[:, 0] = ((cl[:, 0] + 180) % 360) - 180
    cs = C[C.coast == coast].case
    P = T[T.case.isin(cs)]
    lon_r, lat_r = cl[:, 0].mean(), cl[:, 1].mean()
    plon = np.degrees(np.unwrap(np.radians(np.r_[lon_r, P.lon.values])))
    best = None
    for nx, ny in ((391, 311), (431, 311), (391, 351), (431, 351), (471, 351), (471, 391)):   # grow d01 only if needed
      if best is not None and best[6] >= EDGE_MIN:
          break
      best = None
      for w in np.linspace(0.2, 0.6, 9):            # weight towards the storms
        lon0 = (1 - w) * lon_r + w * plon[1:].mean(); lat0 = (1 - w) * lat_r + w * P.lat.mean()
        info, pr = make_proj(lat0, lon0)
        x0, y0 = pr(lon0, lat0)
        hx, hy = nx * D01['dx'] * 500, ny * D01['dx'] * 500
        xs, ys = pr(P.lon.values, P.lat.values); xr_, yr_ = pr(cl[:, 0], cl[:, 1])
        edge_s = np.min(np.minimum(hx - np.abs(xs - x0), hy - np.abs(ys - y0))) / 1000
        edge_r = np.min(np.minimum(hx - np.abs(xr_ - x0), hy - np.abs(yr_ - y0))) / 1000
        score = min(edge_s, edge_r)
        if best is None or score > best[0]:
            best = (score, w, lon0, lat0, info, pr, edge_s, edge_r, nx, ny)
    score, w, lon0, lat0, info, pr, edge_s, edge_r, nx1, ny1 = best
    x0, y0 = pr(lon0, lat0)
    xr_, yr_ = pr(cl[:, 0], cl[:, 1])
    m = MARGIN02 * 1000 + 50e3                    # cells are 1-degree centres: +-0.5 degree ~ 50 km
    x_lo, x_hi, y_lo, y_hi = xr_.min() - m, xr_.max() + m, yr_.min() - m, yr_.max() + m
    # d01 lower-left mass point (1-based index 1) is at x0 - (nx-1)/2 dx
    dx1 = D01['dx'] * 1000
    xll, yll = x0 - (nx1 - 1) / 2 * dx1, y0 - (ny1 - 1) / 2 * dx1
    ips = int(np.floor((x_lo - xll) / dx1)) + 1; jps = int(np.floor((y_lo - yll) / dx1)) + 1
    nxp = int(np.ceil((x_hi - (xll + (ips - 1) * dx1)) / dx1)); nyp = int(np.ceil((y_hi - (yll + (jps - 1) * dx1)) / dx1))
    e_we2, e_sn2 = nxp * RATIO + 1, nyp * RATIO + 1
    dom[coast] = dict(ref_lat=round(float(lat0), 3), ref_lon=round(float(lon0), 3), **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()},
                      e_we=[nx1, e_we2], e_sn=[ny1, e_sn2], dx=[12000, 3000], i_parent_start=[1, ips], j_parent_start=[1, jps],
                      storm_edge_km=round(float(edge_s)), region_edge_km=round(float(edge_r)), d02_km=[round((e_we2 - 1) * 3), round((e_sn2 - 1) * 3)])
    say(f'{coast}: d01 {nx1}x{ny1} centre {lat0:.2f}N {lon0:.2f}E ({info["map_proj"]}), storms >= {edge_s:.0f} km and region >= {edge_r:.0f} km'
        f' inside d01; d02 {e_we2}x{e_sn2} ({(e_we2 - 1) * 3}x{(e_sn2 - 1) * 3} km) at i,j {ips},{jps}')

C.to_csv(os.path.join(OUT, 'wrfnc_cases.csv'), index=False)
T.to_csv(os.path.join(OUT, 'wrfnc_tracks.csv'), index=False)
O.to_csv(os.path.join(OUT, 'wrfnc_others.csv'), index=False)
json.dump(dom, open(os.path.join(OUT, 'domains.json'), 'w'), indent=1)
say(C[['case', 'coast', 'name', 'V', 'start', 'end', 'dist', 'anom', 'optional']].to_string(index=False))
say('track rows', len(T), 'other-storm rows', len(O))
open(os.path.join(OUT, 'select_log.txt'), 'w').write('\n'.join(log) + '\n')
