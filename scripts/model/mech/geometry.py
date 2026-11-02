"""A4 exploratory: does the storm's outer circulation blow off or onto the land at each coast?

Idealized outer-circulation wind at the region centre: the tangential unit vector of a cyclone (anticlockwise in the
Northern Hemisphere, clockwise in the Southern) plus a * the unit vector pointing to the storm (inflow), a = 0, 0.4, 1.
Implied onshore = -(V . n) / |V|, with n the seaward coast normal (coast_normals.csv).  Negative = the outer
circulation blows off the land.  No model output enters this number: it uses only where the storm is and how the
coast faces.  It is compared afterwards with the GraphCast changes (onshore wind, 2 m effect) and with the map change.

  map level : storm bearing = centre of the region's sector (all 41 regions of gcg_regions.json)
  case level: storm bearing = best-track position at V (gc_tracks.csv), for the A2 cases of usable regions

Also flags cases whose storm bearing at V lies two or more sectors away from the region's sector: on such days the
exposure code came from another, nearer storm (often a weakening one), while the experiment removed the one at V.

These groupings (latitude band, sign) were chosen after the first results were seen; nothing here is a test.

Usage: python geometry.py MECH_OUT_DIR GGC_DATA_DIR OUT_DIR
  MECH_OUT_DIR: holds coast_normals.csv and gc/ (mech_cases_oper.csv, mech_regions_oper.csv, gc_tracks.csv)
  GGC_DATA_DIR: holds gcg_regions.json, all_gc_cases.csv, all_{oper,small}_cases.csv, all_{oper,small}_regions.csv"""
import json, os, sys
import numpy as np, pandas as pd

M, G, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
A = (0.0, 0.4, 1.0)
lines = []


def say(s=''):
    print(s); lines.append(s)


def implied(r, lat, n):
    """r: unit vector region -> storm (east, north); n: seaward normal. Returns sea_dot and implied onshore for A."""
    t = np.array([r[1], -r[0]]) if lat >= 0 else np.array([-r[1], r[0]])
    out = []
    for a in A:
        V = t + a * r
        out.append(-(V @ n) / np.hypot(*V))
    return float(r @ n), out


nr = pd.read_csv(os.path.join(M, 'coast_normals.csv')).set_index('region')
R = json.load(open(os.path.join(G, 'gcg_regions.json')))

# ---- map level
rows = []
for k, v in R.items():
    b = np.radians(SECT.index(v['sector']) * 45.0)
    sd, im = implied(np.array([np.sin(b), np.cos(b)]), v['lat'], np.array([nr.loc[k, 'nx'], nr.loc[k, 'ny']]))
    rows.append(dict(region=k, sign=v['sign'], sector=v['sector'], lat=v['lat'], ex_map=v['ex_map'], sea_dot=sd,
                     imp0=im[0], imp04=im[1], imp1=im[2], seaward=nr.loc[k, 'seaward_deg']))
D = pd.DataFrame(rows); D['band'] = np.where(D.lat.abs() < 30, 'low', 'mid')
D.to_csv(os.path.join(OUT, 'geometry_map_regions.csv'), index=False, float_format='%.4g')
say('== map level: storm at the centre of the region sector, 41 regions')
for (s, b), g in D.groupby(['sign', 'band']):
    say(f'{s:4s} {b:3s} {len(g):2d} regions: implied offshore (a = 0.4) {int((g.imp04 < 0).sum()):2d}; a = 0: '
        f'{int((g.imp0 < 0).sum()):2d}; a = 1: {int((g.imp1 < 0).sum()):2d}; storm on the seaward side {int((g.sea_dot > 0).sum()):2d}')
lo = D[D.band == 'low']
say(f'corr(implied onshore, map change): all 41 {D.imp04.corr(D.ex_map):+.2f}; |lat| < 30 ({len(lo)}) '
    f'{lo.imp04.corr(lo.ex_map):+.2f}; >= 30 ({len(D) - len(lo)}) {D[D.band == "mid"].imp04.corr(D[D.band == "mid"].ex_map):+.2f}')
say('exceptions (sign of the map not matched by the implied wind):')
for _, x in D.sort_values('imp04').iterrows():
    if (x.sign == 'warm') != (x.imp04 < 0):
        say(f'  {x.region:4s} {x.sign} {x.sector:2s} lat {x.lat:+6.1f} map {x.ex_map:+.2f} K implied onshore {x.imp04:+.2f}')

# ---- case level
C = pd.read_csv(os.path.join(M, 'gc', 'mech_cases_oper.csv')); T = pd.read_csv(os.path.join(M, 'gc', 'gc_tracks.csv'))
Rg = pd.read_csv(os.path.join(M, 'gc', 'mech_regions_oper.csv')).set_index('region')
C = C[C.region.isin(Rg.index)].merge(T, left_on=['case', 'V'], right_on=['case', 'time'], how='left')
miss = int(C.lon.isna().sum())
rows = []
for _, c in C.dropna(subset=['lon']).iterrows():
    lo0, la0 = nr.loc[c.region, 'lon'], nr.loc[c.region, 'lat']
    r = np.array([((c.lon - lo0 + 180) % 360 - 180) * np.cos(np.radians(la0)), c.lat - la0]); r /= np.hypot(*r)
    sd, im = implied(r, la0, np.array([nr.loc[c.region, 'nx'], nr.loc[c.region, 'ny']]))
    rows.append(dict(case=c.case, sea_dot=sd, imp0=im[0], imp04=im[1], imp1=im[2]))
X = C.merge(pd.DataFrame(rows), on='case'); X['band'] = np.where(X.region.map(Rg.lat).abs() < 30, 'low', 'mid')
gc = pd.read_csv(os.path.join(G, 'all_gc_cases.csv'))[['case', 'sector', 'brg', 'name']]
gc['bs'] = ((gc.brg + 22.5) % 360 // 45).astype(int); gc['s'] = gc.sector.map(SECT.index)
gc['sect_off'] = np.minimum((gc.bs - gc.s) % 8, (gc.s - gc.bs) % 8)
X = X.merge(gc[['case', 'brg', 'name', 'sect_off']], on='case', how='left')
X.to_csv(os.path.join(OUT, 'geometry_cases_oper.csv'), index=False, float_format='%.4g')
say(f'\n== case level: storm at its best-track position at V, {len(X)} cases of usable regions ({miss} without a track fix)')
for (k, b), g in X.groupby(['kind', 'band']):
    say(f'{k:4s} {b:3s} {len(g):3d} cases: implied offshore a = 0 {100 * (g.imp0 < 0).mean():3.0f}%, a = 0.4 '
        f'{100 * (g.imp04 < 0).mean():3.0f}%, a = 1 {100 * (g.imp1 < 0).mean():3.0f}% | corr with GraphCast onshore '
        f'{g.imp04.corr(g.onshore):+.2f}, with 2 m effect {g.imp04.corr(g.t2m):+.2f}')
say(f'all cases: corr(implied, GraphCast onshore) {X.imp04.corr(X.onshore):+.2f}, corr(implied, 2 m effect) {X.imp04.corr(X.t2m):+.2f}')
Q = X.groupby('region').agg(t2m=('t2m', 'mean'), imp=('imp04', 'mean'), on=('onshore', 'mean'), kind=('kind', 'first'),
                            band=('band', 'first'), n=('case', 'size'))
Q.to_csv(os.path.join(OUT, 'geometry_regions_gc.csv'), float_format='%.4g')
say(f'region level ({len(Q)} usable regions): corr(implied, GraphCast onshore) {Q.imp.corr(Q.on):+.2f}; '
    f'corr(implied, 2 m effect) {Q.imp.corr(Q.t2m):+.2f}')
for kb, g in Q.groupby(['kind', 'band']):
    say(f'  {kb[0]} {kb[1]}: {len(g)} regions, implied offshore {int((g.imp < 0).sum())}, mean 2 m effect {g.t2m.mean():+.2f} K')

# ---- cases where the removed storm is not the one that set the direction
say('\n== cases with the storm at V two or more sectors from the region sector')
bad = X[X.sect_off >= 2]
bad[['case', 'region', 'kind', 'V', 'name', 'brg', 'sect_off']].to_csv(os.path.join(OUT, 'geometry_flagged_cases.csv'), index=False, float_format='%.4g')
for _, x in bad.iterrows():
    say(f'  case {x.case} {x.region} {x.kind} {x.V} {x["name"]}: bearing {x.brg:.0f} deg vs sector '
        f'{R[x.region]["sector"]}; 2 m effect {x.t2m:+.2f} K, implied onshore {x.imp04:+.2f}')
for model in ('oper', 'small'):
    rr = pd.read_csv(os.path.join(G, f'all_{model}_regions.csv')); cc = pd.read_csv(os.path.join(G, f'all_{model}_cases.csv'))
    cool = rr[(rr.kind == 'cool') & rr.usable].region
    cc = cc[(cc.kind == 'cool') & cc.region.isin(cool)]
    pooled = lambda d: d.groupby(['region', 'sid']).d_free.mean().groupby('region').mean().mean()
    mp = rr[rr.region.isin(cool)].ex_map.mean(); mp8 = rr[rr.region.isin(cool) & (rr.region != 'C8')].ex_map.mean()
    a, b = pooled(cc), pooled(cc[cc.region != 'C8'])
    say(f'cool regions, {model}: pooled effect {a:+.3f} K ({len(cool)} regions, {100 * a / mp:.0f}% of map {mp:+.3f} K); '
        f'without C8 {b:+.3f} K ({len(cool) - 1} regions, {100 * b / mp8:.0f}% of {mp8:+.3f} K)')
open(os.path.join(OUT, 'geometry_result.txt'), 'w').write('\n'.join(lines) + '\n')
