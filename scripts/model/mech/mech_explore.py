"""A4 exploration of the GraphCast mechanism outputs (mech_gc.py, mech_win.py), run locally on the copies in mech_out/gc/.
Exploratory: tables, groupings and composites, no verdict (plan plan A4 (internal), section 4).

1. Region fingerprint table (both models) with latitude, ERA5 dry/moist class, ISD station count and A2 map warming.
2. Case-level relations of the 2-m effect with the mechanism terms, within regions (region means removed), pooled and by
   latitude group.
3. Hierarchical clustering (Ward) of the usable warm regions on the standardised fingerprint, 2-4 groups.
4. Storm-relative composites: each case's window (1 degree) is put in km around the region centre, rotated so that the
   storm (CTL sea-level-pressure minimum 300-1,800 km from the centre, checked against the best-track bearing) lies on the
   +x axis, and mirrored in the Southern Hemisphere so that cyclones turn the same way; then averaged per group of regions
   (storms weighted equally within a region, regions equally).

Usage: python mech_explore.py GC_DIR GGC_DATA_DIR MECH_OUT_DIR OUT_DIR [GC_TRACKS_CSV [LAND_SHP]]
  GC_DIR       folder with mech_cases_*.csv, mech_regions_*.csv, win_*.npz
  GGC_DATA_DIR folder with all_gc_cases.csv, all_oper_regions.csv
  MECH_OUT_DIR folder with mech_era5.csv, isd_counts.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import griddata

GC, GG, MO, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
lines = []
def say(s=''):
    print(s, flush=True); lines.append(str(s))

G = pd.read_csv(os.path.join(GG, 'all_gc_cases.csv'))
A2 = pd.read_csv(os.path.join(GG, 'all_oper_regions.csv')).set_index('region')
E5 = pd.read_csv(os.path.join(MO, 'mech_era5.csv')).set_index('region')
ISD = pd.read_csv(os.path.join(MO, 'isd_counts.csv')).set_index('region') if os.path.exists(os.path.join(MO, 'isd_counts.csv')) else None
FP = ['t2m', 't850', 'q850', 'w500', 'w850', 'onshore', 'along', 'tp6', 'msl']
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)

# ---------------------------------------------------------------- 1. fingerprint table
tabs = {}
for model in ('oper', 'small'):
    T = pd.read_csv(os.path.join(GC, f'mech_regions_{model}.csv')).set_index('region')
    T['abslat'] = T.lat.abs(); T['band'] = np.where(T.abslat < 30, 'low', 'mid')
    T['q850_gkg'] = T.q850; T['tp6_mm'] = T.tp6; T['ratio_2m_850'] = T.t2m / T.t850   # case files store q850 in g/kg and tp6 in mm
    T['era5'] = [E5.loc[r, 'class'] if r in E5.index else '' for r in T.index]
    T['ex_map'] = [A2.loc[r, 'ex_map'] if r in A2.index else np.nan for r in T.index]
    T['isd'] = [ISD.loc[r, 'isd_2016_2021'] if ISD is not None and r in ISD.index else np.nan for r in T.index]
    tabs[model] = T
    cols = ['kind', 'lat', 'band', 'era5', 'n_cases', 'n_storms', 'ex_map', 't2m', 't2m_lo', 't2m_hi', 't850', 'q850_gkg', 'w500',
            'w850', 'onshore', 'along', 'tp6_mm', 'msl', 'ctl_onshore', 'isd']
    T[cols].to_csv(os.path.join(OUT, f'fingerprint_{model}.csv'), float_format='%.4g')
    say(f'================ 1. fingerprint, model {model}')
    say(T[cols].round(3).to_string())

# ---------------------------------------------------------------- 2. case-level relations
for model in ('oper', 'small'):
    C = pd.read_csv(os.path.join(GC, f'mech_cases_{model}.csv'))
    T = tabs[model]
    C = C[C.region.isin(T.index)].merge(G[['case', 'dist', 'brg', 'wind', 'anom']], on='case', how='left')
    C['band'] = C.region.map(T.band); C['kind'] = C.region.map(T.kind)
    say(''); say(f'================ 2. case-level relations, model {model} ({len(C)} cases in usable regions)')
    for (kind, band), g in C.groupby(['kind', 'band']):
        if len(g) < 10:
            continue
        dm = g[['t2m'] + FP[1:] + ['dist', 'wind', 'anom']].sub(g.groupby('region')[['t2m'] + FP[1:] + ['dist', 'wind', 'anom']].transform('mean'))
        r = {v: dm[v].corr(dm.t2m) for v in FP[1:] + ['dist', 'wind', 'anom'] if dm[v].notna().sum() > 10}
        say(f'{kind} {band} ({len(g)} cases, {g.region.nunique()} regions), within-region correlation with the 2-m effect: '
            + ', '.join(f'{k} {v:+.2f}' for k, v in r.items()))
        say(f'    2-m effect by lead: ' + ', '.join(f'{L} h {g[f"t2m_L{L}"].mean():+.2f}' for L in (24, 36, 48))
            + f'; by distance: ' + ', '.join(f'{a}-{b} km {g[(g.dist >= a) & (g.dist < b)].t2m.mean():+.2f} (n {((g.dist >= a) & (g.dist < b)).sum()})'
                                            for a, b in ((550, 800), (800, 1000), (1000, 1300))))
    C.to_csv(os.path.join(OUT, f'cases_{model}.csv'), index=False, float_format='%.5g')

# ---------------------------------------------------------------- 3. clustering of warm regions
for model in ('oper', 'small'):
    T = tabs[model]; W = T[T.kind == 'warm'].copy()
    use = [v for v in ('t2m', 't850', 'q850', 'w500', 'onshore', 'tp6') if W[v].notna().all()]
    Z = (W[use] - W[use].mean()) / W[use].std()
    L = linkage(Z.values, 'ward')
    say(''); say(f'================ 3. Ward clustering of {len(W)} warm regions, model {model}, on {use}')
    for k in (2, 3, 4):
        lab = fcluster(L, k, 'maxclust'); W[f'k{k}'] = lab
        for c in sorted(set(lab)):
            g = W[lab == c]
            say(f'  k={k} group {c}: {" ".join(g.index)} | |lat| {g.abslat.mean():.0f}; ' +
                ', '.join(f'{v} {g[v].mean():+.3g}' for v in use) + f'; ERA5 ' + ','.join(g.era5))
    W.to_csv(os.path.join(OUT, f'clusters_{model}.csv'), float_format='%.4g')

# ---------------------------------------------------------------- 4. storm-relative composites
# Storm position at the case time from the best-track file (gc_tracks.csv, argument 5) when given, otherwise the CTL
# sea-level-pressure minimum 300-1,800 km from the region centre.  A land fraction (Natural Earth 1:50m, argument 6) is
# composited in the same frame, so that the mean position of the coast relative to the storm can be drawn.
TRK = pd.read_csv(sys.argv[5]) if len(sys.argv) > 5 else None
# Optional: cases left out of the composites (env MECH_EXCLUDE = csv with a 'case' column, e.g. geometry_flagged_cases.csv from
# geometry.py: days on which the removed storm lay two or more sectors from the region's direction)
EXCL = set(pd.read_csv(os.environ['MECH_EXCLUDE']).case.astype(int)) if os.environ.get('MECH_EXCLUDE') else set()
LAND = None
if len(sys.argv) > 6:
    import shapefile
    from shapely.geometry import shape
    from rasterio import features
    from rasterio.transform import from_origin
    geoms = [shape(g.__geo_interface__) for g in shapefile.Reader(sys.argv[6]).shapes()]
    LAND = features.rasterize(((g, 1) for g in geoms), out_shape=(720, 1440), transform=from_origin(-180, 90, 0.25, 0.25),
                              fill=0, dtype='uint8').astype(float)
def land_at(lon, lat):
    i = np.clip(((90 - lat) / 0.25).astype(int), 0, 719); j = (((lon + 180) % 360) / 0.25).astype(int) % 1440
    return LAND[i, j]
XG = np.arange(-1500, 2501, 50.0); YG = np.arange(-2000, 2001, 50.0)
XX, YY = np.meshgrid(XG, YG)
for model in ('oper', 'small'):
    f = os.path.join(GC, f'win_{model}.npz')
    if not os.path.exists(f):
        continue
    Zf = np.load(f, allow_pickle=False); V = list(Zf['vars']); T = tabs[model]
    iu, iv, im = V.index('u10'), V.index('v10'), V.index('msl')
    VO = V + (['land'] if LAND is not None else [])
    rows, fields = [], []
    for i, case in enumerate(Zf['case']):
        r = str(Zf['region'][i])
        if r not in T.index or int(case) in EXCL:
            continue
        la, lo = Zf['lat'][i], Zf['lon'][i]; la0, lo0 = float(Zf['lat_c'][i]), float(Zf['lon_c'][i])
        LO, LA = np.meshgrid(lo, la)
        # wlon in the case files is longitude relative to the region centre (windows run from about -20 to +20); wlat is absolute
        rel = np.nanmax(np.abs(lo)) <= 20.6
        dlon = LO if rel else (LO - lo0 + 180) % 360 - 180
        kx = 111.2 * np.cos(np.radians(la0))
        x = dlon * kx; y = (LA - la0) * 111.2
        src = 'track'
        tr = TRK[(TRK.case == case) & (TRK.time == str(Zf['V'][i]))] if TRK is not None else []
        if len(tr):
            xs = ((float(tr.lon.iloc[0]) - lo0 + 180) % 360 - 180) * kx; ys = (float(tr.lat.iloc[0]) - la0) * 111.2
        else:
            src = 'msl'; dist = np.hypot(x, y); msl = Zf['ctl'][i, im].astype(float)
            m = (dist > 300) & (dist < 1800)
            if not m.any():
                continue
            k = np.nanargmin(np.where(m, msl, np.nan)); xs, ys = x.flat[k], y.flat[k]
        th = np.arctan2(ys, xs)
        c, s_ = np.cos(th), np.sin(th)
        xr, yr = x * c + y * s_, -x * s_ + y * c
        sh = la0 < 0
        if sh:
            yr = -yr
        d = Zf['d'][i].astype(float).copy()
        ur, vr = d[iu] * c + d[iv] * s_, -d[iu] * s_ + d[iv] * c
        if sh:
            vr = -vr
        d[iu], d[iv] = ur, vr
        if LAND is not None:
            d = np.concatenate([d, land_at(lo0 + dlon, LA)[None]])
        pts = np.column_stack([xr.ravel(), yr.ravel()])
        fields.append(np.stack([griddata(pts, d[jj].ravel(), (XX, YY), method='linear') for jj in range(d.shape[0])]))
        rows.append(dict(case=int(case), region=r, sid=str(Zf['sid'][i]), kind=T.loc[r, 'kind'], band=T.loc[r, 'band'],
                         storm_km=float(np.hypot(xs, ys)), src=src))
    R = pd.DataFrame(rows); F = np.stack(fields)
    say(''); say(f'================ 4. storm-relative composites, model {model}: {len(R)} cases (storm from the best track in '
                 f'{int((R.src == "track").sum())}); median storm distance from the region centre {R.storm_km.median():.0f} km'
                 + (f'; {len(EXCL)} cases left out ({os.path.basename(os.environ["MECH_EXCLUDE"])})' if EXCL else ''))
    comp = {}
    for nm, sel in (('warm_low', (R.kind == 'warm') & (R.band == 'low')), ('warm_mid', (R.kind == 'warm') & (R.band == 'mid')),
                    ('cool', R.kind == 'cool'), ('cool_low', (R.kind == 'cool') & (R.band == 'low'))):
        sel = sel.values
        if sel.sum() < 3:
            continue
        sub = R[sel]; Fs = F[sel]
        per_reg = []
        for r in sub.region.unique():
            rr = (sub.region == r).values
            per_storm = [np.nanmean(Fs[rr][(sub[rr].sid == s).values], axis=0) for s in sub[rr].sid.unique()]
            per_reg.append(np.nanmean(per_storm, axis=0))
        PR = np.stack(per_reg); cm = np.nanmean(PR, axis=0); comp[nm] = cm
        comp[nm + '_agree'] = np.nanmean(np.sign(PR) == np.sign(cm)[None], axis=0)
        comp[nm + '_nreg'] = np.sum(np.isfinite(PR[:, 0]), axis=0)
        comp[nm + '_n'] = np.array([sel.sum(), sub.sid.nunique(), sub.region.nunique()])
        comp[nm + '_dist'] = np.array([sub.storm_km.median()])
        jt = V.index('t2m'); o = (np.abs(XX) < 150) & (np.abs(YY) < 150)
        say(f'  {nm}: {sel.sum()} cases, {sub.sid.nunique()} storms, {sub.region.nunique()} regions, median storm distance '
            f'{sub.storm_km.median():.0f} km; within 150 km of the region centre: ' +
            ', '.join(f'{v} {np.nanmean(cm[VO.index(v)][o]):+.3g}' for v in VO))
    np.savez_compressed(os.path.join(OUT, f'stormrel_{model}.npz'), x=XG, y=YG, vars=np.array(VO), **comp)
    R.to_csv(os.path.join(OUT, f'stormrel_cases_{model}.csv'), index=False, float_format='%.4g')
open(os.path.join(OUT, 'mech_explore_result.txt'), 'w').write('\n'.join(lines) + '\n')
