"""Export compact WRF results for the paper figures.

Pairs (control, experiment) per case, where both runs exist: (ctl, wide), (ctl, nullw), (ctl_s30, wide_s30).
Usage: TCH=/scratch/$USER/tcheat python wrf_export.py
Output ($TCH/export/):
  wrx_series.csv  hourly 19-station means per run, V-30 h .. V+18 h: T2, Q2, SWDOWN, PBLH, CLDMAX (column max CLDFRA),
                  RAIN1 (mm/h), T850, Q850 (at the station columns), ONSHORE (5 coastal stations, wind from 100-260 deg)
  wrx_maps.npz    at V (14 LT): d02 dT2 = control - experiment on a regular 0.1-deg grid, dU10/dV10 (earth-relative)
                  on 0.3 deg, land mask on 0.1 deg; int8 + scales
  wrx_coast.csv   dT2 at V and mean over 12-16 LT vs distance from the coast (10-km bins, d02 land cells)
  wrx_xsec.npz    d01 at V along the great circle Guangdong (113.5E 23N) -> control storm centre, -300..1500 km every
                  30 km, 0.25..8 km every 250 m above sea level: d(theta), d(w), d(qv), cloud fraction of both runs
"""
import glob, os
import numpy as np, pandas as pd, netCDF4
from scipy import ndimage
from scipy.interpolate import griddata

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
OUT = os.path.join(TCH, 'export')
STN = pd.read_csv(os.path.join(TCH, 'inputs', 'gd_stations.csv'), index_col=0)
COAST = [59493099999, 59501099999, 59663099999, 59673099999, 59658099999]
CASES = pd.read_csv(os.path.join(TCH, 'inputs', 'gc_cases.csv')).set_index('case')
TRK = pd.read_csv(os.path.join(TCH, 'inputs', 'wrf_tracks.csv'))
GD = (113.5, 23.0)
PAIRS = [('ctl', 'wide'), ('ctl', 'nullw'), ('ctl_s30', 'wide_s30')]
ZLEV = np.arange(250.0, 8001.0, 250.0)
SDIST = np.arange(-300.0, 1501.0, 30.0)


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gc_line(lon0, lat0, lon1, lat1, s_km):
    """Points at distance s along the great circle from (lon0, lat0) towards (lon1, lat1); negative s = other side."""
    p0, p1 = np.radians([lon0, lat0]), np.radians([lon1, lat1])
    y = np.sin(p1[0] - p0[0]) * np.cos(p1[1])
    x = np.cos(p0[1]) * np.sin(p1[1]) - np.sin(p0[1]) * np.cos(p1[1]) * np.cos(p1[0] - p0[0])
    b = np.arctan2(y, x); d = np.asarray(s_km) / 6371.0
    la = np.arcsin(np.sin(p0[1]) * np.cos(d) + np.cos(p0[1]) * np.sin(d) * np.cos(b))
    lo = p0[0] + np.arctan2(np.sin(b) * np.sin(d) * np.cos(p0[1]), np.cos(d) - np.sin(p0[1]) * np.sin(la))
    return np.degrees(lo), np.degrees(la)


def frames(run_dir, dom):
    """Yield (time, netCDF dataset, index) for every output time of a run."""
    for f in sorted(glob.glob(os.path.join(run_dir, f'wrfout_{dom}_*'))):
        nc = netCDF4.Dataset(f)
        for k, t in enumerate(nc['Times'][:]):
            yield pd.Timestamp(b''.join(t).decode().replace('_', ' ')), nc, k
        nc.close()


def nearest_idx(la, lo, lats, lons):
    return np.array([np.unravel_index(np.argmin((la - a) ** 2 + ((lo - b) * np.cos(np.radians(a))) ** 2), la.shape)
                     for a, b in zip(lats, lons)]).T


def interp_p(var, p, target):
    """Log-p interpolation of var[k, n] (k levels, p decreasing with k) to pressure `target` (Pa) for each column n."""
    lp = np.log(p); lt = np.log(target); out = np.full(var.shape[1], np.nan)
    for n in range(var.shape[1]):
        out[n] = np.interp(-lt, -lp[:, n], var[:, n])
    return out


def earth_wind(nc, k, u, v):
    ca, sa = nc['COSALPHA'][k], nc['SINALPHA'][k]
    return u * ca - v * sa, v * ca + u * sa


def series(run_dir, V):
    rows, prev_rain, jj = [], None, None
    for t, nc, k in frames(run_dir, 'd02'):
        if jj is None:
            la, lo = nc['XLAT'][0], nc['XLONG'][0]
            jj, ii = nearest_idx(la, lo, STN.lat.values, STN.lon.values)
            ic = np.isin(STN.index.values, COAST)
        rain = (nc['RAINC'][k] + nc['RAINNC'][k])[jj, ii]
        r1 = np.nan if prev_rain is None else float(np.mean(rain - prev_rain)); prev_rain = rain
        if not (V - pd.Timedelta('30h') <= t <= V + pd.Timedelta('18h')):
            continue
        has3d = all(x in nc.variables for x in ('P', 'PB', 'T', 'QVAPOR'))
        if has3d:
            p = (nc['P'][k] + nc['PB'][k])[:, jj, ii]
            th = (nc['T'][k][:, jj, ii] + 300.0); tk = th * (p / 1e5) ** 0.2857
            qv = nc['QVAPOR'][k][:, jj, ii]
        ue, ve = earth_wind(nc, k, nc['U10'][k], nc['V10'][k])
        wd = (np.degrees(np.arctan2(-ue[jj, ii], -ve[jj, ii])) + 360) % 360
        ons = ((wd >= 100) & (wd <= 260))[ic]
        rows.append(dict(time=t, h=(t - V) / pd.Timedelta('1h'), T2=float(nc['T2'][k][jj, ii].mean()),
                         Q2=float(1000 * nc['Q2'][k][jj, ii].mean()), SWDOWN=float(nc['SWDOWN'][k][jj, ii].mean()),
                         PBLH=float(nc['PBLH'][k][jj, ii].mean()),
                         CLDMAX=float(nc['CLDFRA'][k][:, jj, ii].max(0).mean()) if 'CLDFRA' in nc.variables else np.nan,
                         RAIN1=r1, T850=float(np.nanmean(interp_p(tk, p, 85000.0))) if has3d else np.nan,
                         Q850=float(1000 * np.nanmean(interp_p(qv, p, 85000.0))) if has3d else np.nan, ONSHORE=float(ons.mean())))
    return pd.DataFrame(rows)


def at_time(run_dir, dom, V):
    for t, nc, k in frames(run_dir, dom):
        if t == V:
            return nc.filepath(), k
    return None, None


def d02_fields(path, k):
    nc = netCDF4.Dataset(path)
    la, lo = nc['XLAT'][k], nc['XLONG'][k]
    ue, ve = earth_wind(nc, k, nc['U10'][k], nc['V10'][k])
    f = dict(T2=nc['T2'][k].astype(float), U=ue.astype(float), V=ve.astype(float), LAND=nc['LANDMASK'][k].astype(float))
    nc.close()
    return la, lo, f


def storm_centre(path, k, guess):
    nc = netCDF4.Dataset(path)
    la, lo = nc['XLAT'][k], nc['XLONG'][k]
    slp = nc['PSFC'][k] * np.exp(nc['HGT'][k] / (29.3 * nc['T2'][k]))
    d = hav(lo, la, *guess); m = np.where(d <= 600, slp, np.inf); q = np.unravel_index(np.argmin(m), m.shape)
    nc.close()
    return float(lo[q]), float(la[q])


def xsec(path, k, plon, plat):
    nc = netCDF4.Dataset(path)
    la, lo = nc['XLAT'][k], nc['XLONG'][k]
    jj, ii = nearest_idx(la, lo, plat, plon)
    z = ((nc['PH'][k] + nc['PHB'][k]) / 9.81)[:, jj, ii]; z = 0.5 * (z[1:] + z[:-1])
    th = nc['T'][k][:, jj, ii] + 300.0
    w = nc['W'][k][:, jj, ii]; w = 0.5 * (w[1:] + w[:-1])
    qv = 1000 * nc['QVAPOR'][k][:, jj, ii]
    cf = nc['CLDFRA'][k][:, jj, ii] if 'CLDFRA' in nc.variables else np.zeros_like(qv)
    hgt = nc['HGT'][k][jj, ii]
    nc.close()
    out = {}
    for nm, v in (('th', th), ('w', w), ('qv', qv), ('cf', cf)):
        o = np.full((len(ZLEV), len(plon)), np.nan, np.float32)
        for n in range(len(plon)):
            o[:, n] = np.interp(ZLEV, z[:, n], v[:, n], left=np.nan, right=np.nan)
        out[nm] = o
    out['hgt'] = np.asarray(hgt, np.float32)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ser, coast, maps, xs = [], [], {'_keys': [], '_scales': []}, {'_keys': [], '_scales': []}
    def put(store, key, x, cnt):
        sc = float(np.nanmax(np.abs(x))) / 127 or 1.0
        store['m%03d' % len(store['_keys'])] = np.where(np.isfinite(x), np.round(x / sc), -128).astype(np.int8)
        store['_keys'].append(key); store['_scales'].append(sc)
    grid = None
    for d in sorted(glob.glob(os.path.join(TCH, 'wrf', 'case_*'))):
        c = int(d.split('_')[-1]); V = pd.Timestamp(CASES.loc[c, 'V'])
        for run in sorted({r for p in PAIRS for r in p}):
            if glob.glob(os.path.join(d, run, 'wrfout_d02_*')):
                s = series(os.path.join(d, run), V); s.insert(0, 'run', run); s.insert(0, 'case', c); ser.append(s)
                print('series', c, run, len(s), flush=True)
        for cr, xr_ in PAIRS:
            pc, kc = at_time(os.path.join(d, cr), 'd02', V); px, kx = at_time(os.path.join(d, xr_), 'd02', V)
            if pc is None or px is None:
                continue
            la, lo, fc = d02_fields(pc, kc); _, _, fx = d02_fields(px, kx)
            if grid is None:
                g_lo = np.arange(np.ceil(lo.min() * 10) / 10, lo.max(), 0.1); g_la = np.arange(np.ceil(la.min() * 10) / 10, la.max(), 0.1)
                G = np.meshgrid(g_lo, g_la); g3 = (g_lo[1::3], g_la[1::3]); G3 = np.meshgrid(*g3)
                land = griddata((lo.ravel(), la.ravel()), fc['LAND'].ravel(), tuple(G), method='nearest')
                maps['lon01'], maps['lat01'] = g_lo.astype(np.float32), g_la.astype(np.float32)
                maps['lon03'], maps['lat03'] = g3[0].astype(np.float32), g3[1].astype(np.float32)
                maps['land01'] = np.packbits(land > 0.5); maps['land_shape'] = np.array(land.shape)
                grid = True
            dT = fc['T2'] - fx['T2']
            put(maps, f'{c}|{cr}-{xr_}|dT2', griddata((lo.ravel(), la.ravel()), dT.ravel(), tuple(G), method='linear'), None)
            for nm in ('U', 'V'):
                put(maps, f'{c}|{cr}-{xr_}|d{nm}10', griddata((lo.ravel(), la.ravel()), (fc[nm] - fx[nm]).ravel(), tuple(G3), method='linear'), None)
            # distance from the coast (d02 land cells, 3 km grid)
            dist = ndimage.distance_transform_edt(fc['LAND'] > 0.5) * 3.0
            b = np.floor(dist / 10.0).astype(int)
            aft = []
            for tt in pd.date_range(V - pd.Timedelta('2h'), V + pd.Timedelta('2h'), freq='h'):
                p1, k1 = at_time(os.path.join(d, cr), 'd02', tt); p2, k2 = at_time(os.path.join(d, xr_), 'd02', tt)
                if p1 and p2:
                    aft.append(d02_fields(p1, k1)[2]['T2'] - d02_fields(p2, k2)[2]['T2'])
            dA = np.mean(aft, 0) if aft else dT * np.nan
            for bb in range(0, 20):
                m = (fc['LAND'] > 0.5) & (b == bb)
                if m.sum() >= 20:
                    coast.append(dict(case=c, pair=f'{cr}-{xr_}', km=10 * bb + 5, n=int(m.sum()), dT2_V=float(dT[m].mean()), dT2_12_16=float(dA[m].mean())))
            # cross-section on d01 at V
            pc1, kc1 = at_time(os.path.join(d, cr), 'd01', V); px1, kx1 = at_time(os.path.join(d, xr_), 'd01', V)
            if pc1 and px1:
                tr = TRK[(TRK.case == c)].copy(); tr['t'] = pd.to_datetime(tr.time)
                g = tr.iloc[np.argmin(np.abs((tr.t - V).dt.total_seconds()))]
                cl = storm_centre(pc1, kc1, (g.lon, g.lat))
                plon, plat = gc_line(GD[0], GD[1], cl[0], cl[1], SDIST)
                a1, a2 = xsec(pc1, kc1, plon, plat), xsec(px1, kx1, plon, plat)
                pre = f'{c}|{cr}-{xr_}|'
                for nm in ('th', 'w', 'qv'):
                    put(xs, pre + 'd' + nm, a1[nm] - a2[nm], None)
                put(xs, pre + 'cf_' + cr, a1['cf'], None); put(xs, pre + 'cf_' + xr_, a2['cf'], None)
                xs[f'hgt_{c}'] = a1['hgt']; xs[f'centre_{c}_{cr}'] = np.array(cl, np.float32)
                xs[f'dist_storm_{c}'] = np.float32(hav(*GD, *cl))
            print('pair', c, cr, xr_, flush=True)
    pd.concat(ser).to_csv(os.path.join(OUT, 'wrx_series.csv'), index=False, float_format='%.4g')
    pd.DataFrame(coast).to_csv(os.path.join(OUT, 'wrx_coast.csv'), index=False, float_format='%.4g')
    for store, name, extra in ((maps, 'wrx_maps.npz', {}), (xs, 'wrx_xsec.npz', dict(zlev=ZLEV.astype(np.float32), sdist=SDIST.astype(np.float32)))):
        k, sc = store.pop('_keys'), store.pop('_scales')
        np.savez(os.path.join(OUT, name), keys=np.array(k), scales=np.array(sc, np.float32), **extra, **store)
        print(name, len(k), 'arrays')


if __name__ == '__main__':
    main()
