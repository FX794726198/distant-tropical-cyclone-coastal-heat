"""Multi-coast WRF runs: region values, storm effect (ctl - edited) and storm checks.  Exploratory, no thresholds.

Per case and run (ctl, widel, nullw, and Guangdong's pilot ctl):
  - region value = mean over the region's 1-degree cells of the mean over d02 land points inside the cell (all points
    if the cell has no land); Guangdong: the 19 stations (nearest d02 point), as in the pilot;
  - at the case time V (the GraphCast case time, 6-hourly closest to 14 LT) and at the hour closest to 14 LT:
    T2, Q2, 10-m wind and the onshore component on the region's seaward normal (paper/data/mech/coast_normals.csv),
    T and specific humidity at 850 hPa (log-p interpolation), downward short-wave at the surface, PBL height,
    vertical velocity at 500 hPa (w500, m/s, positive up) and the column-maximum cloud fraction (CLDMAX);
  - Tmax: the 08-18 LT maximum of hourly T2 at each d02 point (each station for Guangdong), then the region mean, as
    the map and the pilot (Tmax_series: maximum of the hourly region-mean series, for reference);
  - rain24_V: region-mean rain in the 24 h before V (accumulated RAINNC + RAINC at V minus at V-24 h);
  - hourly region-mean series of T2, Q2, onshore wind, SWDOWN, PBLH and accumulated rain for the whole run;
  - storm check in d01 at V-24 h (Vm24) and V, in every run: lowest sea-level pressure within 600 km of the control
    storm centre (trackmin) and its deficit against the 800-1000 km ring (deficit); and the lowest sea-level pressure
    anywhere in d01 at least 25 points from the edge (lowest_lon/lat) with its deficit against its own 800-1000 km ring
    (lowest_def).  This is the lowest point, not a closed-low test: a new storm in an edited run shows up as a
    deep lowest point far from the control track; check the maps before calling it a storm;
  - maps at V, thinned, for figures: d02 T2, Q2, 10-m wind (every 3rd point); d01 SLP, T850, 850-hPa wind (every 2nd).
v2 (11-01 11:10Z) after the dry run on 393 and 6: per-point Tmax, rain over 24 h, d01 maps, L14 values also when
L14 = V, storm-check labels relative to V and numeric, w500 and CLDMAX.
v3 (11-01 19:00Z): daytime means (08-18 LT, hourly) of the process variables, less noisy than one time step for the
process figure: T850, q850, w500, CLDMAX (3-D, every hour in the window) and T2, Q2, onshore wind, SWDOWN, PBLH (from the
hourly series), all as <var>_day; rain_day = rain accumulated over the window.  The 3-D variables of Guangdong are now
means over 1-degree boxes around the 19 stations (single 3-km points made w500 noisy); its T2, Tmax and the surface
series stay at the station points, as in the pilot.
Usage (DelftBlue, as a job: wrfnc_analyse.slurm): TCH=/scratch/$USER/tcheat python wrfnc_analyse.py [CASES...]  -> $ROOT/out/
"""
import glob, os, sys
import numpy as np, pandas as pd, netCDF4

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
ROOT = os.environ.get('WRFNC', os.path.join(TCH, 'wrfnc'))
OUT = os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
CASES = pd.read_csv(os.path.join(ROOT, 'inputs', 'wrfnc_cases.csv')).set_index('case')
TR = pd.read_csv(os.path.join(ROOT, 'inputs', 'wrfnc_tracks.csv')); TR['time'] = pd.to_datetime(TR.time)
NORM = pd.read_csv(os.path.join(ROOT, 'inputs', 'coast_normals.csv')).set_index('region')
GD_NORMAL = (0.0, -1.0)            # Guangdong coast faces south (pilot onshore = southerly wind)
RUNS = ('ctl', 'widel', 'nullw', 'wide')


def gc_dist(lon, lat, lon0, lat0):
    p1, p2 = np.radians(lat), np.radians(lat0); dl = np.radians(lon0 - lon)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def times_of(nc):
    return [pd.Timestamp(b''.join(x).decode().replace('_', ' ')) for x in nc['Times'][:]]


def cell_weights(coast, la, lo, land):
    """[ncell, ny, nx] boolean masks (Guangdong: nearest points of the 19 stations)."""
    if coast == 'GD':
        st = pd.read_csv(os.path.join(ROOT, 'inputs', 'cells_GD.csv'), index_col=0)
        M = np.zeros((len(st),) + la.shape, bool)
        for k, (a, b) in enumerate(zip(st.lat, st.lon)):
            j, i = np.unravel_index(np.argmin((la - a) ** 2 + ((lo - b) * np.cos(np.radians(a))) ** 2), la.shape); M[k, j, i] = True
        return M
    return box_weights(os.path.join(ROOT, 'inputs', f'cells_{coast}.csv'), la, lo, land)


def box_weights(path, la, lo, land):
    """[ncell, ny, nx] masks of the 1-degree boxes centred on the rows of path (land points if any)."""
    cl = pd.read_csv(path, index_col=0)
    M = []
    for a, b in zip(cl.lat, cl.lon):
        box = (np.abs(la - a) <= 0.5) & (np.abs(((lo - b + 180) % 360) - 180) <= 0.5)
        M.append(box & (land > 0.5) if (box & (land > 0.5)).any() else box)
    return np.array(M)


def interp_p(field, p, target):
    """field, p [nz, ny, nx] -> value at pressure target (Pa), log-p; NaN below the surface."""
    lp = np.log(p); lt = np.log(target)
    k = np.argmax(lp < lt, axis=0)                      # first level above the target (pressure decreasing upward)
    k = np.clip(k, 1, p.shape[0] - 1)
    j, i = np.indices(k.shape)
    w = (lt - lp[k - 1, j, i]) / (lp[k, j, i] - lp[k - 1, j, i])
    v = field[k - 1, j, i] + w * (field[k, j, i] - field[k - 1, j, i])
    v[p[0] < target] = np.nan
    return v


def slp_of(nc, k):
    ps = nc['PSFC'][k]; h = nc['HGT'][0]; t2 = nc['T2'][k]
    return ps * (1 - 0.0065 * h / (t2 + 0.0065 * h)) ** -5.257 / 100


def run_values(run, coast, V, lt14, nrm):
    fs = sorted(glob.glob(os.path.join(run, 'wrfout_d02_*')))
    if not fs:
        return None
    res, ser, M, TX = {}, [], None, None
    acc = {}                                                       # 3-D process variables, every hour of 08-18 LT
    off = pd.Timedelta(hours=float(np.round(nrm[2] / 15.0)))      # local time offset
    dayV = (V + off).normalize(); w0, w1 = dayV + pd.Timedelta('8h') - off, dayV + pd.Timedelta('18h') - off   # 08-18 LT in UTC
    for f in fs:
        nc = netCDF4.Dataset(f)
        if M is None:
            la, lo = nc['XLAT'][0], nc['XLONG'][0]; land = nc['LANDMASK'][0]
            M = cell_weights(coast, la, lo, land); cnt = M.sum((1, 2))
            MB = box_weights(os.path.join(ROOT, 'inputs', 'cells_GD.csv'), la, lo, land) if coast == 'GD' else M
        def reg(a):
            return float(np.nanmean([np.nanmean(a[m]) for m in M]))
        def regb(a):                                               # 3-D fields: 1-degree boxes also for Guangdong
            return float(np.nanmean([np.nanmean(a[m]) for m in MB]))
        tt = times_of(nc)
        for k, t in enumerate(tt):
            u, v = nc['U10'][k], nc['V10'][k]
            on = -(u * nrm[0] + v * nrm[1])
            t2 = np.asarray(nc['T2'][k], float)
            row = dict(time=t, T2=reg(t2), Q2=reg(1000 * nc['Q2'][k]), onshore=reg(on), SWDOWN=reg(nc['SWDOWN'][k]),
                       PBLH=reg(nc['PBLH'][k]), rainacc=reg(nc['RAINNC'][k] + nc['RAINC'][k]))
            inwin = w0 <= t <= w1
            if inwin:
                TX = t2.copy() if TX is None else np.fmax(TX, t2)
            if t in (V, lt14) or inwin:
                p = nc['P'][k] + nc['PB'][k]; th = nc['T'][k] + 300.0; T = th * (p / 1e5) ** 0.2854
                q = nc['QVAPOR'][k] / (1 + nc['QVAPOR'][k]) * 1000
                ua = 0.5 * (nc['U'][k][:, :, :-1] + nc['U'][k][:, :, 1:]); va = 0.5 * (nc['V'][k][:, :-1, :] + nc['V'][k][:, 1:, :])
                extra = dict(T850=regb(interp_p(T, p, 85000.)), q850=regb(interp_p(q, p, 85000.)),
                             u850=regb(interp_p(ua, p, 85000.)), v850=regb(interp_p(va, p, 85000.)),
                             SW=reg(nc['SWDOWN'][k]), PBLH=reg(nc['PBLH'][k]), U10=reg(u), V10=reg(v),
                             rainacc=reg(nc['RAINNC'][k] + nc['RAINC'][k]))
                wz = 0.5 * (nc['W'][k][:-1] + nc['W'][k][1:]); extra['w500'] = regb(interp_p(wz, p, 50000.))
                if 'CLDFRA' in nc.variables:
                    extra['CLDMAX'] = regb(nc['CLDFRA'][k].max(0))
                if inwin:
                    for a in ('T850', 'q850', 'w500', 'CLDMAX'):
                        if a in extra:
                            acc.setdefault(a, []).append(extra[a])
                for tag in [x for x, tt in (('V', V), ('L14', lt14)) if t == tt]:     # both tags when L14 = V
                    for a, b in extra.items():
                        res[f'{a}_{tag}'] = b
                if t == V:
                    res['_maps'] = dict(lat=la[::3, ::3], lon=lo[::3, ::3], T2=nc['T2'][k][::3, ::3], Q2=1000 * nc['Q2'][k][::3, ::3],
                                        U10=u[::3, ::3], V10=v[::3, ::3])
            ser.append(row)
        nc.close()
    S = pd.DataFrame(ser).set_index('time')
    for key, t in (('V', V), ('L14', lt14)):
        if t in S.index:
            for c in ('T2', 'Q2', 'onshore'):
                res[f'{c}_{key}'] = float(S.loc[t, c])
    m = (S.index >= w0) & (S.index <= w1)
    res['Tmax'] = reg(TX) if (TX is not None and m.sum() >= 9) else np.nan     # needs most of the 11 hours
    res['Tmax_series'] = float(S.T2[m].max()) if m.any() else np.nan
    res['n_tmax_hours'] = int(m.sum())
    for a, v_ in acc.items():                                      # daytime means (most of the 11 hours needed)
        res[f'{a}_day'] = float(np.mean(v_)) if len(v_) >= 9 else np.nan
    for a in ('T2', 'Q2', 'onshore', 'SWDOWN', 'PBLH'):
        res[f'{a}_day'] = float(S[a][m].mean()) if m.sum() >= 9 else np.nan
    res['rain_day'] = float(np.clip(np.diff(S.rainacc[m].values), 0, None).sum()) if m.sum() >= 9 else np.nan
    vm = V - pd.Timedelta('24h')
    res['rain24_V'] = float(S.rainacc[V] - S.rainacc[vm]) if (V in S.index and vm in S.index) else np.nan
    return res, S


def storm_check(run, c, times, V):
    out, mp = {}, None
    g = TR[TR.case == c].set_index('time').sort_index()
    for t in times:
        lab = 'V' if t == V else f'Vm{int((V - t) / pd.Timedelta("1h"))}'

        s = t.strftime('%Y-%m-%d_%H:%M:%S')
        for f in sorted(glob.glob(os.path.join(run, 'wrfout_d01_*'))):
            nc = netCDF4.Dataset(f); tt = [b''.join(x).decode() for x in nc['Times'][:]]
            if s in tt:
                k = tt.index(s); slp = slp_of(nc, k); la, lo = nc['XLAT'][0], nc['XLONG'][0]
                if t == V:
                    p = nc['P'][k] + nc['PB'][k]; T = (nc['T'][k] + 300.0) * (p / 1e5) ** 0.2854
                    ua = 0.5 * (nc['U'][k][:, :, :-1] + nc['U'][k][:, :, 1:]); va = 0.5 * (nc['V'][k][:, :-1, :] + nc['V'][k][:, 1:, :])
                    mp = dict(lat=la[::2, ::2], lon=lo[::2, ::2], SLP=slp[::2, ::2], T850=interp_p(T, p, 85000.)[::2, ::2],
                              u850=interp_p(ua, p, 85000.)[::2, ::2], v850=interp_p(va, p, 85000.)[::2, ::2])
                nc.close()
                gi = g.reindex(g.index.union([t])).interpolate('time', limit_area='inside')
                if t in gi.index and np.isfinite(gi.loc[t, 'lon']):
                    d = gc_dist(lo, la, gi.loc[t, 'lon'], gi.loc[t, 'lat'])
                    inner, ring = d < 600, (d > 800) & (d < 1000)
                    mn = float(slp[inner].min()) if inner.any() else np.nan
                    out[f'trackmin_{lab}'] = mn
                    out[f'deficit_{lab}'] = float(slp[ring].mean()) - mn if ring.any() else np.nan
                # lowest sea-level pressure anywhere at least 25 points (300 km) from the edge, and its deficit
                e = np.zeros_like(slp, bool); e[25:-25, 25:-25] = True
                j, i = np.unravel_index(np.argmin(np.where(e, slp, np.inf)), slp.shape)
                d0 = gc_dist(lo, la, lo[j, i], la[j, i]); rg = (d0 > 800) & (d0 < 1000)
                out[f'lowest_lon_{lab}'] = float(lo[j, i]); out[f'lowest_lat_{lab}'] = float(la[j, i])
                out[f'lowest_def_{lab}'] = float(slp[rg].mean() - slp[j, i]) if rg.any() else np.nan
                break
            nc.close()
    return out, mp


cases = [int(x) for x in sys.argv[1:]] or list(CASES.index)
rows, series, maps = [], [], {}
for c in cases:
    cs = CASES.loc[c]; coast = cs.coast; V = pd.Timestamp(cs.V)
    D = os.path.join(ROOT, f'case_{c:03d}')
    if coast == 'GD':
        nrm = (GD_NORMAL[0], GD_NORMAL[1], 113.5)
    else:
        n = NORM.loc[coast]; nrm = (float(n.nx), float(n.ny), float(n.lon))
    off = pd.Timedelta(hours=float(np.round(nrm[2] / 15.0)))
    lt14 = (V + off).normalize() + pd.Timedelta('14h') - off          # 14 LT of the case's local day (within 3 h of V)
    for r in RUNS:
        run = os.path.join(D, r)
        if not os.path.isdir(run):
            continue
        rv = run_values(run, coast, V, lt14, nrm)
        if rv is None:
            continue
        res, S = rv
        mp = res.pop('_maps', None)
        if mp is not None:
            maps[f'{c}_{r}'] = mp
        sc, mp1 = storm_check(run, c, [V - pd.Timedelta('24h'), V], V)
        res.update(sc)
        if mp1 is not None:
            maps[f'{c}_{r}_d01'] = mp1
        rows.append(dict(case=c, coast=coast, name=cs['name'], V=str(V), L14=str(lt14), run=r, **res))
        S = S.reset_index(); S.insert(0, 'run', r); S.insert(0, 'case', c); series.append(S)
        print(c, coast, r, f"T2_V {res.get('T2_V', np.nan):.2f} Tmax {res.get('Tmax', np.nan):.2f}", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, 'wrfnc_runs.csv'), index=False)
pd.concat(series).to_csv(os.path.join(OUT, 'wrfnc_hourly.csv'), index=False)
np.savez_compressed(os.path.join(OUT, 'wrfnc_maps.npz'), **{f'{k}__{v}': a for k, d in maps.items() for v, a in d.items()})
# storm effect = ctl - edited (positive = the storm warms / moistens / adds onshore wind)
eff = []
num = [x for x in R.columns if R[x].dtype.kind == 'f' and not x.startswith(('lowest_lon', 'lowest_lat'))]
for c, g in R.groupby('case'):
    g = g.set_index('run')
    if 'ctl' not in g.index:
        continue
    for r in [x for x in g.index if x != 'ctl']:
        eff.append(dict(case=c, coast=g.coast.iloc[0], name=g.name.iloc[0], pair=f'ctl-{r}', **{k: g.loc['ctl', k] - g.loc[r, k] for k in num}))
E = pd.DataFrame(eff); E.to_csv(os.path.join(OUT, 'wrfnc_effects.csv'), index=False)
if not E.empty:
    print(E[['case', 'coast', 'name', 'pair'] + [x for x in ('T2_V', 'Tmax', 'Q2_day', 'onshore_day', 'T850_day', 'w500_day', 'CLDMAX_day',
                                                             'SWDOWN_day', 'PBLH_day', 'rain_day', 'rain24_V')
                                                 if x in E]].round(2).to_string(index=False))
