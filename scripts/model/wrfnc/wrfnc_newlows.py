"""New lows in the edited WRF runs of the multi-coast experiment (the check of the Guangdong pilot, S6, made systematic).

At every 3rd hour of each run's d01 output, a low is a local minimum of sea-level pressure (the lowest point within about
150 km) that lies at least 2 hPa below the mean of the ring 300-500 km around it, at least 300 km from the domain edge
and over terrain below 500 m (sea-level pressure reduced over high ground is unreliable).  A low of an edited run
(widel, nullw) is new when it lies more than 400 km from every low of the control run at the same time.  This uses no
other threshold; how deep and how close to the region a new low is decides how a case is reported, after the runs.

Outputs ($ROOT/out/):
  wrfnc_newlows.csv        every new low: case, coast, run, time, hours from V, lon, lat, slp, deficit (hPa),
                           distance (km) to the removed storm's track position and to the nearest region cell
  wrfnc_newlow_summary.csv per edited run: deepest new low up to V + 12 h and over the whole run (deficit, time, position,
                           distances) and the number of output times with a new low
Usage (as a job: wrfnc_newlows.slurm): TCH=/scratch/$USER/tcheat python wrfnc_newlows.py [CASES...]
2026-11-01 (after case 330 widel grew a 10 hPa low in the northern Bay of Bengal, 600 km from the edit centre).
"""
import glob, os, sys
import numpy as np, pandas as pd, netCDF4
from scipy.ndimage import minimum_filter, uniform_filter

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
ROOT = os.environ.get('WRFNC', os.path.join(TCH, 'wrfnc'))
OUT = os.path.join(ROOT, 'out'); os.makedirs(OUT, exist_ok=True)
CASES = pd.read_csv(os.path.join(ROOT, 'inputs', 'wrfnc_cases.csv')).set_index('case')
TR = pd.read_csv(os.path.join(ROOT, 'inputs', 'wrfnc_tracks.csv')); TR['time'] = pd.to_datetime(TR.time)
EDITED = ('widel', 'nullw', 'wide')
LOCAL_KM, EDGE_KM, RING, MIN_DEF, NEW_KM, HGT_MAX = 150.0, 300.0, (300.0, 500.0), 2.0, 400.0, 500.0
STEP_H = 3                                          # every 3rd hour


def gc_dist(lon, lat, lon0, lat0):
    p1, p2 = np.radians(lat), np.radians(lat0); dl = np.radians(lon0 - lon)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def region_points(coast):
    f = os.path.join(ROOT, 'inputs', f'cells_{coast}.csv')
    c = pd.read_csv(f, index_col=0)
    return c.lon.values.astype(float), c.lat.values.astype(float)


def lows_of_run(run):
    """{time: DataFrame(lon, lat, slp, deficit)} for every d01 output time."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run, 'wrfout_d01_*'))):
        nc = netCDF4.Dataset(f)
        la, lo = np.asarray(nc['XLAT'][0], float), np.asarray(nc['XLONG'][0], float)
        h = np.asarray(nc['HGT'][0], float); dx = float(nc.DX) / 1000.0
        k = max(1, int(round(LOCAL_KM / dx))); e = max(1, int(round(EDGE_KM / dx)))
        ok = np.zeros(la.shape, bool); ok[e:-e, e:-e] = True; ok &= h < HGT_MAX
        tt = [pd.Timestamp(b''.join(x).decode().replace('_', ' ')) for x in nc['Times'][:]]
        for n, t in enumerate(tt):
            if t.hour % STEP_H:
                continue
            ps = np.asarray(nc['PSFC'][n], float); t2 = np.asarray(nc['T2'][n], float)
            slp = ps * (1 - 0.0065 * h / (t2 + 0.0065 * h)) ** -5.257 / 100
            w = 2 * k + 1                                  # local minimum, and below the local mean (drops flat areas)
            cand = np.argwhere((slp == minimum_filter(slp, size=w, mode='nearest')) & (slp < uniform_filter(slp, size=w, mode='nearest') - 0.1) & ok)
            rows = []
            for j, i in cand:
                d = gc_dist(lo, la, lo[j, i], la[j, i]); ring = (d >= RING[0]) & (d <= RING[1])
                if ring.any():
                    dfc = float(slp[ring].mean() - slp[j, i])
                    if dfc >= MIN_DEF:
                        rows.append(dict(lon=float(lo[j, i]), lat=float(la[j, i]), slp=float(slp[j, i]), deficit=dfc))
            out[t] = pd.DataFrame(rows, columns=['lon', 'lat', 'slp', 'deficit'])
        nc.close()
    return out


def track_pos(c, t):
    g = TR[TR.case == c].set_index('time').sort_index()[['lon', 'lat']]
    g = g.reindex(g.index.union([t])).interpolate('time', limit_area='inside')
    return (float(g.loc[t, 'lon']), float(g.loc[t, 'lat'])) if np.isfinite(g.loc[t, 'lon']) else (np.nan, np.nan)


cases = [int(x) for x in sys.argv[1:]] or list(CASES.index)
new, summ = [], []
for c in cases:
    cs = CASES.loc[c]; coast = cs.coast; V = pd.Timestamp(cs.V); D = os.path.join(ROOT, f'case_{c:03d}')
    if not os.path.isdir(os.path.join(D, 'ctl')):
        continue
    rlon, rlat = region_points(coast)
    ctl = lows_of_run(os.path.join(D, 'ctl'))
    for r in EDITED:
        run = os.path.join(D, r)
        if not os.path.isdir(run) or not glob.glob(os.path.join(run, 'wrfout_d01_*')):
            continue
        ed = lows_of_run(run); rows = []
        for t, L in ed.items():
            C = ctl.get(t)
            if C is None:
                continue
            tl = track_pos(c, t)
            for _, lw in L.iterrows():
                if len(C) and gc_dist(C.lon.values, C.lat.values, lw.lon, lw.lat).min() <= NEW_KM:
                    continue
                rows.append(dict(case=c, coast=coast, run=r, time=t, h_from_V=(t - V) / pd.Timedelta('1h'), lon=lw.lon, lat=lw.lat,
                                 slp=lw.slp, deficit=lw.deficit,
                                 d_track=float(gc_dist(lw.lon, lw.lat, *tl)) if np.isfinite(tl[0]) else np.nan,
                                 d_region=float(gc_dist(rlon, rlat, lw.lon, lw.lat).min())))
        N = pd.DataFrame(rows); new.append(N)
        s = dict(case=c, coast=coast, name=cs['name'], run=r, n_times=len(ed),
                 n_times_newlow=int(N.time.nunique()) if len(N) else 0)
        for tag, sel in (('toV12', N[N.h_from_V <= 12] if len(N) else N), ('all', N)):
            if len(sel):
                b = sel.loc[sel.deficit.idxmax()]
                s.update({f'{tag}_deficit': b.deficit, f'{tag}_h': b.h_from_V, f'{tag}_lon': b.lon, f'{tag}_lat': b.lat,
                          f'{tag}_d_track': b.d_track, f'{tag}_d_region': b.d_region})
            else:
                s[f'{tag}_deficit'] = 0.0
        summ.append(s)
        print(c, coast, r, f"new-low times {s['n_times_newlow']}/{s['n_times']}, deepest to V+12 h {s['toV12_deficit']:.1f} hPa,"
              f" whole run {s['all_deficit']:.1f} hPa", flush=True)
NL = pd.concat([x for x in new if len(x)], ignore_index=True) if any(len(x) for x in new) else pd.DataFrame()
NL.to_csv(os.path.join(OUT, 'wrfnc_newlows.csv'), index=False, float_format='%.3f')
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, 'wrfnc_newlow_summary.csv'), index=False, float_format='%.3f')
if not S.empty:
    print(S.round(1).to_string(index=False))
