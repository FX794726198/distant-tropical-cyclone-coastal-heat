"""Remove the typhoon vortex from WRF met_em files (NOTC run), using tcfilter.DiskInfill on the conformal grid.

Usage: python metem_filter.py --indir RUN_CTL/metem --outdir RUN_NOTC/metem --case 25 --amp 0 [--wide]
--wide (WIDE run): after the disk infill, also remove the storm's axisymmetric outer circulation out to 1500 km
  (tangential UU/VV, GHT k >= 1, PMSL with PSFC by the same relative change; TT and RH untouched), in d01 and,
  with the same centre, in d02.
- Copies every met_em.d0?.*.nc from indir to outdir and filters d01 at every time.
- d02 is filtered only if the disk reaches into it (it should not; the radius is limited so that it does not).
- Isobaric levels (k >= 1) of TT, RH/SPECHUMD, GHT, UU, VV are filtered; the surface level (k = 0) of TT, RH/SPECHUMD,
  UU, VV gets the disturbance of level 1; GHT level 0 (terrain) is untouched; PMSL is filtered and PSFC gets the same
  relative change.
"""
import argparse, glob, os, re, shutil, sys
import numpy as np, pandas as pd
import netCDF4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcfilter import DiskInfill, find_centre, gc_dist, plane_deficit
from symvortex import SymRemover

R_OUT = 1500.0

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
STN = pd.read_csv(os.path.join(TCH, 'inputs', 'gd_stations.csv'), index_col=0)


def guess(tracks, t):
    s = tracks.set_index('time').sort_index()
    if t in s.index:
        r = s.loc[t]; return float(r.lon), float(r.lat)
    s2 = s.reindex(s.index.union([t])).interpolate('time')
    if np.isnan(s2.loc[t, 'lon']):
        return None
    return float(s2.loc[t, 'lon']), float(s2.loc[t, 'lat'])


def edge_dist(LAT, LON, lon0, lat0):
    e = np.concatenate([np.c_[LON[0], LAT[0]], np.c_[LON[-1], LAT[-1]], np.c_[LON[:, 0], LAT[:, 0]], np.c_[LON[:, -1], LAT[:, -1]]])
    return gc_dist(e[:, 0], e[:, 1], lon0, lat0).min()


def sym_profiles(path, c, R):
    """Azimuthal-mean profiles around c on a d01 file: PMSL and GHT anomalies (vs the R_OUT ring), tangential wind."""
    nc = netCDF4.Dataset(path)
    S = SymRemover(nc['XLAT_M'][0], nc['XLONG_M'][0], *c, R_in=R, R_out=R_OUT)
    pm = nc['PMSL'][0]
    prof = {'PMSL': S.profile(pm) - pm[S.ring].mean()}
    g = nc['GHT'][0]
    prof['GHT'] = [None] + [S.profile(g[j]) - g[j][S.ring].mean() for j in range(1, g.shape[0])]
    U, V = nc['UU'][0], nc['VV'][0]
    um, vm = 0.5 * (U[:, :, :-1] + U[:, :, 1:]), 0.5 * (V[:, :-1, :] + V[:, 1:, :])
    prof['VT'] = [S.vt_profile(um[j], vm[j]) for j in range(U.shape[0])]
    nc.close()
    return prof


def sym_apply(path, c, R, prof, frac=1.0):
    """Subtract frac x the tapered axisymmetric parts (profiles from d01) on this file's grids (in place)."""
    nc = netCDF4.Dataset(path, 'r+')
    S = {k: SymRemover(nc[f'XLAT_{k}'][0], nc[f'XLONG_{k}'][0], *c, R_in=R, R_out=R_OUT) for k in ('M', 'U', 'V')}
    sm, su, sv = S['M'], S['U'], S['V']
    nb = len(prof['PMSL'])
    bm, bu, bv = (np.minimum(x.bins, nb - 1) for x in (sm, su, sv))
    d = sm.w * prof['PMSL'][bm]
    nc['PSFC'][0] = nc['PSFC'][0] - frac * d * nc['PSFC'][0] / nc['PMSL'][0]
    nc['PMSL'][0] = nc['PMSL'][0] - frac * d
    g = nc['GHT'][0]
    for j in range(1, g.shape[0]):
        g[j] = g[j] - frac * sm.w * prof['GHT'][j][bm]
    nc['GHT'][0] = g
    U, V = nc['UU'][0], nc['VV'][0]
    for j in range(U.shape[0]):
        U[j] = U[j] - frac * su.w * prof['VT'][j][bu] * su.etx
        V[j] = V[j] - frac * sv.w * prof['VT'][j][bv] * sv.ety
    nc['UU'][0], nc['VV'][0] = U, V
    nc.close()


def filt_file(path, tracks, amp, d02_box=None):
    t = pd.Timestamp(re.search(r'(\d{4}-\d\d-\d\d_\d\d)', os.path.basename(path)).group(1).replace('_', ' ') + ':00')
    g = guess(tracks, t)
    if g is None:
        return f'{os.path.basename(path)}: no track position, unchanged', None
    nc = netCDF4.Dataset(path, 'r+')
    LAT, LON = nc['XLAT_M'][0], nc['XLONG_M'][0]
    pm = nc['PMSL'][0] / 100.0
    c = find_centre(pm, LAT, LON, *g)
    dmin = gc_dist(STN.lon.values, STN.lat.values, *c).min()
    R = min(800.0, dmin - 150.0, edge_dist(LAT, LON, *c) - 60.0)
    if d02_box is not None:
        R = min(R, d02_box(c) - 50.0)
    if R < 300:
        nc.close(); return f'{os.path.basename(path)}: R={R:.0f} km too small, unchanged', None
    FM = DiskInfill(LAT, LON, *c, R_km=R, metric='conformal')
    FU = DiskInfill(nc['XLAT_U'][0], nc['XLONG_U'][0], *c, R_km=R, metric='conformal')
    FV = DiskInfill(nc['XLAT_V'][0], nc['XLONG_V'][0], *c, R_km=R, metric='conformal')
    k = 1 - amp
    before = plane_deficit(pm, LAT, LON, *c, min(R, 600))
    dP = FM.disturbance(nc['PMSL'][0])
    nc['PSFC'][0] = nc['PSFC'][0] - k * dP * nc['PSFC'][0] / nc['PMSL'][0]
    nc['PMSL'][0] = nc['PMSL'][0] - k * dP
    for name, F in (('TT', FM), ('RH', FM), ('SPECHUMD', FM), ('GHT', FM), ('UU', FU), ('VV', FV)):
        if name not in nc.variables:
            continue
        a = nc[name][0]
        D = np.stack([F.disturbance(a[j]) if j >= 1 else np.zeros_like(a[0]) for j in range(a.shape[0])])
        if name != 'GHT':
            D[0] = D[1]
        a = a - k * D
        if name in ('RH', 'SPECHUMD'):
            a = np.clip(a, 0, 100 if name == 'RH' else None)
        nc[name][0] = a
    after = plane_deficit(nc['PMSL'][0] / 100.0, LAT, LON, *c, min(R, 600))
    nc.close()
    return f'{os.path.basename(path)}: centre {c[0]:.2f},{c[1]:.2f} R={R:.0f} km deficit {before:.1f} -> {after:.1f} hPa', (c, R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir'); ap.add_argument('--outdir'); ap.add_argument('--case', type=int)
    ap.add_argument('--amp', type=float, default=0.0)
    ap.add_argument('--wide', action='store_true')
    a = ap.parse_args()
    tr = pd.read_csv(os.path.join(TCH, 'inputs', 'wrf_tracks.csv'))
    tr = tr[tr.case == a.case].copy(); tr['time'] = pd.to_datetime(tr.time)
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.indir, 'met_em.d0*.nc')))
    d02 = [f for f in files if '.d02.' in f]
    box = None
    if d02:
        nc = netCDF4.Dataset(d02[0]); LA, LO = nc['XLAT_M'][0], nc['XLONG_M'][0]; nc.close()
        box = lambda c: 0.0 if (LA.min() <= c[1] <= LA.max() and LO.min() <= c[0] <= LO.max()) else min(
            gc_dist(LO.ravel()[::7], LA.ravel()[::7], *c))
    done = {}
    for f in [x for x in files if '.d01.' in x]:
        o = os.path.join(a.outdir, os.path.basename(f)); shutil.copy(f, o)
        msg, cr = filt_file(o, tr, a.amp, d02_box=box)
        if a.wide and cr is not None:
            prof = sym_profiles(o, *cr)
            sym_apply(o, *cr, prof, frac=1 - a.amp)
            after = sym_profiles(o, *cr)
            k6, k10 = int(600 / 50), int(1000 / 50)
            msg += (f'; outer circulation removed to {R_OUT:.0f} km: level-1 vt at 600/1000 km {prof["VT"][1][k6]:.1f}/{prof["VT"][1][k10]:.1f}'
                    f' -> {after["VT"][1][k6]:.1f}/{after["VT"][1][k10]:.1f} m/s')
            done[os.path.basename(f).replace('.d01.', '.d02.')] = (cr, prof)
        print(msg, flush=True)
    for f in d02:
        o = os.path.join(a.outdir, os.path.basename(f)); shutil.copy(f, o)
        dd = done.get(os.path.basename(f))
        if dd is not None:
            sym_apply(o, *dd[0], dd[1], frac=1 - a.amp)
            print(os.path.basename(f), ': d02 outer circulation removed with the d01 centre and R', flush=True)
        elif f == d02[0]:
            print(os.path.basename(f), ': d02 initial time left unchanged (disk kept outside d02)', flush=True)


if __name__ == '__main__':
    main()
