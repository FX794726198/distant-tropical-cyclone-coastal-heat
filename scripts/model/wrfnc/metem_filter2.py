"""Storm removal in WRF met_em files for the multi-coast runs (NC upgrade, 2026-11-01).

Generalises paper/scripts/model/wrf/metem_filter.py (Guangdong) to any coast and adds the WIDEL edit used by the
GraphCast all-exposure-day test (A2), so that WRF and GraphCast remove the same thing:
  widel  core disk infill (R = min(800, dmin - 150) km, dmin = distance to the nearest region cell) + the storm's
         axisymmetric outer circulation (tangential wind at all levels, geopotential, PMSL with PSFC by the same relative
         change) removed with a cosine taper that ends 50 km before the nearest region cell; T and humidity untouched.
  wide   as the Guangdong runs: outer removal tapering to zero at 1,500 km.
  nullw  the widel procedure applied at a storm-free sea point on another side of the region; checks that the
         procedure itself does not warm or cool the coast.  The point sits at the storm's distance from the region
         centre (capped, at the first time, at the largest distance that fits in d01) with a bearing offset from the
         storm chosen once at the first time (180, +-150, +-120, +-90, +-60 degrees) and kept.  It uses the real storm's
         radii at that time, shrunk where needed so that the disk stays 200 km and the taper 100 km inside d01, the disk
         150 km and the taper 50 km from the nearest region cell, and the disk outside d02.  (v2, 11-01: v1 kept the full
         radii and found no point when the storm was ~2,000 km away, as for case 393 at the start.)
Every d01 time is edited (initial and lateral boundary conditions); d02 gets the outer-circulation part with the d01
centre and profiles; the disk is kept outside d02.

Usage: python metem_filter2.py --indir CASE/metem_ctl --outdir CASE/metem_widel --case 393 --cells cells_W17.csv
                                --tracks wrfnc_tracks.csv [--others wrfnc_others.csv] --mode widel|wide|nullw
Needs tcfilter.py and symvortex.py (the Guangdong versions) beside this file or in $TCH/scripts.
"""
import argparse, glob, json, os, re, shutil, sys
import numpy as np, pandas as pd
import netCDF4
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')), 'scripts'))
from tcfilter import DiskInfill, destination, find_centre, gc_dist, plane_deficit
from symvortex import SymRemover

R_WIDE, R_MAX, STN_BUF, OUT_BUF = 1500.0, 800.0, 150.0, 50.0


def guess(tracks, t):
    s = tracks.set_index('time').sort_index()
    if t in s.index:
        r = s.loc[t]; return float(r.lon), float(r.lat)
    s2 = s[['lon', 'lat']].reindex(s.index.union([t])).interpolate('time', limit_area='inside')
    if np.isnan(s2.loc[t, 'lon']):
        return None
    return float(s2.loc[t, 'lon']), float(s2.loc[t, 'lat'])


def edge_dist(LAT, LON, lon0, lat0):
    e = np.concatenate([np.c_[LON[0], LAT[0]], np.c_[LON[-1], LAT[-1]], np.c_[LON[:, 0], LAT[:, 0]], np.c_[LON[:, -1], LAT[:, -1]]])
    return gc_dist(e[:, 0], e[:, 1], lon0, lat0).min()


def bearing(lon0, lat0, lon1, lat1):
    p0, p1, dl = np.radians(lat0), np.radians(lat1), np.radians(lon1 - lon0)
    return np.degrees(np.arctan2(np.sin(dl) * np.cos(p1), np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)))


def sym_profiles(path, c, R, R_out):
    nc = netCDF4.Dataset(path)
    S = SymRemover(nc['XLAT_M'][0], nc['XLONG_M'][0], *c, R_in=R, R_out=R_out)
    pm = nc['PMSL'][0]
    prof = {'PMSL': S.profile(pm) - pm[S.ring].mean()}
    g = nc['GHT'][0]
    prof['GHT'] = [None] + [S.profile(g[j]) - g[j][S.ring].mean() for j in range(1, g.shape[0])]
    U, V = nc['UU'][0], nc['VV'][0]
    um, vm = 0.5 * (U[:, :, :-1] + U[:, :, 1:]), 0.5 * (V[:, :-1, :] + V[:, 1:, :])
    prof['VT'] = [S.vt_profile(um[j], vm[j]) for j in range(U.shape[0])]
    nc.close()
    return prof


def sym_apply(path, c, R, R_out, prof):
    nc = netCDF4.Dataset(path, 'r+')
    S = {k: SymRemover(nc[f'XLAT_{k}'][0], nc[f'XLONG_{k}'][0], *c, R_in=R, R_out=R_out) for k in ('M', 'U', 'V')}
    sm, su, sv = S['M'], S['U'], S['V']
    nb = len(prof['PMSL'])
    bm, bu, bv = (np.minimum(x.bins, nb - 1) for x in (sm, su, sv))
    d = sm.w * prof['PMSL'][bm]
    nc['PSFC'][0] = nc['PSFC'][0] - d * nc['PSFC'][0] / nc['PMSL'][0]
    nc['PMSL'][0] = nc['PMSL'][0] - d
    g = nc['GHT'][0]
    for j in range(1, g.shape[0]):
        g[j] = g[j] - sm.w * prof['GHT'][j][bm]
    nc['GHT'][0] = g
    U, V = nc['UU'][0], nc['VV'][0]
    for j in range(U.shape[0]):
        U[j] = U[j] - su.w * prof['VT'][j][bu] * su.etx
        V[j] = V[j] - sv.w * prof['VT'][j][bv] * sv.ety
    nc['UU'][0], nc['VV'][0] = U, V
    nc.close()


def disk(nc, c, R):
    LAT, LON = nc['XLAT_M'][0], nc['XLONG_M'][0]
    FM = DiskInfill(LAT, LON, *c, R_km=R, metric='conformal')
    FU = DiskInfill(nc['XLAT_U'][0], nc['XLONG_U'][0], *c, R_km=R, metric='conformal')
    FV = DiskInfill(nc['XLAT_V'][0], nc['XLONG_V'][0], *c, R_km=R, metric='conformal')
    dP = FM.disturbance(nc['PMSL'][0])
    nc['PSFC'][0] = nc['PSFC'][0] - dP * nc['PSFC'][0] / nc['PMSL'][0]
    nc['PMSL'][0] = nc['PMSL'][0] - dP
    for name, F in (('TT', FM), ('RH', FM), ('SPECHUMD', FM), ('GHT', FM), ('UU', FU), ('VV', FV)):
        if name not in nc.variables:
            continue
        a = nc[name][0]
        D = np.stack([F.disturbance(a[j]) if j >= 1 else np.zeros_like(a[0]) for j in range(a.shape[0])])
        if name != 'GHT':
            D[0] = D[1]
        a = a - D
        if name in ('RH', 'SPECHUMD'):
            a = np.clip(a, 0, 100 if name == 'RH' else None)
        nc[name][0] = a


class NullPicker:
    """Storm-free sea point at the storm's distance from the region centre (capped at the first time to the largest
    distance that fits); bearing offset and cap fixed at the first call that finds a point."""
    OFFS = (180, 150, -150, 120, -120, 90, -90, 60, -60)

    def __init__(self, cen, others, cells, d02_box):
        self.cen, self.others, self.cells, self.box = cen, others, cells, d02_box
        self.off, self.cap = None, None

    def radii(self, x, y, R, R_out, LAT, LON):
        e = edge_dist(LAT, LON, x, y)
        dn = gc_dist(self.cells.lon.values, self.cells.lat.values, x, y).min()
        Rn = min(R, e - 200.0, dn - STN_BUF)
        if self.box is not None:
            Rn = min(Rn, self.box((x, y)) - 50.0)
        return Rn, min(R_out, e - 100.0, dn - OUT_BUF)

    def ok(self, x, y, c, R, R_out, LAT, LON, LM, t):
        if gc_dist(x, y, *c) < 1200:
            return None
        j, i = np.unravel_index(np.argmin(gc_dist(LON, LAT, x, y)), LAT.shape)
        if LM[j, i] >= 0.5 or edge_dist(LAT, LON, x, y) < 400 or gc_dist(LON[j, i], LAT[j, i], x, y) > 50:
            return None
        o = self.others[self.others.time == t]
        if not all(gc_dist(x, y, a, b) > 1000 for a, b in zip(o.lon, o.lat)):
            return None
        Rn, Ron = self.radii(x, y, R, R_out, LAT, LON)
        return (Rn, Ron) if (Rn >= 300 and Ron >= Rn + 100) else None

    def point(self, b0, db, d):
        x, y = destination(self.cen[0], self.cen[1], b0 + db, d)
        return float(((x + 180) % 360) - 180), float(y)

    def __call__(self, c, R, R_out, LAT, LON, LM, t):
        d0 = gc_dist(c[0], c[1], *self.cen); b0 = bearing(self.cen[0], self.cen[1], *c)
        if self.off is None:
            for d in np.arange(d0, 599.0, -100.0):               # storm distance first, then shorter ones
                for db in self.OFFS:
                    x, y = self.point(b0, db, d)
                    r = self.ok(x, y, c, R, R_out, LAT, LON, LM, t)
                    if r is not None:
                        self.off, self.cap = db, float(d)
                        return (x, y), r
            return None
        x, y = self.point(b0, self.off, min(d0, self.cap))
        r = self.ok(x, y, c, R, R_out, LAT, LON, LM, t)
        return None if r is None else ((x, y), r)


def filt_file(path, tr, cells, mode, d02_box, null):
    t = pd.Timestamp(re.search(r'(\d{4}-\d\d-\d\d_\d\d)', os.path.basename(path)).group(1).replace('_', ' ') + ':00')
    g = guess(tr, t)
    if g is None:
        return f'{os.path.basename(path)}: no track position, unchanged', None
    nc = netCDF4.Dataset(path, 'r+')
    LAT, LON = nc['XLAT_M'][0], nc['XLONG_M'][0]
    pm = nc['PMSL'][0] / 100.0
    c = find_centre(pm, LAT, LON, *g)
    dmin = gc_dist(cells.lon.values, cells.lat.values, *c).min()
    R = min(R_MAX, dmin - STN_BUF, edge_dist(LAT, LON, *c) - 60.0)
    if d02_box is not None:
        R = min(R, d02_box(c) - 50.0)
    R_out = R_WIDE if mode == 'wide' else max(dmin - OUT_BUF, R + 100.0)
    if R < 300:
        nc.close(); return f'{os.path.basename(path)}: R={R:.0f} km too small, unchanged', None
    where = c
    if mode == 'nullw':
        got = null(c, R, R_out, LAT, LON, nc['LANDMASK'][0], t.strftime('%Y-%m-%dT%H'))
        if got is None:
            nc.close(); return f'{os.path.basename(path)}: no null point, unchanged', None
        where, (R, R_out) = got
    before = plane_deficit(pm, LAT, LON, *where, min(R, 600))
    disk(nc, where, R)
    after = plane_deficit(nc['PMSL'][0] / 100.0, LAT, LON, *where, min(R, 600))
    nc.close()
    msg = (f'{os.path.basename(path)}: storm {c[0]:.2f},{c[1]:.2f} dmin {dmin:.0f} R {R:.0f} R_out {R_out:.0f} km; edit at '
           f'{where[0]:.2f},{where[1]:.2f}; deficit {before:.1f} -> {after:.1f} hPa')
    return msg, (where, R, R_out, c, dmin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir'); ap.add_argument('--outdir'); ap.add_argument('--case', type=int)
    ap.add_argument('--cells'); ap.add_argument('--tracks'); ap.add_argument('--others', default=None)
    ap.add_argument('--mode', choices=['widel', 'wide', 'nullw'], default='widel')
    a = ap.parse_args()
    cells = pd.read_csv(a.cells, index_col=0)
    tr = pd.read_csv(a.tracks); tr = tr[tr.case == a.case].copy(); tr['time'] = pd.to_datetime(tr.time)
    oth = pd.read_csv(a.others) if a.others else pd.DataFrame(columns=['case', 'time', 'lon', 'lat'])
    oth = oth[oth.case == a.case]
    cen = (float(cells.lon.mean()), float(cells.lat.mean()))
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.indir, 'met_em.d0*.nc')))
    d02 = [f for f in files if '.d02.' in f]
    box = None
    if d02:
        nc = netCDF4.Dataset(d02[0]); LA, LO = nc['XLAT_M'][0], nc['XLONG_M'][0]; nc.close()
        box = lambda c: 0.0 if (LA.min() <= c[1] <= LA.max() and LO.min() <= c[0] <= LO.max()) else min(
            gc_dist(LO.ravel()[::7], LA.ravel()[::7], *c))
    null = NullPicker(cen, oth, cells, box) if a.mode == 'nullw' else None
    done, diag = {}, []
    for f in [x for x in files if '.d01.' in x]:
        o = os.path.join(a.outdir, os.path.basename(f)); shutil.copy(f, o)
        msg, cr = filt_file(o, tr, cells, a.mode, box, null)
        if cr is not None:
            where, R, R_out, c, dmin = cr
            prof = sym_profiles(o, where, R, R_out)
            sym_apply(o, where, R, R_out, prof)
            after = sym_profiles(o, where, R, R_out)
            k6, k10 = int((R + 50) / 50), int(0.5 * (R + R_out) / 50)      # just outside the disk, middle of the taper
            msg += (f'; outer part removed to {R_out:.0f} km: level-1 vt at {k6 * 50}/{k10 * 50} km {prof["VT"][1][k6]:.1f}/'
                    f'{prof["VT"][1][k10]:.1f} -> {after["VT"][1][k6]:.1f}/{after["VT"][1][k10]:.1f} m/s')
            done[os.path.basename(f).replace('.d01.', '.d02.')] = (where, R, R_out, prof)
            diag.append(dict(file=os.path.basename(f), storm_lon=c[0], storm_lat=c[1], edit_lon=where[0], edit_lat=where[1],
                             dmin=dmin, R=R, R_out=R_out, vt_in_before=float(prof['VT'][1][k6]), vt_in_after=float(after['VT'][1][k6]),
                             null_offset=(null.off if null else np.nan), null_cap_km=(null.cap if null else np.nan)))
        print(msg, flush=True)
    for f in d02:
        o = os.path.join(a.outdir, os.path.basename(f)); shutil.copy(f, o)
        dd = done.get(os.path.basename(f))
        if dd is not None:
            sym_apply(o, dd[0], dd[1], dd[2], dd[3])
            print(os.path.basename(f), ': d02 outer part removed with the d01 centre and radii', flush=True)
    pd.DataFrame(diag).to_csv(os.path.join(a.outdir, 'edit_diag.csv'), index=False)


if __name__ == '__main__':
    main()
