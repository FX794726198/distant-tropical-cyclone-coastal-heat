"""Remove (or weaken, or move) a tropical-cyclone vortex in gridded fields.

Inside a disk of radius R around the storm centre, each 2-D field is split into
an environment part, the harmonic (Laplace) infill that matches the field on the
disk edge, and a disturbance part (field minus infill). The disturbance is
removed with a cosine taper over the outer `taper_km`. Works on regular lat-lon
grids (metric='latlon') and on conformal map projections such as WRF Mercator
grids (metric='conformal', Laplacian in grid-index space).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RE = 6371.0


def gc_dist(lon, lat, lon0, lat0):
    lon, lat, lon0, lat0 = map(np.radians, (lon, lat, lon0, lat0))
    c = np.sin(lat) * np.sin(lat0) + np.cos(lat) * np.cos(lat0) * np.cos(lon - lon0)
    return RE * np.arccos(np.clip(c, -1, 1))


def destination(lon0, lat0, bearing_deg, dist_km):
    b, d = np.radians(bearing_deg), dist_km / RE
    la0, lo0 = np.radians(lat0), np.radians(lon0)
    la = np.arcsin(np.sin(la0) * np.cos(d) + np.cos(la0) * np.sin(d) * np.cos(b))
    lo = lo0 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la0), np.cos(d) - np.sin(la0) * np.sin(la))
    return (np.degrees(lo) + 360) % 360, np.degrees(la)


def find_centre(mslp, LAT, LON, lon_g, lat_g, search_km=300):
    d = gc_dist(LON, LAT, lon_g, lat_g)
    m = np.where(d <= search_km, mslp, np.inf)
    j, i = np.unravel_index(np.argmin(m), m.shape)
    return float(LON[j, i]), float(LAT[j, i])


def plane_deficit(mslp, LAT, LON, lon0, lat0, R_km):
    """Max of (plane fitted on the R..R+100 km ring) minus field, inside R/2 (hPa if mslp in hPa)."""
    d = gc_dist(LON, LAT, lon0, lat0)
    ring = (d > R_km) & (d < R_km + 100)
    X = np.c_[np.ones(ring.sum()), LON[ring], LAT[ring]]
    b = np.linalg.lstsq(X, mslp[ring], rcond=None)[0]
    plane = b[0] + b[1] * LON + b[2] * LAT
    inner = d < R_km / 2
    return float((plane - mslp)[inner].max())


class DiskInfill:
    def __init__(self, LAT, LON, lon0, lat0, R_km=800.0, taper_km=150.0, metric='latlon'):
        """LAT, LON: 2-D arrays of the grid (any regular index grid)."""
        self.d = gc_dist(LON, LAT, lon0, lat0)
        self.inside = self.d < R_km
        ny, nx = self.inside.shape
        if self.inside[0, :].any() or self.inside[-1, :].any() or self.inside[:, 0].any() or self.inside[:, -1].any():
            raise ValueError('disk touches the window edge; use a larger window')
        pts = np.argwhere(self.inside)
        idx = -np.ones((ny, nx), int)
        idx[self.inside] = np.arange(len(pts))
        rows, cols, vals, bk, bj, bi, bw = [], [], [], [], [], [], []
        for k, (j, i) in enumerate(pts):
            ax = 1.0 / np.cos(np.radians(LAT[j, i])) ** 2 if metric == 'latlon' else 1.0
            for (jj, ii, w) in ((j, i - 1, ax), (j, i + 1, ax), (j - 1, i, 1.0), (j + 1, i, 1.0)):
                if self.inside[jj, ii]:
                    rows.append(k); cols.append(idx[jj, ii]); vals.append(w)
                else:
                    bk.append(k); bj.append(jj); bi.append(ii); bw.append(w)
            rows.append(k); cols.append(k); vals.append(-2 * ax - 2)
        self.lu = spla.splu(sp.csc_matrix((vals, (rows, cols)), shape=(len(pts), len(pts))))
        self.pts = pts
        self.bk, self.bj, self.bi, self.bw = map(np.asarray, (bk, bj, bi, bw))
        w = np.zeros_like(self.d)
        core = self.d <= R_km - taper_km
        ring = (~core) & self.inside
        w[core] = 1.0
        w[ring] = 0.5 * (1 + np.cos(np.pi * (self.d[ring] - (R_km - taper_km)) / taper_km))
        self.w = w

    def disturbance(self, f):
        """Tapered disturbance w*(f - harmonic infill); zero outside the disk."""
        b = np.zeros(len(self.pts))
        np.add.at(b, self.bk, -self.bw * f[self.bj, self.bi])
        h = f.astype(np.float64).copy()
        h[self.inside] = self.lu.solve(b)
        return (self.w * (f - h)).astype(f.dtype)

    def apply(self, f, amp=0.0):
        """amp=0 removes the vortex, 0.5 halves it, 1 leaves f unchanged."""
        return f - (1 - amp) * self.disturbance(f)


def shift_array(D, dj, di):
    """Translate a 2-D array by integer grid offsets (zero fill)."""
    out = np.zeros_like(D)
    ny, nx = D.shape
    ys, yd = (slice(0, ny - dj), slice(dj, ny)) if dj >= 0 else (slice(-dj, ny), slice(0, ny + dj))
    xs, xd = (slice(0, nx - di), slice(di, nx)) if di >= 0 else (slice(-di, nx), slice(0, nx + di))
    out[yd, xd] = D[ys, xs]
    return out
