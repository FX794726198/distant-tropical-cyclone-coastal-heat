"""Remove the axisymmetric outer circulation of a tropical cyclone (used by the WIDE variants).

The disk infill in tcfilter removes the storm core inside R, but a large storm keeps a cyclonic
outer circulation and a broad low far beyond R (e.g. Lekima: 850-hPa azimuthal-mean tangential
wind 12-17 m/s at 500-800 km). Here the azimuthal-mean tangential wind and the azimuthal-mean
height/pressure anomaly (relative to the R_out..R_out+100 km ring) are removed, with weight 1
inside R_in and a cosine taper to 0 at R_out. Temperature and humidity are not touched.
"""
import numpy as np
from tcfilter import gc_dist

RE = 6371.0


class SymRemover:
    def __init__(self, LAT, LON, lon0, lat0, R_in, R_out=1500.0, dr=50.0):
        self.r = gc_dist(LON, LAT, lon0, lat0)
        dx = np.radians(((LON - lon0 + 180) % 360) - 180) * RE * np.cos(np.radians(0.5 * (LAT + lat0)))
        dy = np.radians(LAT - lat0) * RE
        rr = np.maximum(np.hypot(dx, dy), 1e-6)
        self.etx, self.ety = -dy / rr, dx / rr          # counter-clockwise tangential unit vector
        self.dr = dr
        self.nb = int((R_out + 100) / dr) + 1
        self.bins = np.minimum((self.r / dr).astype(int), self.nb - 1)
        self.sel = self.r < R_out + 100
        self.cnt = np.bincount(self.bins[self.sel], minlength=self.nb)
        w = np.zeros_like(self.r)
        w[self.r <= R_in] = 1.0
        t = (self.r > R_in) & (self.r < R_out)
        w[t] = 0.5 * (1 + np.cos(np.pi * (self.r[t] - R_in) / (R_out - R_in)))
        self.w = w
        self.ring = (self.r >= R_out) & (self.r < R_out + 100)

    def profile(self, f):
        s = np.bincount(self.bins[self.sel], weights=f[self.sel].astype(np.float64), minlength=self.nb)
        return s / np.maximum(self.cnt, 1)

    def vt_profile(self, u, v):
        return self.profile(u * self.etx + v * self.ety)

    def wind_part(self, u, v):
        """(du, dv): the tapered axisymmetric tangential wind, to be subtracted."""
        m = self.vt_profile(u, v)[self.bins]
        return (self.w * m * self.etx).astype(u.dtype), (self.w * m * self.ety).astype(v.dtype)

    def scalar_part(self, f):
        """Tapered axisymmetric anomaly relative to the outer ring, to be subtracted."""
        ref = f[self.ring].mean()
        return (self.w * (self.profile(f)[self.bins] - ref)).astype(f.dtype)
