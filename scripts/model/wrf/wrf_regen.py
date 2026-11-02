"""WRF regeneration check (d01 min SLP near the ctl storm centre) and CTL vs ERA5 station-mean T2 at V."""
import glob, os, numpy as np, pandas as pd, netCDF4, xarray as xr
from tcfilter import gc_dist
T = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat')); W = T + "/wrf"
CASES = pd.read_csv(T + "/inputs/gc_cases.csv").set_index("case")
TR = pd.read_csv(T + "/inputs/wrf_tracks.csv"); TR["time"] = pd.to_datetime(TR.time)
STN = pd.read_csv(T + "/inputs/gd_stations.csv", index_col=0)

def field(run, t):
    s = t.strftime("%Y-%m-%d_%H:%M:%S")
    for f in sorted(glob.glob(f"{run}/wrfout_d01_*")):
        nc = netCDF4.Dataset(f); tt = [b"".join(x).decode() for x in nc["Times"][:]]
        if s in tt:
            k = tt.index(s); ps = nc["PSFC"][k]; h = nc["HGT"][0]; t2 = nc["T2"][k]
            slp = ps * (1 - 0.0065 * h / (t2 + 0.0065 * h)) ** -5.257 / 100
            la, lo = nc["XLAT"][0], nc["XLONG"][0]; nc.close(); return slp, la, lo
        nc.close()
    return None

for c in (6, 16, 24, 44):
    V = pd.Timestamp(CASES.loc[c, "V"])
    for t in (V - pd.Timedelta("24h"), V):
        ctl = field(f"{W}/case_{c:03d}/ctl", t); wide = field(f"{W}/case_{c:03d}/wide", t)
        g = TR[(TR.case == c)].set_index("time").sort_index()
        g = g.reindex(g.index.union([t])).interpolate("time").loc[t]
        slp, la, lo = ctl
        d = gc_dist(lo, la, g.lon, g.lat); m = np.where(d < 1000, slp, np.inf)
        j, i = np.unravel_index(np.argmin(m), m.shape); c0 = (float(lo[j, i]), float(la[j, i]))
        dd = gc_dist(lo, la, *c0); ring = (dd > 800) & (dd < 1000); inner = dd < 600
        out = []
        for name, f in (("ctl", ctl), ("wide", wide)):
            s = f[0]; mn = float(s[inner].min()); jj, ii = np.unravel_index(np.argmin(np.where(inner, s, np.inf)), s.shape)
            out.append(f"{name} minSLP {mn:.1f} hPa at {float(lo[jj,ii]):.1f}E {float(la[jj,ii]):.1f}N deficit {float(s[ring].mean()) - mn:.1f} hPa")
        print(f"case {c} {t}: ctl centre {c0[0]:.1f}E {c0[1]:.1f}N | " + " | ".join(out))
    e = xr.open_dataset(f"{T}/era5/{V.strftime('%Y%m%d%H')}.nc").sortby("latitude")
    pts = dict(latitude=xr.DataArray(STN.lat.values, dims="s"), longitude=xr.DataArray(STN.lon.values, dims="s"))
    print(f"case {c} V {V}: ERA5 station-mean T2 {float(e['2m_temperature'].interp(**pts).mean()):.2f} K")
