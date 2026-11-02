"""Fig. S4: the storm's footprint in GraphCast (0.25° model, 29 hot cases, full removal): sea-level pressure, rainfall, and 850-hPa
temperature and humidity (the last panel was Fig. 4b in v06).
Inputs: GCX (fp_maps.npz), DB (gc_cases.csv, gc_tracks.csv, gd_stations.csv)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap

GCX = os.environ.get('GCX', 'data/figpack')
DB = os.environ.get('DB', 'data/db')
OUT = os.environ.get('FIGOUT', 'fig')
EXT = [106, 128, 16, 29]
pc = ccrs.PlateCarree()
get, grids = load_maps(f'{GCX}/fp_maps.npz'); la, lo = grids['oper']
cases = pd.read_csv(f'{DB}/gc_cases.csv'); trk = pd.read_csv(f'{DB}/gc_tracks.csv'); stn = pd.read_csv(f'{DB}/gd_stations.csv')
pos = trk.merge(cases[['case', 'V', 'kind']], left_on=['case', 'time'], right_on=['case', 'V']); pos = pos[pos.kind == 'heat']


def disc(base, bounds, lo_=0.06, hi_=0.94):
    cols = plt.get_cmap(base)(np.linspace(lo_, hi_, len(bounds) + 1))
    cm = ListedColormap(cols[1:-1]); cm.set_under(cols[0]); cm.set_over(cols[-1]); return cm, BoundaryNorm(bounds, cm.N)


W = 180; MW = 55.0; MH = MW * (EXT[3] - EXT[2]) / (EXT[1] - EXT[0]); H = MH + 24.5
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
PAN = [('oper|heat|WIDE|msl|mean', 'PuOr', np.arange(-6, 0.01, 1.0), 'Sea-level pressure, with storm\nminus storm removed (hPa)'),
       ('oper|heat|WIDE|tp6|mean', 'BrBG', np.array([-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2]), 'Rainfall in the 6 h before 14 LT, with\nstorm minus storm removed (mm)'),
       ('oper|heat|WIDE|t850|mean', 'RdBu_r', np.arange(-2.5, 2.51, 0.5), '850-hPa temperature, with storm minus\nstorm removed (K); green: humidity (±0.5,\n1, 2 g kg$^{-1}$; dashed: drier)')]
for n, (k, cmn, bnd, lab) in enumerate(PAN):
    x0 = 6.5 + n * (MW + 5.5)
    ax = fig.add_axes(box(x0, H - 4.5 - MH, MW, MH), projection=pc)
    ax.set_extent(EXT, crs=pc); ax.set_xticks([]); ax.set_yticks([])
    if n == 0:
        cm, nm = disc('PuOr', np.arange(-6, 0.01, 1.0), 0.02, 0.5)
    else:
        cm, nm = disc(cmn, bnd)
    ax.contourf(lo, la, get(k), levels=bnd, cmap=cm, norm=nm, extend='both' if n else 'min', transform=pc, zorder=1)
    if n == 0:
        c = ax.contour(lo, la, get('oper|heat|CTL|msl|mean'), levels=np.arange(-12, 6, 1.0), colors='k', linewidths=0.45, transform=pc, zorder=2)
        ax.clabel(c, levels=[l for l in c.levels[::2] if l != -4], fontsize=6, fmt='%d', inline_spacing=1)   # -4 sits under the storm positions
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.45, edgecolor='#333333', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.3, edgecolor='#666666', zorder=3)
    ax.scatter(pos.lon, pos.lat, s=8, facecolor='none', edgecolor='k', lw=0.45, transform=pc, zorder=5)
    ax.scatter(stn.lon, stn.lat, s=6, marker='^', color='k', lw=0, transform=pc, zorder=5)
    gl = ax.gridlines(draw_labels=True, lw=0.25, color='#9E9E9E', ls=':', xlocs=range(105, 131, 5), ylocs=range(15, 31, 5), zorder=4)
    gl.top_labels = gl.right_labels = False; gl.left_labels = (n == 0)
    gl.xlabel_style = gl.ylabel_style = dict(size=6.5); gl.xpadding = gl.ypadding = 2
    cax = fig.add_axes(box(x0 + 5, 11.5, MW - 10, 1.8))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation='horizontal', extend='both' if n else 'min', ticks=bnd)
    cb.set_label(lab, fontsize=6.5, labelpad=1, linespacing=1.0); cb.ax.tick_params(labelsize=6.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
    if n == 1: cb.ax.set_xticklabels(['−2', '−1', '−0.5', '−0.25', '0.25', '0.5', '1', '2'])
    if n == 2:
        cb.set_ticks(bnd[::2]); cb.ax.set_xticklabels(['−2.5', '−1.5', '−0.5', '0.5', '1.5', '2.5'])
        ql = [-2, -1, -0.5, 0.5, 1, 2]
        q8 = get('oper|heat|WIDE|q850|mean')            # already in g/kg (gc_run.py stores specific humidity x 1000)
        ax.contour(lo, la, q8, levels=ql, colors='#1B5E20', linewidths=0.6, linestyles=['--' if l_ < 0 else '-' for l_ in ql],
                   transform=pc, zorder=4)
        print('850-hPa humidity change (g/kg): min', round(float(np.nanmin(q8)), 2), 'max', round(float(np.nanmax(q8)), 2),
              'mean over the stations box', round(float(np.nanmean(q8[(la[:, None] >= 21) & (la[:, None] <= 25.5) & (lo[None, :] >= 109.5) & (lo[None, :] <= 117.5)])), 2))
    fig.text((x0 - (6.0 if n == 0 else 4.6)) / W, 1 - 0.6 / H, 'abc'[n], fontsize=9, fontweight='bold', va='top', ha='left')
os.makedirs(OUT, exist_ok=True)
print('QA:', overlap_check(fig))
fig.savefig(f'{OUT}/FigS8.png', dpi=600); fig.savefig(f'{OUT}/FigS8.pdf')
