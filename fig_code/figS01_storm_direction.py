"""Fig. S1: direction of the warmest storm sector, hot-day ratio and the evolution as the storm approaches
(ERA5 1980-2021, IBTrACS; gl_map.py and gl_stage2.py).  Inputs: GL (gl_fig.npz, gl_regions.csv), S2 (gl_s2_onset.csv)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Wedge

GL = os.environ.get('GL', 'data/global')
S2 = os.environ.get('S2', 'data/global_s2')
OUT = os.environ.get('FIGOUT', 'fig')
GD = (113.5, 23.0)
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
z = np.load(f'{GL}/gl_fig.npz')
q = lambda k, s=0.05: np.where(z[k] == -128, np.nan, z[k].astype(float) * s)
lon, lat = z['lon'], z['lat']
RG = pd.read_csv(f'{GL}/gl_regions.csv'); W_ = RG[RG.sign == 'warm'].reset_index(drop=True)
NS, NOEXP = '#BDBDBD', '#ECECEC'
# eight directions on a colour wheel (N at the top, clockwise)
SCOL = ['#3B6FB6', '#2BA6A0', '#7CB342', '#E0B32B', '#E07B39', '#C44E52', '#9C4F96', '#5E5BB0']

pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
W = 180.0; MW = 175.0; MH = MW * (Y1 - Y0) / (X1 - X0)
ons = pd.read_csv(f'{S2}/gl_s2_onset.csv') if os.path.exists(f'{S2}/gl_s2_onset.csv') else None
HC = 40.0 if ons is not None else 0.0
H = 3.0 + MH + 2.0 + MH + (HC if HC else 1.0)
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()
lone = np.r_[lon - 0.5, lon[-1] + 0.5]; late = np.clip(np.r_[lat - 0.5, lat[-1] + 0.5], -90, 90)


def base(ax, tag):
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=NOEXP, edgecolor='none', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#555555', zorder=3)
    ax.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=(0, (2, 2)), xlocs=range(-180, 181, 30), ylocs=[-30, 0, 30], zorder=0.5)
    for la_ in (-30, 0, 30):
        x, y = proj.transform_point(-168, la_, pc)
        ax.text(x, y, {-30: '30° S', 0: '0°', 30: '30° N'}[la_], fontsize=6.5, color='#666666', ha='left', va='bottom', zorder=6)
    ax.plot(*GD, marker='*', ms=6, mfc='none', mec='k', mew=0.6, transform=pc, zorder=7)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.text(0.006, 0.975, tag, transform=ax.transAxes, fontsize=7, ha='left', va='top', zorder=8, path_effects=HALO)


def at(ax, lo_, la_):
    return inv.transform(ax.transData.transform(proj.transform_point(lo_, la_, pc)))


# a: direction of the warmest storm sector (significant warming only)
axa = fig.add_axes(box(4.5, H - 3.0 - MH, MW, MH), projection=proj)
base(axa, 'Direction of the TC on the warmest days')
eb, sec, sig = q('ex_best'), z['sec_best'].astype(float), z['sig_best']
ok = np.isfinite(eb)
axa.pcolormesh(lone, late, np.ma.masked_where(~(ok & ~((sig == 1) & (eb > 0))), np.ones_like(eb)).T, cmap=ListedColormap([NS]),
               vmin=0, vmax=2, transform=pc, zorder=1, rasterized=True)
axa.pcolormesh(lone, late, np.ma.masked_where(~(ok & (sig == 1) & (eb > 0)), sec).T, cmap=ListedColormap(SCOL),
               norm=BoundaryNorm(np.arange(-0.5, 8), 8), transform=pc, zorder=2, rasterized=True)
# compass key in the empty southeast Pacific
c0 = at(axa, -112, -16); r = 7.0
kax = fig.add_axes([c0[0] - r / W, c0[1] - r / H, 2 * r / W, 2 * r / H]); kax.set_xlim(-1.45, 1.45); kax.set_ylim(-1.45, 1.45); kax.axis('off')
kax.set_aspect('equal')
for k in range(8):
    th = 90 - 45 * k
    kax.add_patch(Wedge((0, 0), 1.0, th - 22.5, th + 22.5, width=0.5, facecolor=SCOL[k], edgecolor='white', lw=0.5))
    if k % 2 == 0:
        kax.text(1.3 * np.cos(np.radians(th)), 1.3 * np.sin(np.radians(th)), SECT[k], fontsize=6.5, ha='center', va='center')

# b: hot-day ratio
axb = fig.add_axes(box(4.5, H - 3.0 - 2 * MH - 2.0, MW, MH), projection=proj)
base(axb, 'Hottest 5% of days: share with a TC, observed over expected')
hr = q('hot_ratio')
hb = np.array([0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0])
hc = plt.get_cmap('PuOr_r')(np.linspace(0.08, 0.92, len(hb) + 1))
hcm = ListedColormap(hc[1:-1]); hcm.set_under(hc[0]); hcm.set_over(hc[-1]); hn = BoundaryNorm(hb, hcm.N)
axb.pcolormesh(lone, late, np.ma.masked_invalid(hr).T, cmap=hcm, norm=hn, transform=pc, zorder=2, rasterized=True)
q0, q1 = at(axb, 188, -14), at(axb, 255, -14)
cax = fig.add_axes([q0[0], q0[1], q1[0] - q0[0], 1.6 / H])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=hn, cmap=hcm), cax=cax, orientation='horizontal', extend='both', ticks=hb)
cb.set_label('Observed over expected share', fontsize=6.5, labelpad=1)
cb.ax.set_xticklabels(['0.5', '0.67', '0.8', '1', '1.25', '1.5', '2', '3'])
cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)

letters = [('a', 0.3, H - 2.6), ('b', 0.3, H - 3.0 - MH - 2.6)]
# c: evolution as the storm approaches, warm regions with at least 30 approaches; d: rise per region
if ons is not None:
    axc = fig.add_axes(box(14.0, 8.5, 72.0, HC - 10.0))
    o = ons[ons.region.str.startswith('W')]
    nmax = o.groupby('region').n.first()
    keep = nmax[nmax >= 30].index
    for i, reg in enumerate(keep):
        g = o[o.region == reg]
        axc.plot(g.lag, g['mean'], color='#E3A3A5', lw=0.5, label='Single regions' if i == 0 else None, zorder=2)
    allm = o[o.region.isin(keep)].groupby('lag')['mean'].mean()
    axc.plot(allm.index, allm.values, color='k', lw=1.2, marker='o', ms=2.5, label=f'Mean of {len(keep)} regions', zorder=4)
    g = ons[ons.region == 'GD_E_SE']
    if len(g):
        axc.plot(g.lag, g['mean'], color=BLUE, lw=1.0, ls=(0, (3, 1.5)), label='Guangdong grid points', zorder=3)
    axc.axhline(0, color='k', lw=0.4); axc.axvline(0, ymax=0.74, color=GREY, lw=0.5, ls='--')
    axc.set_ylim(-0.85, 1.75); axc.set_yticks([-0.5, 0, 0.5, 1.0])
    axc.set_xticks(range(-5, 4)); axc.set_xlabel('Days since the TC came within 1,300 km', labelpad=1); axc.set_ylabel('Tmax anomaly (K)')
    axc.legend(loc='upper left', frameon=False, fontsize=6.5, handlelength=1.6, borderaxespad=0.1, ncol=2, columnspacing=1.0)
    for sp in ('top', 'right'): axc.spines[sp].set_visible(False)

    axd = fig.add_axes(box(101.0, 8.5, 77.0, HC - 10.0))
    s2 = pd.read_csv(f'{S2}/gl_s2_regions.csv'); w2 = s2[s2.region.str.startswith('W')].reset_index(drop=True)
    xs = np.arange(1, len(w2) + 1); sg = (w2.onset_rise_lo > 0).values
    axd.vlines(xs, w2.onset_rise_lo, w2.onset_rise_hi, color=RED, lw=0.8)
    axd.scatter(xs[sg], w2.onset_rise[sg], s=11, color=RED, zorder=3, label='95% interval above 0')
    axd.scatter(xs[~sg], w2.onset_rise[~sg], s=11, facecolor='white', edgecolor=RED, lw=0.7, zorder=3, label='Interval includes 0')
    axd.axhline(0, color='k', lw=0.4)
    axd.set_xlim(0.4, len(w2) + 0.6); axd.set_xticks(xs); axd.set_xticklabels([str(x) for x in xs], fontsize=6.5)
    axd.set_ylim(-1.7, 2.3); axd.set_yticks([-1, 0, 1, 2])
    axd.set_xlabel('Warm region (numbered as in Fig. 1a)', labelpad=1); axd.set_ylabel('Rise on approach (K)')
    axd.legend(loc='upper left', frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.1, ncol=2, columnspacing=1.0)
    for sp in ('top', 'right'): axd.spines[sp].set_visible(False)
    letters += [('c', 0.3, HC - 0.5), ('d', 90.0, HC - 0.5)]
for s, x, y in letters:
    fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top', ha='left')
os.makedirs(OUT, exist_ok=True)
print('QA:', overlap_check(fig))
fig.savefig(f'{OUT}/FigS1.png', dpi=600); fig.savefig(f'{OUT}/FigS1.pdf')
