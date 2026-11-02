"""Fig. 2: observed afternoon heat in Guangdong with a distant tropical cyclone (input: 10_obs_gd.py output dir)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.patches import Wedge, Circle
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import gaussian_kde

OBS = os.environ.get('OBS', 'data/obs')
OUT = os.environ.get('FIGOUT', 'fig')
GD = (113.5, 23.0)
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

rose = pd.read_csv(f'{OBS}/rose.csv'); stn = pd.read_csv(f'{OBS}/stations.csv')
ons = pd.read_csv(f'{OBS}/onset.csv'); share = pd.read_csv(f'{OBS}/hot_share.csv')
fp = pd.read_csv(f'{OBS}/fingerprints.csv').set_index('label'); K = np.load(f'{OBS}/kde.npz')

bounds = np.arange(-2.5, 2.51, 0.5)
cols = plt.get_cmap('RdBu_r')(np.linspace(0.06, 0.94, len(bounds) + 1))
cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1]); norm = BoundaryNorm(bounds, cmap.N)

fig = plt.figure(figsize=(180 * MM, 104 * MM))
proj = ccrs.AzimuthalEquidistant(central_longitude=GD[0], central_latitude=GD[1])
R = 1450e3
axa = fig.add_axes([0.012, 0.115, 0.485, 0.845], projection=proj)
axa.set_extent([-R, R, -R, R], crs=proj)
axa.add_feature(cfeature.LAND.with_scale('50m'), facecolor=LAND, edgecolor='none', zorder=0)
axa.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor=SEA, edgecolor='none', zorder=0)
for _, r in rose.iterrows():
    b0, b1 = map(float, str(r.band).split('-')); k = SECT.index(r.sector)
    th_c = 90 - 45 * k                                     # bearing -> math angle
    sig = np.isfinite(r.get('p', np.nan)) and r.p < 0.05
    if not np.isfinite(r.get('excess', np.nan)):
        w = Wedge((0, 0), b1 * 1e3, th_c - 22.5, th_c + 22.5, width=(b1 - b0) * 1e3, facecolor='none', edgecolor='#B0B0B0', lw=0.4, zorder=2)
        axa.add_patch(w); continue
    c = cmap(norm(r.excess))
    w = Wedge((0, 0), b1 * 1e3, th_c - 22.5, th_c + 22.5, width=(b1 - b0) * 1e3, facecolor=c, alpha=0.88 if sig else 0.45,
              edgecolor='white', lw=0.6, zorder=2, hatch=None if sig else '////')
    axa.add_patch(w)
    rm = 0.5 * (b0 + b1) * 1e3; x, y = rm * np.cos(np.radians(th_c)), rm * np.sin(np.radians(th_c))
    axa.text(x, y, f'{r.excess:+.1f}', ha='center', va='center', fontsize=6.5, zorder=6, path_effects=HALO)
axa.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4, edgecolor='#4D4D4D', zorder=4)
axa.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.3, edgecolor='#8C8C8C', zorder=4)
for rr in (300, 700, 1300):
    axa.add_patch(Circle((0, 0), rr * 1e3, facecolor='none', edgecolor='#333333', lw=0.5, ls=(0, (3, 2)), zorder=5))
    axa.text(rr * 1e3 * np.cos(np.radians(-112.5)), rr * 1e3 * np.sin(np.radians(-112.5)), f'{rr:,} km', fontsize=6, ha='center', va='center',
             color='#333333', zorder=6, path_effects=HALO)
sc = axa.scatter(stn.lon, stn.lat, c=stn.excess, cmap=cmap, norm=norm, s=9, edgecolor='k', linewidths=0.35, transform=ccrs.PlateCarree(), zorder=7)
for s, (dx, dy) in (('N', (0, 1)), ('E', (1, 0)), ('S', (0, -1)), ('W', (-1, 0))):
    axa.text(dx * 1.385e6, dy * 1.385e6, s, fontsize=7, ha='center', va='center', fontweight='normal', color='#333333', zorder=6)
axa.set_xticks([]); axa.set_yticks([])
gl = axa.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=':', xlocs=range(95, 140, 10), ylocs=range(5, 45, 10), zorder=1)
cax = fig.add_axes([0.09, 0.075, 0.33, 0.022])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.set_label('Daily maximum temperature above circulation analogues (K)', fontsize=7, labelpad=1.5); cb.ax.tick_params(labelsize=6.5, length=2)
cb.outline.set_linewidth(0.4)
axa.text(0.015, 0.985, 'Hatched: P ≥ 0.05\nOutline only: < 15 days', transform=axa.transAxes, fontsize=6.5, va='top', ha='left', zorder=8,
         bbox=dict(facecolor='white', edgecolor='none', pad=1))

# b: onset composite
axb = fig.add_axes([0.585, 0.665, 0.405, 0.30])
for g, col, lab in (('east', RED, 'Storm to the east'), ('southwest', BLUE, 'Storm to the southwest')):
    o = ons[ons.group == g]
    if o['mean'].notna().sum() == 0: continue
    axb.fill_between(o.lag, o.lo, o.hi, color=col, alpha=0.18, lw=0)
    axb.plot(o.lag, o['mean'], color=col, lw=1.2, marker='o', ms=2.5, label=f'{lab} (n = {int(o.n.iloc[0])})')
axb.axhline(0, color='k', lw=0.4); axb.axvline(0, color=GREY, lw=0.5, ls='--')
axb.set_xlabel('Days since the storm entered the sector', labelpad=1)
axb.set_ylabel('Tmax anomaly (K)'); axb.set_xticks(range(-5, 4)); axb.set_xlim(-5.3, 3.3)
axb.legend(loc='lower left', frameon=False, fontsize=6.5, handlelength=1.4, borderaxespad=0.1)
for sp in ('top', 'right'): axb.spines[sp].set_visible(False)

# c: dry heat: analogue-controlled excess of temperature and humidity measures on east-side cyclone days
axc = fig.add_axes([0.6, 0.105, 0.15, 0.42])
rows = [('Tmax', 'Tmax'), ('T850', '850-hPa\ntemperature'), ('T_14LT', 'Temperature,\n14 LT'), ('Td_14LT', 'Dew point,\n14 LT'),
        ('Tw_14LT', 'Wet bulb,\n14 LT'), ('Tw_at_Tmax', 'Wet bulb at\nhottest hour'), ('Tw_daily_max', 'Highest wet\nbulb of day')]
yy = np.arange(len(rows))[::-1]
for yi, (v, lab) in zip(yy, rows):
    r = fp.loc[f'east {v}']; sig = r.p < 0.05
    col = (RED if r.excess > 0 else BLUE) if sig else '#BDBDBD'
    axc.barh(yi, r.excess, height=0.6, color=col, edgecolor='none')
    axc.errorbar(r.excess, yi, xerr=[[r.excess - r.lo], [r.hi - r.excess]], fmt='none', ecolor='k', elinewidth=0.6, capsize=1.5)
axc.set_yticks(yy); axc.set_yticklabels([l for _, l in rows], fontsize=6.5, linespacing=0.9); axc.set_ylim(-0.6, len(rows) - 0.4)
axc.axvline(0, color='k', lw=0.5); axc.set_xlim(-1.5, 2.2); axc.set_xticks([-1, 0, 1, 2]); axc.set_xlabel('Excess over analogues (K)', labelpad=1)
on = fp.loc['east onshore_share']; cs = fp.loc['east crossshore_14LT']
axc.text(0.99, 0.015, f'Onshore wind\n{100 * (on.raw - on.excess):.0f}% → {100 * on.raw:.0f}%\nof hours\n({cs.excess:+.1f} m s$^{{-1}}$\nat 14 LT)'.replace('-', '−'),
         transform=axc.transAxes, fontsize=6.5, ha='right', va='bottom', linespacing=1.08)
for sp in ('top', 'right'): axc.spines[sp].set_visible(False)

# d: share of the hottest days with a cyclone to the east
axd = fig.add_axes([0.835, 0.105, 0.14, 0.42])
y = np.arange(len(share))[::-1]
for yi, (_, r) in zip(y, share.iterrows()):
    axd.plot([100 * r.expected, 100 * r.observed], [yi, yi], color='#BDBDBD', lw=2.2, solid_capstyle='round', zorder=1)
    axd.errorbar(100 * r.observed, yi, xerr=[[100 * (r.observed - r.obs_lo)], [100 * (r.obs_hi - r.observed)]], fmt='none', ecolor=RED, elinewidth=0.6, capsize=1.5, zorder=2)
    axd.scatter(100 * r.expected, yi, s=16, color=GREY, zorder=3); axd.scatter(100 * r.observed, yi, s=16, color=RED, zorder=3)
    axd.text(100 * r.obs_hi + 1.0, yi, f'×{r.ratio:.1f}', va='center', fontsize=6.5, color=RED)
axd.set_yticks(y); axd.set_yticklabels([p.replace('-', '–\n') for p in share.period], fontsize=6.5); axd.set_ylim(-0.6, len(share) - 0.4)
axd.set_xlim(0, 50); axd.set_xticks([0, 10, 20, 30, 40, 50]); axd.set_xlabel('Hottest 5% of days with\na storm to the east (%)', labelpad=1, fontsize=7)
axd.text(0.5, 0.97, 'Expected', transform=axd.transAxes, color='#6E6E6E', fontsize=6.5, va='top', ha='right')
axd.text(0.55, 0.97, 'Observed', transform=axd.transAxes, color=RED, fontsize=6.5, va='top', ha='left')
for sp in ('top', 'right'): axd.spines[sp].set_visible(False)

for ax, s in ((axa, 'a'), (axb, 'b'), (axc, 'c'), (axd, 'd')):
    bb = ax.get_position()
    fig.text(bb.x0 - (0.008 if ax is axa else (0.085 if ax is axc else 0.06)), bb.y1 + 0.035, s, fontsize=9, fontweight='bold', va='top', ha='left')
issues = overlap_check(fig)
print('QA:', issues if issues else 'clean')
fig.savefig(f'{OUT}/FigS3.png', dpi=600); fig.savefig(f'{OUT}/FigS3.pdf')
print('saved', f'{OUT}/FigS3.png')
