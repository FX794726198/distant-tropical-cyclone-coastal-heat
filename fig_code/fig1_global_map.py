"""v10 Fig. 1: heat with distant tropical storms on land worldwide (ERA5 1980-2021, IBTrACS; gl_map.py, gl_stage2.py;
ISD-Lite stations, 12_isd_hourly.py).  From v09 Fig. 1 (a-d unchanged) with panel e replaced: v09 e (land/sea fetch
histogram) goes to the SI, and e is now the station check of the map by region (v09 Fig. S8e).
Inputs: GL (gl_fig.npz, gl_regions.csv, gl_placebo.csv), S2 (gl_s2_regions.csv), ISD (isd_regions.csv)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

P9 = os.environ.get('P9', '.')
GL = os.environ.get('GL', f'{P9}/data/global')
S2 = os.environ.get('S2', f'{P9}/data/global_s2')
ISD = os.environ.get('ISD', f'{P9}/data/isd')
OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
import matplotlib.lines as mlines
GD = (113.5, 23.0)
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
# warm regions in the order of gl_regions.csv (west to east from 20 deg E): short name, label position (lon, lat)
NAMES = [('NE South Africa', (35, -31)), ('Inland N Mozambique', (31, -9)), ('Coastal N Mozambique', (44.5, -10)), ('Madagascar', (55, -22)),
         ('SE India', (73, 6)), ('Bangladesh', (91, 16.5)), ('Southern China', (101, 39)), ('W Australia', (107, -25)),
         ('E Borneo', (122, 3.5)), ('Korea and NE China', (117, 45)), ('N Australia', (133, -31)), ('W Japan', (131, 27.5)),
         ('Central Japan', (147, 40)), ('S New Guinea', (143, -0.5)), ('Central Queensland', (156, -27)), ('NE Queensland', (153.5, -15.5)),
         ('NW Mexico', (-117, 17)), ('E Mexico', (-91, 26.5)), ('SE United States', (-99, 41)), ('Belize and Yucatan', (-82.5, 21.5)),
         ('Nicaragua and Honduras', (-79, 12)), ('NE United States', (-85, 48.5)), ('Maritime Canada', (-57, 41))]
GROUPS = [('Africa', 1, 4), ('S Asia', 5, 6), ('E Asia, Australia, New Guinea', 7, 16), ('North America', 17, 23)]

z = np.load(f'{GL}/gl_fig.npz')
q = lambda k, s=0.05: np.where(z[k] == -128, np.nan, z[k].astype(float) * s)
lon, lat = z['lon'], z['lat']
RG = pd.read_csv(f'{GL}/gl_regions.csv'); W_ = RG[RG.sign == 'warm'].reset_index(drop=True)
assert len(W_) == len(NAMES)
PL = pd.read_csv(f'{GL}/gl_placebo.csv')
s2 = pd.read_csv(f'{S2}/gl_s2_regions.csv') if os.path.exists(f'{S2}/gl_s2_regions.csv') else None

bounds = np.arange(-2.5, 2.51, 0.5)
cols = plt.get_cmap('RdBu_r')(np.linspace(0.06, 0.94, len(bounds) + 1))
cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1]); norm = BoundaryNorm(bounds, cmap.N)
NS, NOEXP = '#BDBDBD', '#ECECEC'

pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
W = 180.0; MW = 175.0; MH = MW * (Y1 - Y0) / (X1 - X0)
DH = 38.0
H = 3.0 + MH + 2.0 + MH + 5.0 + DH + 19.0
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
lone = np.r_[lon - 0.5, lon[-1] + 0.5]; late = np.clip(np.r_[lat - 0.5, lat[-1] + 0.5], -90, 90)


def draw_map(ax, ex, sig, tag):
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_xticks([]); ax.set_yticks([])
    ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=NOEXP, edgecolor='none', zorder=0)
    valid = np.isfinite(ex)
    ax.pcolormesh(lone, late, np.ma.masked_where(~(valid & (sig != 1)), np.ones_like(ex)).T, cmap=ListedColormap([NS]),
                  vmin=0, vmax=2, transform=pc, zorder=1, rasterized=True)
    ax.pcolormesh(lone, late, np.ma.masked_where(~(valid & (sig == 1)), ex).T, cmap=cmap, norm=norm, transform=pc, zorder=2, rasterized=True)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#555555', zorder=3)
    ax.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=(0, (2, 2)), xlocs=range(-180, 181, 30), ylocs=[-30, 0, 30], zorder=0.5)
    for la_ in (-30, 0, 30):
        x, y = proj.transform_point(-168, la_, pc)
        ax.text(x, y, {-30: '30° S', 0: '0°', 30: '30° N'}[la_], fontsize=6.5, color='#666666', ha='left', va='bottom', zorder=6)
    ax.plot(*GD, marker='*', ms=6, mfc='none', mec='k', mew=0.6, transform=pc, zorder=7)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.text(0.006, 0.975, tag, transform=ax.transAxes, fontsize=7, ha='left', va='top', zorder=8, path_effects=HALO)


# a: warmest storm direction, numbered warm regions
axa = fig.add_axes(box(4.5, H - 3.0 - MH, MW, MH), projection=proj)
draw_map(axa, q('ex_best'), z['sig_best'], 'TC in the warmest direction')
for k, r in W_.iterrows():
    lx, ly = NAMES[k][1]
    axa.plot([r.lon, lx], [r.lat, ly], color='k', lw=0.35, transform=pc, zorder=6)
    axa.text(lx, ly, str(k + 1), fontsize=6.5, ha='center', va='center', transform=pc, zorder=8, path_effects=HALO)

axa.text(116.0, 21.2, 'Guangdong', fontsize=6, ha='left', va='top', transform=pc, zorder=8, path_effects=HALO)

# c (inset in a, empty southeast Pacific): placebo test
p0 = axa.transData.transform(proj.transform_point(-133, -27, pc)); p1 = axa.transData.transform(proj.transform_point(-90, 6, pc))
inv = fig.transFigure.inverted(); (fx0, fy0), (fx1, fy1) = inv.transform(p0), inv.transform(p1)
axc = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
real, pla = PL.iloc[0], PL.iloc[1:]
mets = [('frac_sig_any', 'Any significant change'), ('frac_best_ge05_sig', 'Warmer by ≥0.5 K')]
for i, (m, lab) in enumerate(mets):
    y0 = 1.6 - 1.25 * i
    axc.barh(y0, 100 * real[m], 0.32, color=RED, lw=0)
    axc.barh(y0 - 0.36, max(100 * pla[m].mean(), 0.4), 0.32, color=GREY, lw=0)
    axc.text(100 * real[m] + 1.5, y0, f'{100 * real[m]:.0f}%', ha='left', va='center', fontsize=6.5, color=RED)
    axc.text(3.0, y0 - 0.36, f'≤{100 * pla[m].max():.1f}%', ha='left', va='center', fontsize=6.5, color='#555555')
    axc.text(1.5, y0 + 0.24, lab, ha='left', va='bottom', fontsize=6.5)
axc.set_xlim(0, 62); axc.set_ylim(-0.25, 2.85); axc.set_yticks([]); axc.set_xticks([0, 25, 50])
axc.set_xlabel('Grid points (%)', fontsize=6.5, labelpad=1); axc.tick_params(labelsize=6.5, length=2, pad=1)
for sp in ('top', 'right', 'left'): axc.spines[sp].set_visible(False)
axc.axvline(0, color='k', lw=0.5)
axc.patch.set_alpha(0)
axc.text(62, 2.85, 'Real tracks', color=RED, fontsize=6.5, ha='right', va='top')
axc.text(62, 2.5, 'Tracks shifted by 1–5 yr', color='#555555', fontsize=6.5, ha='right', va='top')

# b: coolest storm direction
axb = fig.add_axes(box(4.5, H - 3.0 - 2 * MH - 2.0, MW, MH), projection=proj)
draw_map(axb, q('ex_worst'), z['sig_worst'], 'TC in the coolest direction')
# shared colour bar and grey keys in the empty southern Indian Ocean of b
q0 = axb.transData.transform(proj.transform_point(57, -13, pc)); q1 = axb.transData.transform(proj.transform_point(103, -13, pc))
(cx0, cy0), (cx1, _) = inv.transform(q0), inv.transform(q1)
cax = fig.add_axes([cx0, cy0, cx1 - cx0, 1.6 / H])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.set_label('Tmax on TC days minus\nmatched TC-free days (K)', fontsize=6.5, labelpad=1, linespacing=0.95)
cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels(['−2.5', '−1.5', '−0.5', '0.5', '1.5', '2.5'])
k0 = inv.transform(axb.transData.transform(proj.transform_point(66, -34, pc)))
for i, (c, lab) in enumerate(((NS, 'Not significant'), (NOEXP, 'Few TC days'))):
    fig.patches.append(Rectangle((k0[0], k0[1] + (1 - i) * 2.6 / H), 2.2 / W, 1.6 / H, transform=fig.transFigure, facecolor=c,
                                 edgecolor='#888888', lw=0.3, zorder=9))
    fig.text(k0[0] + 3.0 / W, k0[1] + (1 - i) * 2.6 / H + 0.8 / H, lab, fontsize=6.5, va='center', ha='left', zorder=9)

# d: warm regions, warming over circulation analogues (stage 2) against the stage-1 map value
yb = 19.0
axd = fig.add_axes(box(13.0, yb, 106.0, DH))
xs = np.arange(1, len(W_) + 1); OFF = 0.14; xo = xs + OFF
axd.scatter(xs - OFF, W_.ex_mean, marker='D', s=9, facecolor='none', edgecolor=GREY, lw=0.6, zorder=3, label='Map value (a)')
if s2 is not None:
    w2 = s2[s2.sign == 'warm'].reset_index(drop=True)
    sig = w2.tmax_sig_fdr.astype(str).str.lower().isin(['true', '1', '1.0']).values & (w2.tmax_lo.values > 0)   # FDR and 95% percentile interval above zero
    axd.vlines(xo, w2.tmax_lo, w2.tmax_hi, color=RED, lw=0.8, zorder=2)
    axd.scatter(xo[sig], w2.tmax_ex[sig], s=12, color=RED, zorder=4, label='Over circulation analogues')
    axd.scatter(xo[~sig], w2.tmax_ex[~sig], s=12, facecolor='white', edgecolor=RED, lw=0.7, zorder=4)
axd.axhline(0, color='k', lw=0.4)
axd.set_xlim(0.4, len(W_) + 0.6); axd.set_xticks(xs); axd.set_xticklabels([str(x) for x in xs], fontsize=6.5)
axd.set_ylabel('Warming (K)')
leg1 = axd.legend(loc='upper left', frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.1, ncol=1)
if s2 is not None:
    axd.add_artist(leg1)
for sp in ('top', 'right'): axd.spines[sp].set_visible(False)
# fingerprint strips under the axis: 850-hPa temperature and specific humidity over the analogues
if s2 is not None:
    tr_ = axd.get_xaxis_transform()
    for yy, v, lo_, hi_, up, dn, lab in ((-0.145, w2.t850_ex, w2.t850_lo, w2.t850_hi, RED, BLUE, '$T_{850}$'),
                                         (-0.225, -w2.q850_ex, w2.q850_lo, w2.q850_hi, '#B07A2A', '#2A8C82', '$q_{850}$')):
        sg = ((lo_ > 0) | (hi_ < 0)).values                     # 95% percentile interval excludes zero
        fc = np.where(~sg, 'white', np.where(v.values > 0, up, dn))
        axd.scatter(xs, np.full(len(xs), yy), s=13, marker='s', c=list(fc), edgecolors=np.where(sg, 'none', '#888888'), linewidths=0.4,
                    transform=tr_, clip_on=False, zorder=5)
        axd.text(0.15, yy, lab, transform=tr_, ha='right', va='center', fontsize=6.5)
    from matplotlib.lines import Line2D
    kh = [Line2D([], [], ls='none', marker='s', ms=3.6, mfc=c, mec=e, mew=0.4) for c, e in ((RED, 'none'), ('#B07A2A', 'none'), ('#2A8C82', 'none'), ('white', '#888888'))]
    leg2 = axd.legend(kh, ['Warmer', 'Drier', 'Moister', 'Interval includes 0'], loc='upper right', ncol=1, frameon=False, fontsize=6.5, handletextpad=0.1,
                      labelspacing=0.25, borderaxespad=0.1, title='Air at 850 hPa over the analogues', title_fontsize=6.5, alignment='left')
    axd.set_ylim(-0.6, 3.95); axd.set_yticks(np.arange(0, 3.01, 1.0))
for g, a_, b_ in GROUPS:
    yb_ = -0.30
    axd.plot([a_ - 0.35, b_ + 0.35], [yb_, yb_], color='#555555', lw=0.5, transform=axd.get_xaxis_transform(), clip_on=False)
    axd.text(0.5 * (a_ + b_), yb_ - 0.035, g, fontsize=6.5, ha='center', va='top', transform=axd.get_xaxis_transform(), color='#333333')

# e: stations against the map, region by region (ISD-Lite; storm in the region's direction, region's season)
axe = fig.add_axes(box(132.0, yb, 46.0, DH))
RT = pd.read_csv(f'{ISD}/isd_regions.csv'); RT = RT[(RT.n_storm >= 20)].dropna(subset=['Tmax']).copy()
fmt = lambda v: f'{v:g}'.replace('-', '−')
axe.plot([-2.4, 2.4], [-2.4, 2.4], color='#999999', lw=0.6, ls=(0, (3, 2)), zorder=1)
axe.axhline(0, color='#555555', lw=0.4, zorder=1)
for _, g in RT.iterrows():
    col = RED if g.sign == 'warm' else BLUE; low = abs(g.lat) < 30
    axe.vlines(g.ex_map, g.Tmax_lo, g.Tmax_hi, color=col, lw=0.5, zorder=2, alpha=0.6)
    axe.plot(g.ex_map, g.Tmax, marker='o' if g.sign == 'warm' else 's', ms=3.2, mfc=col if low else 'white', mec=col,
             mew=0.5 if low else 0.8, ls='none', zorder=3)
    if (g.Tmax > 0) != (g.sign == 'warm'):                         # the two regions where the stations disagree in sign
        axe.text(g.ex_map + 0.12, g.Tmax, g.region[1:], fontsize=6, ha='left', va='center', color='#555555', zorder=4)
ok = (RT.Tmax > 0) == (RT.sign == 'warm'); lo_ = RT.lat.abs() < 30
axe.text(0.03, 0.99, f'{int(RT.n_st.sum())} stations\nSame sign in {int(ok.sum())} of {len(RT)} regions\n$r$ = {RT[lo_].Tmax.corr(RT[lo_].ex_map):.2f} (<30°), '
         f'{RT[~lo_].Tmax.corr(RT[~lo_].ex_map):.2f} (≥30°)', transform=axe.transAxes, ha='left', va='top', fontsize=6.3, linespacing=1.3)
axe.set_xlim(-2.4, 2.2); axe.set_ylim(-2.9, 2.6); axe.set_xticks([-2, -1, 0, 1, 2]); axe.set_yticks([-2, -1, 0, 1, 2])
axe.set_xticklabels([fmt(v) for v in (-2, -1, 0, 1, 2)]); axe.set_yticklabels([fmt(v) for v in (-2, -1, 0, 1, 2)])
axe.set_xlabel('Map, ERA5 (K)', labelpad=1); axe.set_ylabel('Stations, Tmax (K)', labelpad=1)
he = [mlines.Line2D([], [], marker='o', ms=3.2, mfc=RED, mec=RED, ls='none'), mlines.Line2D([], [], marker='s', ms=3.2, mfc=BLUE, mec=BLUE, ls='none'),
      mlines.Line2D([], [], marker='o', ms=3.2, mfc='white', mec='#555555', mew=0.8, ls='none')]
lg = axe.legend(he, ['Warm region', 'Cool region', '≥30° (open)'], loc='upper left', bbox_to_anchor=(0.0, 0.78), frameon=False, fontsize=6.0,
                handletextpad=0.05, borderaxespad=0.1, labelspacing=0.15, borderpad=0.1)
for sp in ('top', 'right'): axe.spines[sp].set_visible(False)

for s, x, y in (('a', 0.3, H - 2.6), ('b', 0.3, H - 3.0 - MH - 2.6), ('c', None, None), ('d', 0.5, yb + DH + 3.5), ('e', 121.5, yb + DH + 3.5)):
    if s == 'c':
        fig.text(fx0 - 4.5 / W, fy1 + 1.0 / H, 'c', fontsize=9, fontweight='bold', va='top', ha='left'); continue
    fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top', ha='left')
os.makedirs(OUT, exist_ok=True)
print('QA:', overlap_check(fig))
fig.savefig(f'{OUT}/Fig1.png', dpi=600); fig.savefig(f'{OUT}/Fig1.pdf')
