"""v10 SI figure (v09 Fig. 5 without the map, which is v10 Fig. 2a; panels b-d relabelled a-c). Fig. 5: removing the storm from GraphCast on the warm and cool coasts of the global map (2016-2021, all usable exposure days).
a: effect of the storm in each region (0.25° model); b: effect against the region's warming in the map (Fig. 1a), with the pooled
values; c: pooled effect for each removal, the null edit, days not used in the hottest-day set, the hottest-day set and the cool
regions; d: effect by distance of the grid point from the coast. Inputs (GGC, default data/ggc): see ggc_common.py.
Usage: GGC=... FIGOUT=... python3 figS05_graphcast_cells.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
from ggc_common import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines

OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
PA, PH_ = pooled_from('all_result.txt'), pooled_from('hot_result.txt')
O, S = regions('all', 'oper'), regions('all', 'small')
CO = pd.read_csv(f'{GGC}/all_oper_cases.csv')
U = O[O.usable]
rng = np.random.default_rng(7)
CI = all_ci(rng, CO, U.index)                      # same order and seed as Table S10 and Fig. S7a

bounds = np.arange(-2.0, 2.01, 0.5)
cmap, norm = binned('RdBu_r', bounds)
pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
W = 180.0; MW = 174.0; MH = MW * (Y1 - Y0) / (X1 - X0)
YB, PH = 10.5, 44.0                                # bottom and height of the lower row
H = 6.5 + PH + YB
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()

# ---------------------------------------------------------------- b: effect against the warming of the region in the map
BX, BW = 13.0, 46.0
AXB = fig.add_axes(box(BX, YB, BW, PH))
xx = np.array([-2.2, 2.2])
AXB.plot(xx, xx, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)), zorder=1)
AXB.plot(xx, xx / 2, color='#9E9E9E', lw=0.6, ls=(0, (1, 1.5)), zorder=1)
AXB.axhline(0, color='#555555', lw=0.5, zorder=1); AXB.axvline(0, color='#555555', lw=0.5, zorder=1)
for r in U.index:
    o = U.loc[r]; w_ = o.kind == 'warm'; col = RED if w_ else BLUE; mk = 'o' if w_ else 's'; lo_, hi_ = CI[r]
    AXB.plot([o.ex_map, o.ex_map], [lo_, hi_], color=col, lw=0.6, alpha=0.45, zorder=2, solid_capstyle='butt')
    AXB.plot(o.ex_map, o.d_free, marker=mk, ms=4.0, mfc=col if o.consistent else 'white', mec=col, mew=0.6 if o.consistent else 1.0,
             ls='none', zorder=3)
for r, dx, dy, ha in (('W12', -0.09, -0.05, 'right'), ('W22', 0.08, -0.17, 'left'), ('W23', 0.09, -0.02, 'left'), ('C8', -0.11, 0.0, 'right')):
    o = U.loc[r]
    AXB.text(o.ex_map + dx, o.d_free + dy, NUM(r), fontsize=6.0, ha=ha, va='center', color=RED if r[0] == 'W' else BLUE, zorder=5)
# pooled values (equal weight per region) with their intervals
for key, mx, col in (('free', PA['oper']['map'], RED), ('cool', PA['oper']['cool_map'], BLUE)):
    m, lo, hi = PA['oper'][key]
    AXB.errorbar(mx, m, yerr=[[m - lo], [hi - m]], fmt='D', ms=4.6, mfc='k', mec='k', ecolor='k', elinewidth=0.9, capsize=1.6,
                 capthick=0.8, zorder=6)
AXB.set_xlim(-1.9, 2.1); AXB.set_ylim(-1.5, 2.1); AXB.set_xticks([-1, 0, 1, 2]); AXB.set_yticks([-1, 0, 1, 2])
AXB.set_xlabel('Warming in the map (K)', labelpad=1); AXB.set_ylabel('Effect of the TC (K)', labelpad=1)
for s_ in ('top', 'right'):
    AXB.spines[s_].set_visible(False)
AXB.text(1.42, 1.95, '1:1', fontsize=6.0, color='#7A7A7A', ha='right', va='center')
AXB.text(2.08, 1.22, '1:2', fontsize=6.0, color='#7A7A7A', ha='right', va='bottom')
hb = [mlines.Line2D([], [], marker='o', ms=4.0, mfc=RED, mec=RED, ls='none'), mlines.Line2D([], [], marker='s', ms=4.0, mfc=BLUE, mec=BLUE, ls='none'),
      mlines.Line2D([], [], marker='D', ms=4.0, mfc='k', mec='k', ls='none')]
AXB.legend(hb, ['warm regions', 'cool regions', 'pooled'], loc='upper left', frameon=False, fontsize=6.2, handletextpad=0.2,
           borderaxespad=0.1, labelspacing=0.25)

# ---------------------------------------------------------------- c: pooled effect
CX, CW = 72.0, 62.0
AXC = fig.add_axes(box(CX, YB, CW, PH))
cats = [('core', 'Core', PA), ('free', 'Region-\nfree', PA), ('full', 'Full', PA), ('null', 'Null\nedit', PA), ('new', 'New\ndays', PA),
        ('free', 'Hottest\ndays', PH_), ('cool', 'Cool\nregions', PA)]
xpos = np.arange(len(cats)); wd = 0.36
for mdl, off, fc in (('oper', -wd / 2, None), ('small', wd / 2, 'white')):
    for i, (key, _, src) in enumerate(cats):
        m, lo, hi = src[mdl][key]; col = BLUE if key == 'cool' else (GREY if key == 'null' else RED)
        AXC.bar(xpos[i] + off, m, wd, color=col if fc is None else 'white', edgecolor=col, lw=0.8, zorder=2)
        AXC.errorbar(xpos[i] + off, m, yerr=[[m - lo], [hi - m]], fmt='none', ecolor='k', elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
        if mdl == 'oper':
            AXC.text(xpos[i] + off - (0.12 if m < 0 else 0), (hi if m > 0 else lo) + (0.05 if m > 0 else -0.06), f'{m:+.2f}'.replace('-', '−'), fontsize=5.8,
                     ha='center', va='bottom' if m > 0 else 'top', color=col)
for i, (key, _, src) in enumerate(cats):            # warming of the same regions in the global map (short dashes)
    mv = {'new': PA['oper']['new_map'], 'cool': PA['oper']['cool_map'], 'null': None}.get(key, PA['oper']['map'])
    if mv is not None:
        AXC.plot([xpos[i] - 0.42, xpos[i] + 0.42], [mv, mv], color='k', lw=0.7, ls=(0, (2, 1.2)), zorder=4)
AXC.axhline(0, color='#555555', lw=0.5); AXC.set_xticks(xpos); AXC.set_xticklabels([c[1] for c in cats], fontsize=6.2, linespacing=0.95)
AXC.set_ylim(-1.15, 1.55); AXC.set_ylabel('Pooled effect (K)', labelpad=1); AXC.tick_params(axis='x', length=0, pad=2)
for s_ in ('top', 'right'):
    AXC.spines[s_].set_visible(False)
hc = [Rectangle((0, 0), 1, 1, fc=RED, ec=RED, lw=0.8), Rectangle((0, 0), 1, 1, fc='white', ec=RED, lw=0.8),
      mlines.Line2D([], [], color='k', lw=0.7, ls=(0, (2, 1.2)))]
AXC.legend(hc, ['0.25°', '1°', 'warming in the map'], loc='upper right', bbox_to_anchor=(1.0, 1.05), frameon=False, fontsize=6.2,
           handlelength=1.1, handleheight=0.8, borderpad=0.2, ncol=3, columnspacing=0.8, handletextpad=0.4)

# ---------------------------------------------------------------- d: effect by distance of the grid point from the coast
DX, DW = 145.0, 33.0
AXD = fig.add_axes(box(DX, YB, DW, PH))
xe = np.arange(4)
for mdl, off, fill in (('oper', -0.12, True), ('small', 0.12, False)):
    for kind, col, mk in (('warm', RED, 'o'), ('cool', BLUE, 's')):
        v = cell_bins(rng, mdl, kind, O if mdl == 'oper' else S)
        m = np.array([x[0] for x in v]); lo = np.array([x[1] for x in v]); hi = np.array([x[2] for x in v])
        AXD.plot(xe + off, m, color=col, lw=0.7 if fill else 0.5, ls='-' if fill else (0, (3, 2)), zorder=2)
        AXD.errorbar(xe + off, m, yerr=[m - lo, hi - m], fmt='none', ecolor=col, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=2.5)
        AXD.plot(xe + off, m, marker=mk, ms=3.8 if fill else 3.2, mfc=col if fill else 'white', mec=col, mew=0.8, ls='none', zorder=3)
        print('panel d', mdl, kind, 'means', np.round(m, 2), 'lo', np.round(lo, 2), 'hi', np.round(hi, 2), 'grid-point days', [x[3] for x in v])
        if mdl == 'oper':
            nl = cell_bins(rng, mdl, kind, O, 'd_null')
            AXD.plot(xe + off, [x[0] for x in nl], marker='_', ms=5, mec='#7A7A7A', mew=0.9, ls='none', zorder=2.8)
AXD.axhline(0, color='#555555', lw=0.5, zorder=1)
AXD.set_xticks(xe); AXD.set_xticklabels(['0–\n50', '50–\n150', '150–\n300', '>\n300'], fontsize=6.2, linespacing=0.95)
AXD.set_xlim(-0.5, 3.5); AXD.set_ylim(-1.0, 1.1); AXD.tick_params(axis='x', length=0, pad=2)
AXD.set_xlabel('Distance from coast (km)', labelpad=1); AXD.set_ylabel('Effect of the TC (K)', labelpad=1)
for s_ in ('top', 'right'):
    AXD.spines[s_].set_visible(False)
AXD.text(0.03, 0.99, 'warm', transform=AXD.transAxes, fontsize=6.2, color=RED, ha='left', va='top')
AXD.text(0.03, 0.02, 'cool', transform=AXD.transAxes, fontsize=6.2, color=BLUE, ha='left', va='bottom')

# panel letters: the lower row shares one height
yl = (YB + PH + 5.5) / H
for s_, x0 in (('a', 1.0), ('b', CX - 10.0), ('c', DX - 10.0)):
    fig.text(x0 / W, yl, s_, fontsize=9, fontweight='bold', va='top', ha='left')
print('pooled all', PA['oper']); print('pooled hot', PH_['oper'])
print('size mm', W, round(H, 1))
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/FigS5.png', dpi=600); fig.savefig(f'{OUT}/FigS5.pdf')
