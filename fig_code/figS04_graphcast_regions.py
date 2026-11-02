"""Fig. S7: storm removal in GraphCast on the coasts of the global map, region by region (2016-2021); companion of Fig. 5.
a: effect in each region with at least three days (0.25° and 1° models, hottest-day set, null edit); b: effect on each day against
the region's ERA5 anomaly on that day; c: effect at 850 hPa against the effect at 2 m in each region.
Inputs (GGC, default data/ggc): see ggc_common.py. Usage: GGC=... FIGOUT=... python3 figS04_graphcast_regions.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import matplotlib.lines as mlines

from ggc_common import *
OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
PA, PH_ = pooled_from('all_result.txt'), pooled_from('hot_result.txt')
O, S = regions('all', 'oper'), regions('all', 'small'); HO = regions('hot', 'oper')
CO = pd.read_csv(f'{GGC}/all_oper_cases.csv')
U = O[O.usable]
rng = np.random.default_rng(7)
CI = all_ci(rng, CO, U.index)                      # same order and seed as Table S10 and Fig. 5b

W = 180.0
RH = 100.0; YB = 10.5; PH = 41.0; PG = 18.0          # left: panel a of height RH; right: panels b and c of height PH with gap PG
H = 3.0 + RH + YB
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
CX, CW = 117.0, 60.0

# ---------------------------------------------------------------- a: regions
yb0 = YB; AXB = fig.add_axes(box(38.0, yb0, 62.0, RH))
warm = U[U.kind == 'warm'].sort_values('d_free', ascending=False); cool = U[U.kind == 'cool'].sort_values('d_free', ascending=False)
order = list(warm.index) + list(cool.index)
ys = {}; y = 0
for r in warm.index:
    ys[r] = y; y -= 1
y -= 0.8
for r in cool.index:
    ys[r] = y; y -= 1
AXB.axvline(0, color='#555555', lw=0.5, zorder=1)
AXB.axhline(ys[cool.index[0]] + 0.9, color='#BDBDBD', lw=0.5, ls=(0, (3, 2)), zorder=1)
for r in order:
    yy = ys[r]; w_ = r[0] == 'W'; col = RED if w_ else BLUE; mk = 'o' if w_ else 's'
    o, s_ = O.loc[r], S.loc[r]; lo_, hi_ = CI[r]
    AXB.plot([lo_, hi_], [yy, yy], color=col, lw=0.6, alpha=0.55, zorder=2, solid_capstyle='butt')
    AXB.plot(s_.d_free, yy, marker=mk, ms=3.4, mfc='white', mec=col, mew=0.8, ls='none', zorder=3)
    if r in HO.index and HO.loc[r].usable:
        AXB.plot(HO.loc[r].d_free, yy, marker='D', ms=3.0, mfc='#6E6E6E', mec='none', ls='none', zorder=3.2)
    AXB.plot(o.d_free, yy, marker=mk, ms=4.6, mfc=col if o.consistent else 'white', mec=col, mew=0.9 if o.consistent else 1.3, ls='none', zorder=4)
    AXB.plot(o.d_null, yy, marker='|', ms=5, mec='#7A7A7A', mew=0.9, ls='none', zorder=3.5)
AXB.set_yticks([ys[r] for r in order]); AXB.set_yticklabels([f'{NUM(r)}  {NAME[r]}' for r in order], fontsize=6.4)
AXB.set_ylim(min(ys.values()) - 0.8, 0.8); AXB.set_xlim(-2.4, 2.3)
AXB.set_xlabel('Effect of the TC on 14 LT temperature (K)', fontsize=7.5)
AXB.tick_params(axis='y', length=0, pad=2); AXB.tick_params(axis='x', labelsize=6.5)
for s in ('top', 'right'):
    AXB.spines[s].set_visible(False)
AXB.text(-1.35, ys[warm.index[0]] + 0.1, 'warm regions', fontsize=6.5, ha='center', va='center', color=RED)
AXB.text(-1.35, ys[cool.index[0]] + 0.1, 'cool regions', fontsize=6.5, ha='center', va='center', color=BLUE)
hb = [mlines.Line2D([], [], marker='o', ms=4.6, mfc=RED, mec=RED, ls='-', lw=0.6, color=RED, alpha=1, label='0.25°, all days'),
      mlines.Line2D([], [], marker='o', ms=3.4, mfc='white', mec=RED, mew=0.8, ls='none', label='1°, all days'),
      mlines.Line2D([], [], marker='D', ms=3.0, mfc='#6E6E6E', mec='none', ls='none', label='0.25°, hottest days'),
      mlines.Line2D([], [], marker='|', ms=5, mec='#7A7A7A', mew=0.9, ls='none', label='null edit, 0.25°')]
AXB.legend(handles=hb, loc='upper left', bbox_to_anchor=(0.0, 0.955), frameon=False, fontsize=6.2, handlelength=1.2, borderpad=0.2)

# ---------------------------------------------------------------- b: effect against the case-day anomaly (cases)
AXD = fig.add_axes(box(CX, yb0 + PH + PG, CW, PH))
AXD.axhline(0, color='#555555', lw=0.5, zorder=1); AXD.axvline(0, color='#555555', lw=0.5, zorder=1)
xx = np.array([-12, 12]); AXD.plot(xx, xx, color='#BDBDBD', lw=0.6, ls=(0, (3, 2)), zorder=1)
CU = CO[CO.region.isin(U.index)]
for kind, col, mk in (('warm', RED, 'o'), ('cool', BLUE, 's')):
    c = CU[CU.kind == kind]
    AXD.plot(c.anom, c.d_free, marker=mk, ms=2.4, mfc=col, mec='none', alpha=0.45, ls='none', zorder=2)
    # storm-weighted mean effect in quartiles of the anomaly
    q = pd.qcut(c.anom, 4, labels=False); w = 1.0 / c.groupby(['region', 'sid']).sid.transform('size')
    mx = [c.anom[q == k].mean() for k in range(4)]; my = [float((c.d_free * w)[q == k].sum() / w[q == k].sum()) for k in range(4)]
    AXD.plot(mx, my, color=col, lw=0.9, marker=mk, ms=4.2, mfc='white', mec=col, mew=1.0, zorder=3)
    print('panel b', kind, 'cases', len(c), 'r =', round(float(np.corrcoef(c.anom, c.d_free)[0, 1]), 2), 'quartile means', np.round(my, 2))
yl_ = np.ceil(np.abs(CU.d_free).max() * 2 + 0.6) / 2
AXD.set_xlim(-12, 12); AXD.set_ylim(-yl_, yl_); print('panel d y range', CU.d_free.min(), CU.d_free.max(), CU.anom.min(), CU.anom.max())
AXD.set_xlabel('Case-day anomaly of the region, ERA5 (K)', fontsize=7.5); AXD.set_ylabel('Effect of the TC, 0.25° (K)', fontsize=7.5)
AXD.tick_params(labelsize=6.5)
for s in ('top', 'right'):
    AXD.spines[s].set_visible(False)
AXD.text(0.99, 0.97, '1:1', transform=AXD.transAxes, fontsize=6.0, color='#8C8C8C', ha='right', va='top')
AXD.text(0.02, 0.97, 'warm-region days', transform=AXD.transAxes, fontsize=6.2, color=RED, ha='left', va='top')
AXD.text(0.02, 0.86, 'cool-region days', transform=AXD.transAxes, fontsize=6.2, color=BLUE, ha='left', va='top')
AXD.text(0.98, 0.05, 'open: mean of each quarter', transform=AXD.transAxes, fontsize=5.8, color='#333333', ha='right', va='bottom')

# ---------------------------------------------------------------- c: effect at 850 hPa against the effect at 2 m (regions)
AXE = fig.add_axes(box(CX, yb0, CW, PH))
xx = np.array([-1.5, 2.0]); AXE.plot(xx, xx, color='#BDBDBD', lw=0.6, ls=(0, (3, 2)), zorder=1)
AXE.axhline(0, color='#555555', lw=0.5, zorder=1); AXE.axvline(0, color='#555555', lw=0.5, zorder=1)
for mdl, T, fill in (('small', S, False), ('oper', O, True)):
    for r in U.index:
        o = T.loc[r]; w_ = o.kind == 'warm'; col = RED if w_ else BLUE; mk = 'o' if w_ else 's'
        AXE.plot(o.d_free, o.d850_free, marker=mk, ms=4.0 if fill else 3.2, mfc=col if fill else 'white', mec=col, mew=0.8,
                 alpha=1.0 if fill else 0.8, ls='none', zorder=3 if fill else 2)
    ww = T.loc[[r for r in U.index if T.loc[r].kind == 'warm']]
    print('panel c', mdl, 'warm regions: mean 2 m', round(ww.d_free.mean(), 2), 'mean 850 hPa', round(ww.d850_free.mean(), 2),
          'regions with 850 > 0:', int((ww.d850_free > 0).sum()), 'of', len(ww))
AXE.set_xlim(-1.0, 1.6); AXE.set_ylim(-1.0, 1.75)
AXE.set_xlabel('Effect at 2 m (K)', fontsize=7.5); AXE.set_ylabel('Effect at 850 hPa (K)', fontsize=7.5); AXE.tick_params(labelsize=6.5)
for s_ in ('top', 'right'):
    AXE.spines[s_].set_visible(False)
AXE.text(1.22, 1.62, '1:1', fontsize=6.0, color='#8C8C8C', ha='right', va='center')
AXE.text(0.02, 0.97, 'warm regions', transform=AXE.transAxes, fontsize=6.2, color=RED, ha='left', va='top')
AXE.text(0.02, 0.86, 'cool regions', transform=AXE.transAxes, fontsize=6.2, color=BLUE, ha='left', va='top')
AXE.text(0.98, 0.05, 'filled: 0.25°; open: 1°', transform=AXE.transAxes, fontsize=5.8, color='#333333', ha='right', va='bottom')

# panel letters
yl = (yb0 + RH + 2.0) / H
fig.text(1.0 / W, yl, 'a', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text((CX - 12.0) / W, yl, 'b', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text((CX - 12.0) / W, (yb0 + PH + 4.0) / H, 'c', fontsize=9, fontweight='bold', va='top', ha='left')
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/FigS4.png', dpi=400); fig.savefig(f'{OUT}/FigS4.pdf')
print('saved', OUT)
