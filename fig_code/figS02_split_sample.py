"""Fig. S2: split-sample check of the global map (regions found with odd years tested with even years and the
reverse; gl_map_split.py, gl_stage2_split.py, gl_split_compare.py).
Inputs: SPLITFULL (full-precision gl_cells.csv of the all-year, odd-year and even-year maps: gl_cells_full.csv, gl_cells_odd.csv,
gl_cells_even.csv) for panels a and b, SPLIT (out_odd_test_even_s2.csv, out_even_test_odd_s2.csv) for panel c and GL (gl_fig.npz
for the grid).  Grid points are selected from the full-precision values; a missing test in the other half stays missing (third
review, 2026-10-30: the earlier version selected from 0.05 K-quantized values and stored missing tests as 0).  Writes the
source table FigS2a_cells.csv next to the figure."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

SPLIT = os.environ.get('SPLIT', 'data/global_split')
SPLITFULL = os.environ.get('SPLITFULL', 'data/global_split')
GL = os.environ.get('GL', 'data/global')
OUT = os.environ.get('FIGOUT', 'fig')
GD = (113.5, 23.0)

g = np.load(f'{GL}/gl_fig.npz'); lon, lat = g['lon'], g['lat']
def cells(f):
    d = pd.read_csv(f'{SPLITFULL}/{f}')
    for k in ('valid', 'sig_best'):
        d[k] = d[k].astype(str).str.lower().eq('true')
    return d
Fa, Od, Ev = cells('gl_cells_full.csv'), cells('gl_cells_odd.csv'), cells('gl_cells_even.csv')
assert (Fa.i.values == Od.i.values).all() and (Od.j.values == Ev.j.values).all()
both = (Fa.valid & Od.valid & Ev.valid).values                                    # enough storm days in both halves
cross = lambda A, B: np.array([B[f'ex_{s}'].values[n] if isinstance(s, str) else np.nan for n, s in enumerate(A.sec_best.values)])
c = dict(i=Fa.i.values[both], j=Fa.j.values[both])
wo = (Od.sig_best & (Od.ex_best >= 0.5)).values[both]                            # warm and significant in that half
we = (Ev.sig_best & (Ev.ex_best >= 0.5)).values[both]
t_oe, t_eo = cross(Od, Ev)[both], cross(Ev, Od)[both]                              # other half, same direction; NaN = too few days
a_, b_ = np.where(wo, t_oe, np.nan), np.where(we, t_eo, np.nan)
nt = np.isfinite(a_).astype(int) + np.isfinite(b_).astype(int)
# the other half's excess for the direction chosen by the half that found the grid point: the mean of the available tests
v = np.where(nt > 0, np.nansum(np.vstack([a_, b_]), 0) / np.maximum(nt, 1), np.nan)
found = wo | we; tested = found & np.isfinite(v)
os.makedirs(OUT, exist_ok=True)
pd.DataFrame(dict(i=c['i'], j=c['j'], lon=lon[c['i']], lat=lat[c['j']], found_odd=wo, found_even=we, sec_odd=Od.sec_best.values[both],
                  sec_even=Ev.sec_best.values[both], ex_best_odd=Od.ex_best.values[both], ex_best_even=Ev.ex_best.values[both],
                  test_in_even=np.where(wo, t_oe, np.nan), test_in_odd=np.where(we, t_eo, np.nan), n_tests=nt, shown=v,
                  ex_any_odd=Od.ex_any.values[both], ex_any_even=Ev.ex_any.values[both])).to_csv(
    f'{OUT}/FigS2a_cells.csv', index=False, float_format='%.4f')
for lab, f_, t_ in (('odd -> even', wo, t_oe), ('even -> odd', we, t_eo)):
    ok = f_ & np.isfinite(t_)
    print(f'{lab}: found {f_.sum()}, testable {ok.sum()}, warmer {(t_[ok] > 0).sum()}, mean {t_[ok].mean():.3f} K')

bounds = np.arange(-2.5, 2.51, 0.5)
cols = plt.get_cmap('RdBu_r')(np.linspace(0.06, 0.94, len(bounds) + 1))
cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1]); norm = BoundaryNorm(bounds, cmap.N)
NS, NOEXP, NOTEST = '#BDBDBD', '#ECECEC', '#6F6F6F'

pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
W = 180.0; MW = 175.0; MH = MW * (Y1 - Y0) / (X1 - X0)
PH = 52.0
H = 3.0 + MH + 10.0 + PH + 9.0
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()

# a: grid points warm in one half, coloured by the other half
ax = fig.add_axes(box(4.5, H - 3.0 - MH, MW, MH), projection=proj)
ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_xticks([]); ax.set_yticks([])
ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor=NOEXP, edgecolor='none', zorder=0)
G = np.full((len(lon), len(lat)), np.nan); G[c['i'], c['j']] = 0.0                       # valid in both halves
F = np.full_like(G, np.nan); F[c['i'][found], c['j'][found]] = v[found]
lone = np.r_[lon - 0.5, lon[-1] + 0.5]; late = np.clip(np.r_[lat - 0.5, lat[-1] + 0.5], -90, 90)
ax.pcolormesh(lone, late, np.ma.masked_invalid(np.where(np.isfinite(F), np.nan, G)).T * 0 + 1, cmap=ListedColormap([NS]), vmin=0, vmax=2,
              transform=pc, zorder=1, rasterized=True)
ax.pcolormesh(lone, late, np.ma.masked_invalid(F).T, cmap=cmap, norm=norm, transform=pc, zorder=2, rasterized=True)
T = np.full_like(G, np.nan); T[c['i'][found & ~tested], c['j'][found & ~tested]] = 1.0     # found, but no test in the other half
ax.pcolormesh(lone, late, np.ma.masked_invalid(T).T, cmap=ListedColormap([NOTEST]), vmin=0, vmax=2, transform=pc, zorder=2,
              rasterized=True)
ax.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#555555', zorder=3)
ax.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=(0, (2, 2)), xlocs=range(-180, 181, 30), ylocs=[-30, 0, 30], zorder=0.5)
for la_ in (-30, 0, 30):
    x, y = proj.transform_point(-168, la_, pc)
    ax.text(x, y, {-30: '30° S', 0: '0°', 30: '30° N'}[la_], fontsize=6.5, color='#666666', ha='left', va='bottom', zorder=6)
ax.plot(*GD, marker='*', ms=6, mfc='none', mec='k', mew=0.6, transform=pc, zorder=7)
for s in ax.spines.values():
    s.set_linewidth(0.5)
ax.text(0.006, 0.975, 'Warm in one half of the years: Tmax excess in the other half', transform=ax.transAxes, fontsize=7, ha='left', va='top',
        zorder=8, path_effects=HALO)
n_pos = int((v[tested] > 0).sum()); n_f = int(found.sum()); n_t = int(tested.sum())
ax.text(0.006, 0.89, f'{n_pos} of {n_t} testable grid points warmer in the other half', transform=ax.transAxes, fontsize=6.5, ha='left', va='top',
        zorder=8, path_effects=HALO)
q0 = ax.transData.transform(proj.transform_point(57, -13, pc)); q1 = ax.transData.transform(proj.transform_point(103, -13, pc))
(cx0, cy0), (cx1, _) = inv.transform(q0), inv.transform(q1)
cax = fig.add_axes([cx0, cy0, cx1 - cx0, 1.6 / H])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.set_label('Tmax excess in the other half (K)', fontsize=6.5, labelpad=1)
cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels(['−2.5', '−1.5', '−0.5', '0.5', '1.5', '2.5'])
from matplotlib.patches import Rectangle
k0 = inv.transform(ax.transData.transform(proj.transform_point(188, -24, pc)))
KEY = ((NOTEST, 'Warm in one half, not testable in the other'), (NS, 'Not warm in either half'), (NOEXP, 'Few TC days'))
for i, (cl, lab) in enumerate(KEY):
    fig.patches.append(Rectangle((k0[0], k0[1] + (2 - i) * 2.6 / H), 2.2 / W, 1.6 / H, transform=fig.transFigure, facecolor=cl,
                                 edgecolor='#888888', lw=0.3, zorder=9))
    fig.text(k0[0] + 3.0 / W, k0[1] + (2 - i) * 2.6 / H + 0.8 / H, lab, fontsize=6.5, va='center', ha='left', zorder=9)

# b: excess with storms in any direction, odd against even years, all grid points valid in both halves
yb = 9.0
axb = fig.add_axes(box(13.0, yb, PH, PH))
xo, xe = Od.ex_any.values[both], Ev.ex_any.values[both]
axb.scatter(xo, xe, s=2.0, color='#333333', alpha=0.35, lw=0, rasterized=True, zorder=2)
axb.plot([-2.8, 2.8], [-2.8, 2.8], color='k', lw=0.5, ls=(0, (3, 2)))
axb.axhline(0, color='#888888', lw=0.4); axb.axvline(0, color='#888888', lw=0.4)
r = np.corrcoef(xo, xe)[0, 1]
axb.text(0.04, 0.96, f'r = {r:.2f}\n{len(xo):,} grid points', transform=axb.transAxes, ha='left', va='top', fontsize=6.5)
axb.set_xlim(-2.8, 2.8); axb.set_ylim(-2.8, 2.8); axb.set_xticks([-2, -1, 0, 1, 2]); axb.set_yticks([-2, -1, 0, 1, 2])
axb.set_xticklabels(['−2', '−1', '0', '1', '2']); axb.set_yticklabels(['−2', '−1', '0', '1', '2'])
axb.set_xlabel('Excess, odd years (K)'); axb.set_ylabel('Excess, even years (K)')
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
# c: regions found in one half, tested with the other
axc = fig.add_axes(box(84.0, yb, 94.0, PH))
T = []
for f, tag in (('out_odd_test_even_s2.csv', 'odd'), ('out_even_test_odd_s2.csv', 'even')):
    d = pd.read_csv(f'{SPLIT}/{f}'); d = d[d.sign.isin(['warm', 'cool'])].copy(); d['found'] = tag; T.append(d)
T = pd.concat(T, ignore_index=True)
T['sig'] = T.tmax_sig_fdr.astype(str).str.lower().eq('true') & np.where(T.sign == 'warm', T.tmax_lo > 0, T.tmax_hi < 0)
for (sg, fd), d in T.groupby(['sign', 'found']):
    col = RED if sg == 'warm' else BLUE; mk = 'o' if fd == 'odd' else 's'
    axc.vlines(d.ex_stage1, d.tmax_lo, d.tmax_hi, color=col, lw=0.5, alpha=0.55, zorder=2)
    axc.scatter(d.ex_stage1[d.sig], d.tmax_ex[d.sig], s=13, marker=mk, color=col, lw=0, zorder=4)
    axc.scatter(d.ex_stage1[~d.sig], d.tmax_ex[~d.sig], s=13, marker=mk, facecolor='white', edgecolor=col, lw=0.6, zorder=4)
axc.axhline(0, color='k', lw=0.4)
axc.plot([-2.6, 2.6], [-2.6, 2.6], color='k', lw=0.5, ls=(0, (3, 2)), zorder=1)
axc.set_xlim(-2.6, 2.7); axc.set_ylim(-2.6, 2.7)
axc.set_xticks(np.arange(-2, 2.01, 1)); axc.set_yticks(np.arange(-2, 2.01, 1))
axc.set_xticklabels(['−2', '−1', '0', '1', '2']); axc.set_yticklabels(['−2', '−1', '0', '1', '2'])
axc.set_xlabel('Map change of the region in the half that found it (K)')
axc.set_ylabel('Excess over analogues,\nother half (K)')
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sg, x0_, y0_, ha in (('warm', 0.05, 0.97, 'left'), ('cool', 0.97, 0.06, 'right')):
    d = T[T.sign == sg]; s_ = 1 if sg == 'warm' else -1
    axc.text(x0_, y0_, f'{sg.capitalize()} regions: {int((s_ * d.tmax_ex > 0).sum())} of {len(d)} same sign,\n'
             f'{int(d.sig.sum())} significant', transform=axc.transAxes, ha=ha, va='top' if sg == 'warm' else 'bottom',
             fontsize=6.5, color=RED if sg == 'warm' else BLUE)
kh = [Line2D([], [], ls='none', marker='o', ms=3.4, mfc='#555555', mec='none'), Line2D([], [], ls='none', marker='s', ms=3.4, mfc='#555555', mec='none'),
      Line2D([], [], ls='none', marker='o', ms=3.4, mfc='white', mec='#555555', mew=0.6)]
axc.legend(kh, ['Found with odd years', 'Found with even years', 'Not significant'], loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3,
           frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.1, columnspacing=1.2)

for s, x, y in (('a', 0.3, H - 2.6), ('b', 0.3, yb + PH + 5.0), ('c', 70.0, yb + PH + 5.0)):
    fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top', ha='left')
os.makedirs(OUT, exist_ok=True)
print('QA:', overlap_check(fig))
print('found', n_f, 'testable', n_t, 'warmer in other half', n_pos)
fig.savefig(f'{OUT}/FigS2.png', dpi=600); fig.savefig(f'{OUT}/FigS2.pdf')
