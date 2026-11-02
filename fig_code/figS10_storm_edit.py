"""Fig. S6: what the removal changes in the GraphCast initial state, and how the forecast adjusts.
Inputs: REVA (rev_diag.csv, edit_maps_q.npz from rev_pack.py a), REVB (steps_<model>.csv, steps_<model>_maps_q.npz from rev_pack.py b;
optional), DB (gd_stations.csv)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.geodesic import Geodesic
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

REVA = os.environ.get('REVA', 'data/revision/figure_inputs')
REVB = os.environ.get('REVB', 'data/revision/figure_inputs')
DB = os.environ.get('DB', 'data/db')
OUT = os.environ.get('FIGOUT', 'fig')
CASE = int(os.environ.get('CASE', '16'))
EXT = [100, 140, 8, 36]
pc = ccrs.PlateCarree(); GEO = Geodesic()
stn = pd.read_csv(f'{DB}/gd_stations.csv')
D = pd.read_csv(f'{REVA}/rev_diag.csv')
z = np.load(f'{REVA}/edit_maps_q.npz')
la, lo = z['lat'], z['lon']


def qget(zz, k):
    q = zz[k + '|q'].astype(float)
    return np.where(q == -128, np.nan, q * float(zz[k + '|s']))


def disc(base, bounds, lo_=0.06, hi_=0.94):
    cols = plt.get_cmap(base)(np.linspace(lo_, hi_, len(bounds) + 1))
    cm = ListedColormap(cols[1:-1]); cm.set_under(cols[0]); cm.set_over(cols[-1]); return cm, BoundaryNorm(bounds, cm.N)


VAR = [('NOTC', 'Core'), ('WIDEL', 'Station-\nfree'), ('W1000', 'Full to\n1,000 km'), ('WIDE', 'Full'), ('NULLW', 'Null,\nfull')]
VCOL = {'NOTC': CFG['small_std'], 'WIDEL': '#8FB8DE', 'W1000': '#B07A2A', 'WIDE': CFG['small_wide'], 'NULLW': GREY}
cen = z[f'{CASE}|centre']
r48 = D[(D.case == CASE) & (D.lead == 48)].set_index('variant')
steps = {m: f'{REVB}/steps_{m}.csv' for m in ('oper', 'small') if os.path.exists(f'{REVB}/steps_{m}.csv')}
smaps = {m: f'{REVB}/steps_{m}_maps_q.npz' for m in ('oper', 'small') if os.path.exists(f'{REVB}/steps_{m}_maps_q.npz')}

W = 180.0; MW = 54.0; MH = MW * (EXT[3] - EXT[2]) / (EXT[1] - EXT[0]); PH = 30.0
H = 3.0 + MH + 19.0 + PH + 15.0 + MH + 14.0
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
XS = (7.0, 65.5, 124.0)


def base_map(x0, y0, left):
    ax = fig.add_axes(box(x0, y0, MW, MH), projection=pc)
    ax.set_extent(EXT, crs=pc); ax.set_xticks([]); ax.set_yticks([])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4, edgecolor='#333333', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.25, edgecolor='#777777', zorder=3)
    gl = ax.gridlines(draw_labels=True, lw=0.25, color='#9E9E9E', ls=':', xlocs=range(100, 141, 10), ylocs=range(10, 36, 10), zorder=4)
    gl.top_labels = gl.right_labels = False; gl.left_labels = left
    gl.xlabel_style = gl.ylabel_style = dict(size=6.5); gl.xpadding = gl.ypadding = 2
    ax.scatter(stn.lon, stn.lat, s=5, marker='^', color='k', lw=0, transform=pc, zorder=6)
    ax.plot(*cen, marker='x', ms=4, mew=0.8, color='k', transform=pc, zorder=7)
    return ax


def ring(ax, r_km, **k):
    c = np.asarray(GEO.circle(cen[0], cen[1], r_km * 1e3, n_samples=180))
    ax.plot(c[:, 0], c[:, 1], transform=pc, zorder=5, **k)


def hcbar(x0, y0, cm, nm, ticks, lab, labels=None, ext='both', w=None):
    w = w or MW - 14
    cax = fig.add_axes(box(x0 + 0.5 * (MW - w), y0, w, 1.5))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation='horizontal', extend=ext, ticks=ticks)
    cb.set_label(lab, fontsize=6.5, labelpad=1); cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    if labels is not None: cb.ax.set_xticklabels(labels)


# ---- row 1: the edit (full removal, first input time, 48 h before the case afternoon)
y1 = H - 3.0 - MH
ax = base_map(XS[0], y1, True)
du, dv, dp = qget(z, f'{CASE}|WIDE|du850'), qget(z, f'{CASE}|WIDE|dv850'), qget(z, f'{CASE}|WIDE|dmsl')
sp = np.hypot(du, dv)
bw = np.array([0.5, 1, 2, 4, 8, 16, 32]); cmw, nmw = disc('YlGnBu', bw, 0.05, 0.95)
ax.contourf(lo, la, np.where(sp >= 0.5, sp, np.nan), levels=bw, cmap=cmw, norm=nmw, extend='max', transform=pc, zorder=1)
c = ax.contour(lo, la, dp, levels=[1, 2, 5, 10, 20, 40], colors='#7A0177', linewidths=0.5, transform=pc, zorder=2)
ax.clabel(c, levels=[1, 5], fontsize=6, fmt='%d', inline_spacing=1)
LO, LA = np.meshgrid(lo, la); s_ = (slice(None, None, 2), slice(None, None, 2)); m_ = sp[s_] > 1
ax.quiver(LO[s_][m_], LA[s_][m_], du[s_][m_], dv[s_][m_], transform=pc, scale=300, width=0.004, headwidth=3.5, color='k', zorder=4)
ring(ax, float(r48.loc['WIDE', 'R']), color='k', lw=0.5)
ring(ax, 1500.0, color='k', lw=0.6, ls=(0, (3, 2)))
ring(ax, float(r48.loc['WIDEL', 'R_out']), color='#2166AC', lw=0.6, ls=(0, (1, 1)))
ax.text(0.02, 0.97, 'Full removal: change of 850-hPa\nwind and sea-level pressure', transform=ax.transAxes, fontsize=6.5, va='top',
        ha='left', zorder=8, path_effects=HALO)
hcbar(XS[0], y1 - 8.5, cmw, nmw, bw, 'Change of 850-hPa wind (m s$^{-1}$)', ['0.5', '1', '2', '4', '8', '16', '32'], ext='max')

ax = base_map(XS[1], y1, False)
im = qget(z, f'{CASE}|WIDE|imp_1000_500')
LO_, LA_ = np.meshgrid(np.radians(lo), np.radians(la)); c0 = np.radians(cen)
dkm = 2 * 6371 * np.arcsin(np.sqrt(np.sin((LA_ - c0[1]) / 2) ** 2 + np.cos(LA_) * np.cos(c0[1]) * np.sin((LO_ - c0[0]) / 2) ** 2))
im = np.where(dkm >= float(r48.loc['WIDE', 'R']), im, np.nan)           # inside the core disk temperature is edited too: not an imbalance
bi = np.array([-1.5, -1, -0.5, -0.25, -0.1, -0.05]); cmi, nmi = disc('Purples_r', bi, 0.15, 0.95)
ax.contourf(lo, la, np.where(im < -0.05, im, np.nan), levels=bi, cmap=cmi, norm=nmi, extend='min', transform=pc, zorder=1)
ring(ax, float(r48.loc['WIDE', 'R']), color='k', lw=0.5)
ring(ax, 1500.0, color='k', lw=0.6, ls=(0, (3, 2)))
ax.text(0.02, 0.97, 'Imbalance outside the core disk,\n1000–500 hPa', transform=ax.transAxes, fontsize=6.5, va='top', ha='left', zorder=8,
        path_effects=HALO)
hcbar(XS[1], y1 - 8.5, cmi, nmi, bi, 'Implied minus actual layer temperature (K)', ['−1.5', '−1', '−0.5', '−0.25', '−0.1', '−0.05'], ext='min')

# c: null geometry
axc = fig.add_axes(box(134.0, y1 + 4.0, 43.0, MH - 4.0))
N = D[D.variant == 'NULLW'].dropna(subset=['null_sep'])
axc.hist(N.null_sep, bins=np.arange(1150, 1951, 50), color=GREY, lw=0)
axc.axvline(1500, color='k', lw=0.6, ls=(0, (3, 2)))
R = N.R.values; s = N.null_sep.values
wgt = np.where(s <= R, 1, np.where(s >= 1500, 0, 0.5 * (1 + np.cos(np.pi * (s - R) / (1500 - R)))))
axc.set_ylim(0, 26)
axc.text(0.99, 0.70, f'Real storm inside\nthe taper: {(s < 1500).mean():.0%};\nweight there:\nmedian {np.median(wgt[s < 1500]):.2f},\nmax {wgt.max():.2f}',
         transform=axc.transAxes, ha='right', va='top', fontsize=6.5)
axc.text(1510, 25.5, 'End of taper', fontsize=6.5, ha='left', va='top')
axc.set_xlim(1150, 1950); axc.set_xticks([1200, 1400, 1600, 1800]); axc.set_xticklabels(['1,200', '1,400', '1,600', '1,800'])
axc.set_xlabel('Null point to real storm (km)'); axc.set_ylabel('Initial states')
for k in ('top', 'right'): axc.spines[k].set_visible(False)

# ---- row 2: the edit at the 19 stations, all hot cases (mean over leads), and the forecast after the edit
y2 = y1 - 19.0 - PH
C = D.groupby(['variant', 'case'])[['stn_dV850', 'stn_dmsl', 'stn_imp_lo', 'stn_imp_hi']].mean().reset_index()
rng = np.random.default_rng(1)
axd = fig.add_axes(box(13.0, y2, 46.0, PH))
axe = fig.add_axes(box(76.0, y2, 44.0, PH))
for i, (v, lab) in enumerate(VAR):
    g = C[C.variant == v]; x = i + rng.uniform(-0.18, 0.18, len(g))
    axd.scatter(x, g.stn_dV850, s=5, color=VCOL[v], lw=0, alpha=0.85, zorder=3)
    axd.plot([i - 0.28, i + 0.28], [g.stn_dV850.mean()] * 2, color='k', lw=0.9, zorder=4)
    for j, (col, off, mk) in enumerate((('stn_imp_lo', -0.17, 'o'), ('stn_imp_hi', 0.17, 's'))):
        xx = i + off + rng.uniform(-0.08, 0.08, len(g))
        axe.scatter(xx, g[col], s=5, marker=mk, color=VCOL[v], lw=0, alpha=0.85, zorder=3)
        axe.plot([i + off - 0.13, i + off + 0.13], [g[col].mean()] * 2, color='k', lw=0.9, zorder=4)
for a_, yl in ((axd, '850-hPa wind change\nat the stations (m s$^{-1}$)'), (axe, 'Imbalance at the\nstations (K)')):
    a_.set_xticks(range(len(VAR))); a_.set_xticklabels([l for _, l in VAR], fontsize=6.5); a_.set_ylabel(yl)
    a_.set_xlim(-0.5, len(VAR) - 0.5)
    for k in ('top', 'right'): a_.spines[k].set_visible(False)
axd.set_ylim(-0.1, 3.6)
axe.axhline(0, color='k', lw=0.4); axe.set_ylim(-0.75, 0.12)
axe.set_yticks([-0.6, -0.4, -0.2, 0]); axe.set_yticklabels(['−0.6', '−0.4', '−0.2', '0'])
axe.legend([Line2D([], [], ls='none', marker='o', ms=3, mfc='#555555', mec='none'), Line2D([], [], ls='none', marker='s', ms=3, mfc='#555555', mec='none')],
           ['1000–850 hPa', '850–500 hPa'], loc='lower left', frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.1)

# f: forecast difference at the stations after the edit, every 6 h (cases 6 and 16)
axf = fig.add_axes(box(136.0, y2, 41.0, PH))
if steps:
    for m, f in steps.items():
        S = pd.read_csv(f)
        for cn, ls in ((6, '-'), (16, (0, (3, 1.5)))):
            P = S[S.case == cn].pivot_table(index='step_h', columns='variant', values='t2m')
            for v in ('WIDE', 'WIDEL'):
                if v in P:
                    axf.plot(P.index, P['CTL'] - P[v], color=CFG[f'{m}_wide'] if v == 'WIDE' else ('#8FB8DE' if m == 'small' else '#2166AC'),
                             lw=0.8, ls=ls)
    axf.axhline(0, color='k', lw=0.4)
    hl = [Line2D([], [], color=CFG[f'{m}_wide'] if v == 'WIDE' else ('#8FB8DE' if m == 'small' else '#2166AC'), lw=0.8)
          for m in ('small', 'oper') for v in ('WIDE', 'WIDEL') if m in steps]
    ll = [f'{"1°" if m == "small" else "0.25°"}, {"full" if v == "WIDE" else "station-free"}' for m in ('small', 'oper') for v in ('WIDE', 'WIDEL') if m in steps]
    hl += [Line2D([], [], color='#555555', lw=0.8), Line2D([], [], color='#555555', lw=0.8, ls=(0, (3, 1.5)))]; ll += ['Megi', 'Mangkhut']
    axf.legend(hl, ll, loc='upper left', frameon=False, fontsize=6.5, handlelength=1.6, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
else:
    axf.text(0.5, 0.5, 'forecast steps\npending', transform=axf.transAxes, ha='center', va='center', fontsize=6.5, color=GREY)
axf.set_xlim(0, 50); axf.set_xticks([0, 12, 24, 36, 48]); axf.set_xlabel('Forecast hour')
axf.set_ylabel('Control minus edited,\n2-m temperature (K)')
for k in ('top', 'right'): axf.spines[k].set_visible(False)

# ---- row 3: the effect of the full removal on 2-m temperature at +6, +12 and +48 h
y3 = y2 - 15.0 - MH
bt = np.array([-2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2]); cmt, nmt = disc('RdBu_r', bt)
mod = 'oper' if 'oper' in smaps else ('small' if 'small' in smaps else None)
for n, h in enumerate((6, 12, 48)):
    ax = base_map(XS[n], y3, n == 0)
    if mod:
        zs = np.load(smaps[mod]); f_ = qget(zs, f'{CASE}|WIDE|{h}|t2m')
        ax.contourf(zs['lon'], zs['lat'], f_, levels=bt, cmap=cmt, norm=nmt, extend='both', transform=pc, zorder=1)
        cm_ = ax.contour(zs['lon'], zs['lat'], qget(zs, f'{CASE}|WIDE|{h}|msl'), levels=[-20, -10, -5, -2, -1], colors='k', linewidths=0.4,
                         transform=pc, zorder=2)
    ax.text(0.02, 0.03, f'+{h} h', transform=ax.transAxes, fontsize=7, va='bottom', ha='left', zorder=8, path_effects=HALO)
hcbar(XS[1], y3 - 8.5, cmt, nmt, bt, '2-m temperature, control minus full removal (K)', ['−2', '−1', '−0.5', '−0.25', '0.25', '0.5', '1', '2'], w=90)

for s_, x_, y_ in (('a', 0.5, y1 + MH + 2.6), ('b', XS[1] - 3.0, y1 + MH + 2.6), ('c', 123.5, y1 + MH + 2.6),
                   ('d', 0.5, y2 + PH + 4.0), ('e', 63.5, y2 + PH + 4.0), ('f', 123.5, y2 + PH + 4.0),
                   ('g', 0.5, y3 + MH + 2.6), ('h', XS[1] - 3.0, y3 + MH + 2.6), ('i', XS[2] - 1.0, y3 + MH + 2.6)):
    fig.text(x_ / W, y_ / H, s_, fontsize=9, fontweight='bold', va='top', ha='left')
os.makedirs(OUT, exist_ok=True)
print('QA:', overlap_check(fig))
fig.savefig(f'{OUT}/FigS10.png', dpi=600); fig.savefig(f'{OUT}/FigS10.pdf')
