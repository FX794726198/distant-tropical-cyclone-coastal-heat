"""Fig. 4: the convection-permitting model and how the distant storm warms the coast.
a-f: WRF 3 km, control minus storm removed (the former fig3_wrf.py); g-j: WRF cross-section, GraphCast 500-hPa sinking and
10-m wind, WRF station columns and the warming at 2 m and 850 hPa (the former fig4_mech.py; its 850-hPa map is now Fig. S4c,
drawn by figS08_graphcast_circulation.py).
Inputs: WRX (wrx_maps.npz, wrx_series.csv, wrx_coast.csv, wrx_xsec.npz), GCX (fp_maps.npz, fp_case.csv), OBS (fingerprints.csv),
DB (gc_cases.csv, gd_stations.csv)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter

WRX = os.environ.get('WRX', 'data/wrf')
GCX = os.environ.get('GCX', 'data/figpack')
OBS = os.environ.get('OBS', 'data/obs')
DB = os.environ.get('DB', 'data/db')
OUT = os.environ.get('FIGOUT', 'fig')
PAIRS = os.environ.get('WRF_PAIRS', '6|ctl-wide,16|ctl-wide').split(',')   # the two 54-h cases without any new low (main estimate)
# (case, pair used in the main text, name)
CASES = [(6, 'ctl-wide', 'Megi 2016'), (16, 'ctl-wide', 'Mangkhut 2018'), (44, 'ctl_s30-wide_s30', 'Chanthu 2021 (30 h)'), (24, 'ctl-wide', 'Lekima 2019*')]
NEWLOW = {24}                                   # a new cyclone grew near the coast in the run without the storm
MAIN = {6, 16}                                  # 54-h runs without any new low: the main estimate (Chanthu 30 h is a sensitivity run)
NULLS = [(6, 'ctl-nullw'), (24, 'ctl-nullw')]
CC = {6: '#0072B2', 16: '#009E73', 24: '#9E9E9E', 44: '#CC79A7'}
pc = ccrs.PlateCarree()


def disc(base, bounds, lo=0.06, hi=0.94):
    cols = plt.get_cmap(base)(np.linspace(lo, hi, len(bounds) + 1))
    cm = ListedColormap(cols[1:-1]); cm.set_under(cols[0]); cm.set_over(cols[-1])
    return cm, BoundaryNorm(bounds, cm.N)


def hcbar(rect, cm, nm, ticks, label, ticklabels=None):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation='horizontal', extend='both', ticks=ticks)
    cb.set_label(label, fontsize=6.5, labelpad=1); cb.ax.tick_params(labelsize=6.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
    if ticklabels is not None: cb.ax.set_xticklabels(ticklabels)
    return cb


def letter(s, x, y):
    fig.text(x / W, y / H, s, fontsize=9, fontweight='bold', va='top', ha='left')


Z = np.load(f'{WRX}/wrx_maps.npz'); zk = list(Z['keys']); zs = Z['scales']
def mget(k):
    i = zk.index(k); q = Z['m%03d' % i].astype(float); return np.where(q == -128, np.nan, q * zs[i])
lo1, la1 = Z['lon01'], Z['lat01']
cases = pd.read_csv(f'{DB}/gc_cases.csv').set_index('case'); stn = pd.read_csv(f'{DB}/gd_stations.csv')
EXTW = [float(lo1.min()), float(lo1.max()), float(la1.min()), float(la1.max())]

# layout (mm): row 1 WRF maps, row 2 WRF time, coast and height, row 3 mechanism
W = 180.0
MW = 40.0; MH = MW * (EXTW[3] - EXTW[2]) / (EXTW[1] - EXTW[0])
Y3, R2B, R2H = 17.5, 60.5, 35.0                  # bottom of row 3 maps, bottom and height of row 2 axes
Y1 = R2B + R2H + 15.5                           # bottom of row 1 maps
H = Y1 + MH + 2.5
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]

# ---------------------------------------------------------------- a-d: WRF 2-m temperature difference at 14 LT
bounds = np.arange(-4, 4.01, 1.0); cmap, norm = disc('RdBu_r', bounds)
for n, (c, pair, name) in enumerate(CASES):
    ax = fig.add_axes(box(4 + n * 44.5, Y1, MW, MH), projection=pc)
    ax.set_extent(EXTW, crs=pc); ax.set_xticks([]); ax.set_yticks([])
    k = f'{c}|{pair}|dT2'
    if k in zk:
        ax.pcolormesh(lo1, la1, mget(k), cmap=cmap, norm=norm, transform=pc, zorder=1, rasterized=True, shading='nearest')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=0.4, edgecolor='#333333', zorder=3)
    ax.scatter(stn.lon, stn.lat, s=2.5, color='k', lw=0, transform=pc, zorder=4)
    r = cases.loc[c]; b = np.radians(r.brg)
    x0, y0 = 113.5 + 3.4 * np.sin(b), 23.0 + 2.0 * np.cos(b)      # arrow at the map edge pointing towards the storm
    ax.annotate('', xy=(x0 + 0.9 * np.sin(b), y0 + 0.9 * np.cos(b)), xytext=(x0, y0), xycoords=pc._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9, mutation_scale=6), zorder=6)
    ax.text(0.03, 0.04, name, transform=ax.transAxes, ha='left', va='bottom', fontsize=6.5, path_effects=HALO, zorder=7)
    ax.text(0.97, 0.04, f'{r.dist:,.0f} km', transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5, path_effects=HALO, zorder=7)
    letter('abcd'[n], 0.5 + n * 44.5, H - 1.0)
hcbar(box(52, Y1 - 4.2, 76, 1.6), cmap, norm, bounds, '2-m temperature with storm minus storm removed, 14 LT (K)',
      ['−4', '−3', '−2', '−1', '0', '1', '2', '3', '4'])

# ---------------------------------------------------------------- e: hourly station-mean difference
S = pd.read_csv(f'{WRX}/wrx_series.csv')
axe = fig.add_axes(box(14.0, R2B, 56.0, R2H))
for t0, t1 in ((-20, -8), (4, 16)):                        # 18 LT to 06 LT (V = 14 LT)
    axe.axvspan(t0, t1, color='#F2F2F2', zorder=0, lw=0)
D = {}
for c, pair, name in CASES:
    cr, xr_ = pair.split('-')
    a = S[(S.case == c) & (S.run == cr)].set_index('h').T2; b = S[(S.case == c) & (S.run == xr_)].set_index('h').T2
    d = (a - b).dropna()
    if c in MAIN: D[c] = d
    axe.plot(d.index, d.values, color=CC[c], lw=0.9, ls=(0, (1, 1)) if c in NEWLOW else '-',
             label=name.split(' ')[0] + (' (30 h)' if '30 h' in name else '') + ('*' if c in NEWLOW else ''), zorder=3)
for c, pair in NULLS:
    cr, xr_ = pair.split('-')
    a = S[(S.case == c) & (S.run == cr)].set_index('h').T2; b = S[(S.case == c) & (S.run == xr_)].set_index('h').T2
    d = (a - b).dropna()
    if len(d): axe.plot(d.index, d.values, color=GREY, lw=0.9, ls=(0, (3, 1.5)), zorder=2)
m = pd.concat(D, axis=1).mean(axis=1)
axe.plot(m.index, m.values, color='k', lw=1.8, zorder=4, label='Mean, Megi and Mangkhut')
axe.axhline(0, color='#555555', lw=0.5, zorder=1)
axe.set_xlim(-30, 18); axe.set_xticks(np.arange(-24, 19, 6))
axe.set_xticklabels([f'{(14 + h) % 24:02d}' for h in np.arange(-24, 19, 6)])
axe.set_xlabel('Local time (case day at 0; shading: night)', labelpad=1)
axe.set_ylabel('Warming by the storm,\n19 stations (K)')
h_, l_ = axe.get_legend_handles_labels()
h_.append(Line2D([], [], color=GREY, lw=0.9, ls=(0, (3, 1.5)))); l_.append('Same edit, no storm')
axe.legend(h_, l_, loc='upper left', ncol=2, frameon=False, fontsize=6.2, handlelength=1.3, columnspacing=0.7, borderaxespad=0.1,
           handletextpad=0.4, labelspacing=0.3)
axe.set_ylim(-1.0, 8.5); axe.set_yticks(range(0, 9, 2)); axe.axvline(0, ymax=0.68, color='k', lw=0.5, ls=':', zorder=1)
for sp in ('top', 'right'): axe.spines[sp].set_visible(False)

# ---------------------------------------------------------------- f: warming against distance from the coast (12-16 LT mean)
Cc = pd.read_csv(f'{WRX}/wrx_coast.csv')
axf = fig.add_axes(box(83.0, R2B, 36.0, R2H))
for c, pair, name in CASES:
    q = Cc[(Cc.case == c) & (Cc.pair == pair)]
    axf.plot(q.km, q.dT2_12_16, color=CC[c], lw=0.9, marker='o', ms=1.6, ls=(0, (1, 1)) if c in NEWLOW else '-', zorder=3)
qa = pd.concat([Cc[(Cc.case == c) & (Cc.pair == pair)] for c, pair, _ in CASES if c in MAIN]).groupby('km').dT2_12_16.mean()
axf.plot(qa.index, qa.values, color='k', lw=1.8, zorder=4)
for c, pair in NULLS:
    q = Cc[(Cc.case == c) & (Cc.pair == pair)]
    if len(q): axf.plot(q.km, q.dT2_12_16, color=GREY, lw=0.9, ls=(0, (3, 1.5)), zorder=2)
axf.axhline(0, color='#555555', lw=0.5, zorder=1)
axf.set_xlim(0, Cc.km.max() + 5); axf.set_xticks([0, 50, 100, 150, 200])
axf.set_xlabel('Distance from the coast (km)', labelpad=1)
axf.set_ylabel('Warming by the storm,\n12–16 LT (K)')
for sp in ('top', 'right'): axf.spines[sp].set_visible(False)

# ---------------------------------------------------------------- g: WRF cross-section from Guangdong towards the storm
X = np.load(f'{WRX}/wrx_xsec.npz'); xk = list(X['keys']); xsc = X['scales']
def xget(k):
    i = xk.index(k); q_ = X['m%03d' % i].astype(float); return np.where(q_ == -128, np.nan, q_ * xsc[i])
zl, sd = X['zlev'] / 1000, X['sdist']
dth = np.nanmean([xget(p + '|dth') for p in PAIRS], 0)
dw = uniform_filter(np.nan_to_num(100 * np.nanmean([xget(p + '|dw') for p in PAIRS], 0)), size=(3, 7))   # cm/s, smoothed over ~0.75 km x 210 km
dstorm = np.mean([float(X[f'dist_storm_{p.split("|")[0]}']) for p in PAIRS])
axg = fig.add_axes(box(132.5, R2B, 36.5, R2H))
bth = np.arange(-3, 3.01, 0.5); cmh, nmh = disc('RdBu_r', bth)
axg.contourf(sd, zl, dth, levels=bth, cmap=cmh, norm=nmh, extend='both', zorder=1)
axg.contour(sd, zl, dw, levels=[-4, -1.5, 1.5, 4], colors='k', linewidths=0.5, linestyles=['--', '--', '-', '-'], zorder=3)
axg.set_ylim(0.25, 6); axg.set_xlim(sd.min(), min(sd.max(), dstorm + 300)); axg.set_xticks([0, 500, 1000]); axg.tick_params(axis='x', pad=4.5)
axg.set_xlabel('Distance towards the storm (km)', labelpad=1); axg.set_ylabel('Height (km)', labelpad=1)
axg.plot([0], [0.12], marker='^', color='k', ms=4, clip_on=False, zorder=5)
axg.plot([dstorm], [0.12], marker='o', mfc='white', mec='k', mew=0.8, ms=4.5, clip_on=False, zorder=5)
axg.axvline(dstorm, color='k', lw=0.4, ls=':', zorder=2); axg.axvline(0, color='k', lw=0.4, ls=':', zorder=2)
axg.text(dstorm, 5.75, 'storm', ha='center', va='top', fontsize=6.5, path_effects=HALO, zorder=6)
axg.text(0, 5.75, 'stations', ha='center', va='top', fontsize=6.5, path_effects=HALO, zorder=6)
caxg = fig.add_axes(box(171.2, R2B + 1.5, 1.6, R2H - 3.0))
cbg = fig.colorbar(plt.cm.ScalarMappable(norm=nmh, cmap=cmh), cax=caxg, orientation='vertical', extend='both', ticks=bth[::2])
cbg.set_label('$\\Delta\\theta$ (K)', fontsize=6.5, labelpad=1); cbg.ax.tick_params(labelsize=6.2, length=2, pad=1); cbg.outline.set_linewidth(0.4)
cbg.ax.set_yticklabels(['−3', '−2', '−1', '0', '1', '2', '3'])

# ---------------------------------------------------------------- h: GraphCast 500-hPa vertical velocity (sinking positive) and 10-m wind
get, grids = load_maps(f'{GCX}/fp_maps.npz'); la, lo = grids['oper']
EXTG = [108, 126, 17, 28]
HW = 44.0; HH = HW * (EXTG[3] - EXTG[2]) / (EXTG[1] - EXTG[0])
axh = fig.add_axes(box(8.5, Y3, HW, HH), projection=pc)
axh.set_extent(EXTG, crs=pc); axh.set_xticks([]); axh.set_yticks([])
axh.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.45, edgecolor='#333333', zorder=3)
axh.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.3, edgecolor='#666666', zorder=3)
gl = axh.gridlines(draw_labels=True, lw=0.25, color='#9E9E9E', ls=':', xlocs=range(110, 127, 5), ylocs=range(15, 31, 5), zorder=4)
gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = dict(size=6.2); gl.xpadding = gl.ypadding = 1.5
w5 = get('oper|heat|WIDE|w500|mean'); ag = get('oper|heat|WIDE|w500|agree')
bw = np.array([-0.15, -0.1, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.1, 0.15])
cmw, nmw = disc('PuOr_r', bw)
axh.contourf(lo, la, w5, levels=bw, cmap=cmw, norm=nmw, extend='both', transform=pc, zorder=1)
LO, LA = np.meshgrid(lo, la)
sel = (np.abs(ag - 0.5) >= 0.3) & (np.round(LO * 2) % 2 == 0) & (np.round(LA * 2) % 2 == 0)
axh.scatter(LO[sel], LA[sel], s=0.4, color='k', lw=0, transform=pc, zorder=2)
u, v = get('oper|heat|WIDE|u10|mean'), get('oper|heat|WIDE|v10|mean')
sk = (slice(1, None, 3), slice(1, None, 3))
qv = axh.quiver(LO[sk], LA[sk], u[sk], v[sk], transform=pc, scale=40, width=0.0045, headwidth=3.5, headlength=4, color='#262626', zorder=5)
axh.quiverkey(qv, 0.97, 1.06, 2, '2 m s$^{-1}$', labelpos='W', fontproperties=dict(size=6.2), labelsep=0.04)
axh.scatter(stn.lon, stn.lat, s=4, marker='^', color='k', lw=0, transform=pc, zorder=6)
axh.text(0.02, 1.02, 'GraphCast 0.25°', transform=axh.transAxes, fontsize=6.2, ha='left', va='bottom')
hcbar(box(12.5, 8.6, HW - 8.0, 1.6), cmw, nmw, bw[::2], '$\\Delta\\omega_{500}$ (Pa s$^{-1}$; positive = sinking)',
      ['−0.15', '−0.06', '−0.01', '0.03', '0.10'])

# ---------------------------------------------------------------- i: WRF station columns at 13-15 LT, with and without the storm
Sr = pd.read_csv(f'{WRX}/wrx_series.csv'); aft = Sr[Sr.h.between(-1, 1)]
VARS = [('T2', '2-m temperature (°C)', 1, 1), ('T850', '850-hPa temperature (°C)', 1, 1), ('Q2', '2-m humidity (g kg$^{-1}$)', 1, 1),
        ('SWDOWN', 'Sunshine (W m$^{-2}$)', 1, 0), ('CLDMAX', 'Cloud fraction, max. (%)', 100, 0), ('PBLH', 'Mixed-layer depth (m)', 1, 0),
        ('ONSHORE', 'Onshore wind, coast (%)', 100, 0)]
nV = len(VARS); y0i, hi_ = 9.0, 38.0; rh = hi_ / nV; XI, WI = 89.0, 22.0
for k, (vn, lab, f, nd) in enumerate(VARS):
    ax = fig.add_axes(box(XI, y0i + hi_ - (k + 1) * rh + 0.6, WI, rh - 1.2))
    off = 273.15 if vn in ('T2', 'T850') else 0.0
    a_ = [-off + f * aft[(aft.case == int(p.split('|')[0])) & (aft.run == p.split('|')[1].split('-')[0])][vn].mean() for p in PAIRS]
    b_ = [-off + f * aft[(aft.case == int(p.split('|')[0])) & (aft.run == p.split('|')[1].split('-')[1])][vn].mean() for p in PAIRS]
    A_, B_ = np.mean(a_), np.mean(b_)
    ax.plot([B_, A_], [0, 0], color='#BDBDBD', lw=1.6, zorder=1, solid_capstyle='butt')
    ax.scatter(b_, np.full(len(b_), -0.32), s=4, color=BLUE, alpha=0.6, lw=0, zorder=2)
    ax.scatter(a_, np.full(len(a_), 0.32), s=4, color=RED, alpha=0.6, lw=0, zorder=2)
    ax.scatter([B_], [0], s=16, color=BLUE, zorder=3, lw=0); ax.scatter([A_], [0], s=16, color=RED, zorder=3, lw=0)
    lo_, hi2 = min(a_ + b_), max(a_ + b_); rng_ = (hi2 - lo_) or 1.0
    ax.set_xlim(lo_ - 0.05 * rng_, hi2 + 0.05 * rng_); ax.set_ylim(-0.7, 0.7); ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_visible(False); [sp.set_visible(False) for sp in ax.spines.values()]
    for val, xpos, ha in ((min(A_, B_), -0.06, 'right'), (max(A_, B_), 1.06, 'left')):
        ax.text(xpos, 0.5, f'{val:.{nd}f}', transform=ax.transAxes, ha=ha, va='center', fontsize=6.2, color=RED if val == A_ else BLUE)
    ax.text(-0.36, 0.5, lab, transform=ax.transAxes, ha='right', va='center', fontsize=6.2)
    ax.text(1.42, 0.5, f'{A_ - B_:+.{nd}f}'.replace('-', '−'), transform=ax.transAxes, ha='left', va='center', fontsize=6.2,
            fontweight='bold', color=RED if A_ - B_ > 0 else BLUE)
fig.legend(handles=[Line2D([], [], ls='none', marker='o', ms=3.5, color=RED, label='With storm'),
                    Line2D([], [], ls='none', marker='o', ms=3.5, color=BLUE, label='Storm removed')],
           loc='upper left', bbox_to_anchor=((XI - 6.0) / W, (y0i + hi_ + 4.6) / H), ncol=2, frameon=False, fontsize=6.2,
           handletextpad=0.2, columnspacing=1.0, borderaxespad=0.0)

# ---------------------------------------------------------------- j: warming at 2 m and at 850 hPa in the four estimates
P = pd.read_csv(f'{GCX}/fp_case.csv', keep_default_na=False, na_values=['']); P['variant'] = P.variant.fillna('NULL')
P = P[P.kind == 'heat']
rows = []
fp = pd.read_csv(f'{OBS}/fingerprints.csv').set_index('label')
rows.append(('Stations\n', fp.loc['east Tmax', 'excess'], fp.loc['east T850', 'excess']))
for m_, lab in (('small', '1°\n'), ('oper', '0.25°\n')):
    q_ = P[P.model == m_].pivot_table(index='case', columns='variant', values=['t2m', 't850'])
    rows.append((lab, (q_['t2m']['CTL'] - q_['t2m']['WIDE']).mean(), (q_['t850']['CTL'] - q_['t850']['WIDE']).mean()))
def pdiff(p, vn):
    c_, (r1, r2) = int(p.split('|')[0]), p.split('|')[1].split('-')
    return aft[(aft.case == c_) & (aft.run == r1)][vn].mean() - aft[(aft.case == c_) & (aft.run == r2)][vn].mean()
rows.append(('WRF\n3 km', np.mean([pdiff(p, 'T2') for p in PAIRS]), np.mean([pdiff(p, 'T850') for p in PAIRS])))
axj = fig.add_axes(box(137.5, 16.0, 40.5, 32.0))
x = np.arange(len(rows)); wb = 0.36
tot = np.array([r[1] for r in rows]); t85 = np.array([r[2] for r in rows])
axj.bar(x - wb / 2, tot, wb, color='#B2182B', label='2 m (Tmax, stations)', zorder=2)
axj.bar(x + wb / 2, t85, wb, color='#F4A582', label='850 hPa', zorder=2)
for i_, (tt, f8) in enumerate(zip(tot, t85)):
    axj.text(i_ - wb / 2, tt + 0.04, f'{tt:.1f}', ha='center', va='bottom', fontsize=6.2)
    axj.text(i_ + wb / 2, f8 + 0.04, f'{f8:.1f}', ha='center', va='bottom', fontsize=6.2)
axj.axhline(0, color='#555555', lw=0.5)
axj.set_xticks(x); axj.set_xticklabels([r[0] for r in rows], fontsize=6.2, linespacing=0.95); axj.tick_params(axis='x', length=0, pad=2)
axj.set_ylabel('Warming (K)')
axj.set_ylim(0, max(tot.max(), t85.max()) * 1.38); axj.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
axj.legend(loc='upper left', frameon=False, fontsize=6.2, handlelength=1.0, borderaxespad=0.0, ncol=2, columnspacing=0.8)
for sp in ('top', 'right'): axj.spines[sp].set_visible(False)
trj = axj.get_xaxis_transform()
axj.plot([0.72, 2.28], [-0.135, -0.135], color='#555555', lw=0.5, transform=trj, clip_on=False)
axj.text(1.5, -0.15, 'GraphCast', transform=trj, ha='center', va='top', fontsize=6.2)

# panel letters: rows share heights
yl2 = R2B + R2H + 3.2; yl3 = Y3 + HH + 5.4
for s, x0 in (('e', 0.5), ('f', 72.5), ('g', 123.0)): letter(s, x0, yl2)
for s, x0 in (('h', 0.5), ('i', 53.5), ('j', 127.5)): letter(s, x0, yl3)

os.makedirs(OUT, exist_ok=True)
print('station mean at V:', {c: round(float(D[c].get(0, np.nan)), 2) for c in D}, 'mean', round(float(m.get(0, np.nan)), 2))
print('two levels', [(r[0].replace('\n', ' '), round(r[1], 2), round(r[2], 2)) for r in rows])
print('size mm', W, round(H, 1))
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/FigS11.png', dpi=600); fig.savefig(f'{OUT}/FigS11.pdf')
