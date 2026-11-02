"""v10 Fig. 2: the storm itself adds the heat (GraphCast on all coasts, WRF at 3 km on six coasts).
a: effect of the storm on 2-m temperature in each region of the global map, 0.25-deg GraphCast (v09 Fig. 5a); the six coasts run
   in WRF are ringed;
b: WRF at 3 km, effect of the storm (control minus storm removed) on 2-m temperature at the case time, mean over the cases of
   each coast, with the change in 10-m wind; boxes: the region's 1-deg cells (Guangdong: the 19 stations);
c: case by case, the effect on the daily maximum in WRF against the effect at 14 LT in GraphCast for the same day (region-free
   removal, 0.25 deg; Guangdong: core to full removal of the pilot), with the null edits;
d: pooled effects for warm regions <30 deg, warm regions >=30 deg and cool regions: GraphCast removals of increasing size and the
   null edit (all usable case days, regions count equally, 95% intervals from resampling storms within regions), and WRF
   (17 cases; diamonds: single cases); dashes: change in the map for the same days.
Inputs: P9/data/ggc (all_oper_cases.csv, all_oper_regions.csv), P9/data/graphcast/gcx_case.csv (Guangdong pilot), P9/data/obs/stations.csv,
WRFDIR (wrfnc_effects.csv, wrfnc_maps.npz of the multi-coast runs), WRFIN (wrfnc_inputs: wrfnc_cases.csv, cells_*.csv).
Usage: python3 fig2_storm_removal.py  (environment: P9, WRFDIR, WRFIN, FIGOUT)"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
P9 = os.environ.get('P9', '.')
os.environ.setdefault('GGC', f'{P9}/data/ggc')
from ggc_common import CEN, NUM, LPOS, GD, regions
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

GGC = os.environ['GGC']
WRFDIR = os.environ.get('WRFDIR', 'data/wrfnc')
print('WRF input', WRFDIR)
WRFIN = os.environ.get('WRFIN', 'data/wrfnc_inputs')
OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(11)
O = regions('all', 'oper'); U = O[O.usable]
CO = pd.read_csv(f'{GGC}/all_oper_cases.csv')
CO['lat_r'] = CO.region.map(lambda r: CEN[r][1]); CO['band'] = np.where(CO.lat_r.abs() < 30, 'low', 'mid')
CASES = pd.read_csv(f'{WRFIN}/wrfnc_cases.csv').set_index('case')
E = pd.read_csv(f'{WRFDIR}/wrfnc_effects.csv')
COASTS = [('W17', 'NW Mexico'), ('W3', 'N Mozambique'), ('W5', 'SE India'), ('GD', 'Guangdong'), ('W22', 'NE United States'),
          ('C16', 'SE United States')]
GRP = {'W17': 'wl', 'W3': 'wl', 'W5': 'wl', 'GD': 'wl', 'W22': 'wm', 'C16': 'co'}
WRF_REGION = {'W17': 'W17', 'W3': 'W3', 'W5': 'W5', 'W22': 'W22', 'C16': 'C16'}

# GraphCast values of the WRF case days
gx = CO.set_index('case')
P = pd.read_csv(f'{P9}/data/graphcast/gcx_case.csv'); P = P[P.model == 'oper']
gd = {}
for c in (6, 16):
    q = P[P.case == c].groupby('variant').t2m.mean()
    gd[c] = (q['CTL'] - q['NOTC'], q['CTL'] - q['WIDE'])            # core, full (pilot; no region-free variant)
print('Guangdong pilot GraphCast core/full', {k: tuple(round(x, 2) for x in v) for k, v in gd.items()})

# ---------------------------------------------------------------- layout (mm)
W = 180.0
pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
MW = 174.0; MH = MW * (Y1 - Y0) / (X1 - X0)
AX0 = 3.5                                             # left edge of the map in a; b spans the same width

# b: one map per coast, equal height, widths from each map's extent, the row flush with the map frame of a
MAPC = next((f'{d}/wrfnc_maps_coast.npz' for d in (WRFDIR,) if os.path.exists(f'{d}/wrfnc_maps_coast.npz')), None)
ZC = np.load(MAPC) if MAPC else None                  # compact coast means (decode_coast_maps.py): T2 0.15 deg, wind 0.75 deg
ST = pd.read_csv(f'{P9}/data/obs/stations.csv')
MERC = ccrs.Mercator()
PB = lambda e: MERC.transform_points(pc, np.array([e[0], e[1]]), np.array([e[2], e[3]]))


def need(coast):                                      # region cells (Guangdong: stations) plus a margin
    q = ST if coast == 'GD' else pd.read_csv(f'{WRFIN}/cells_{coast}.csv')
    h = 0.0 if coast == 'GD' else 0.5
    return [q.lon.min() - h - 0.3, q.lon.max() + h + 0.3, q.lat.min() - h - 0.3, q.lat.max() + h + 0.3]


def map_box(coast, amin=0.72, amax=1.35):
    """largest box inside the 3-km domain without missing values that holds the region, cropped to an aspect amin..amax"""
    nd = need(coast)
    if ZC is None or f'{coast}__T2' not in ZC:
        e = [nd[0] - 2.0, nd[1] + 2.0, nd[2] - 2.0, nd[3] + 2.0]
    else:
        lo, la, ok = ZC[f'{coast}__lon'], ZC[f'{coast}__lat'], np.isfinite(ZC[f'{coast}__T2'])
        i0, i1, j0, j1 = 0, len(lo) - 1, 0, len(la) - 1
        while not ok[j0:j1 + 1, i0:i1 + 1].all():       # drop the edge row or column with most missing values
            cand = []
            if lo[i0 + 1] <= nd[0]: cand.append(((~ok[j0:j1 + 1, i0]).sum(), 'i0'))
            if lo[i1 - 1] >= nd[1]: cand.append(((~ok[j0:j1 + 1, i1]).sum(), 'i1'))
            if la[j0 + 1] <= nd[2]: cand.append(((~ok[j0, i0:i1 + 1]).sum(), 'j0'))
            if la[j1 - 1] >= nd[3]: cand.append(((~ok[j1, i0:i1 + 1]).sum(), 'j1'))
            s = max(cand)[1]
            i0, i1, j0, j1 = i0 + (s == 'i0'), i1 - (s == 'i1'), j0 + (s == 'j0'), j1 - (s == 'j1')
        e = [float(lo[i0]), float(lo[i1]), float(la[j0]), float(la[j1])]
    p = PB(e); a = (p[1, 1] - p[0, 1]) / (p[1, 0] - p[0, 0]); q = PB(nd)
    if a > amax:
        hh = amax * (p[1, 0] - p[0, 0]); y0 = max(p[0, 1], min(0.5 * (q[0, 1] + q[1, 1]) - hh / 2, p[1, 1] - hh))
        e[2], e[3] = [pc.transform_point(0, y, MERC)[1] for y in (y0, y0 + hh)]
    elif a < amin:
        ww = (p[1, 1] - p[0, 1]) / amin; x0 = max(p[0, 0], min(0.5 * (q[0, 0] + q[1, 0]) - ww / 2, p[1, 0] - ww))
        e[0], e[1] = [pc.transform_point(x, 0, MERC)[0] for x in (x0, x0 + ww)]
    p = PB(e)
    return e, (p[1, 1] - p[0, 1]) / (p[1, 0] - p[0, 0])


EXT = {c: map_box(c) for c, _ in COASTS}
BG_ = 2.0
R2H = (MW - (len(COASTS) - 1) * BG_) / sum(1 / a for _, a in EXT.values())
BXS = np.cumsum([AX0] + [R2H / EXT[c][1] + BG_ for c, _ in COASTS[:-1]])
print('b map height', round(R2H, 1), 'widths', [round(R2H / EXT[c][1], 1) for c, _ in COASTS])
YB3, R3H = 14.0, 42.0                                 # c, d
YB2 = YB3 + R3H + 9.0                                 # b
H = YB2 + R2H + 6.5 + MH + 2.5
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()
bounds = np.arange(-2.0, 2.01, 0.5); cmap, norm = binned('RdBu_r', bounds)

# ---------------------------------------------------------------- a: GraphCast map
axa = fig.add_axes(box(AX0, H - 2.5 - MH, MW, MH), projection=proj)
axa.set_xlim(X0, X1); axa.set_ylim(Y0, Y1)
axa.set_xticks([]); axa.set_yticks([])                  # GeoAxes keeps hidden default ticks that the QA would read
axa.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND, edgecolor='none', zorder=0)
axa.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#555555', zorder=1)
axa.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=(0, (2, 2)), xlocs=range(-180, 181, 30), ylocs=[-30, 0, 30], zorder=0.5)
for la_ in (-30, 0, 30):
    x, y = proj.transform_point(-168, la_, pc)
    axa.text(x, y, {-30: '30° S', 0: '0°', 30: '30° N'}[la_], fontsize=6.5, color='#666666', ha='left', va='bottom', zorder=6)
for s in axa.spines.values():
    s.set_linewidth(0.5)
for r, (lo_, la_) in CEN.items():
    warm = r[0] == 'W'; mk = 'o' if warm else 's'
    if r not in O.index:
        axa.plot(lo_, la_, marker='x', ms=3.2, mec='#9E9E9E', mew=0.7, ls='none', transform=pc, zorder=5)
        continue
    row = O.loc[r]
    if not row.usable:
        axa.plot(lo_, la_, marker=mk, ms=3.0, mfc='#BDBDBD', mec='#8C8C8C', mew=0.5, ls='none', transform=pc, zorder=5)
        continue
    col = cmap(norm(row.d_free)); met = bool(row.consistent)
    axa.plot(lo_, la_, marker=mk, ms=6.0 if warm else 5.4, mfc=col if met else 'white', mec=col, mew=0.6 if met else 1.3,
             ls='none', transform=pc, zorder=6)
    lx, ly = LPOS[r]
    axa.plot([lo_, lx], [la_, ly], color='k', lw=0.35, transform=pc, zorder=5.5)
    axa.text(lx, ly, NUM(r), fontsize=6.5, ha='center', va='center', transform=pc, zorder=8, path_effects=HALO,
             fontweight='bold' if r in WRF_REGION.values() else 'normal')
axa.plot(*GD, marker='*', ms=9, mfc='none', mec='k', mew=0.8, transform=pc, zorder=7)
axa.text(EXT['GD'][0][1] + 2.0, GD[1] - 0.5, 'GD', fontsize=6.5, fontweight='bold', ha='left', va='center', transform=pc, zorder=8,
         path_effects=HALO)                                   # Guangdong pilot coast (not one of the numbered regions)
for coast, _ in COASTS:                                  # areas of the maps in b
    e = EXT[coast][0]; t = np.linspace(0, 1, 40)
    bl = np.r_[e[0] + (e[1] - e[0]) * t, np.full(40, e[1]), e[1] - (e[1] - e[0]) * t, np.full(40, e[0])]
    bb = np.r_[np.full(40, e[2]), e[2] + (e[3] - e[2]) * t, np.full(40, e[3]), e[3] - (e[3] - e[2]) * t]
    axa.plot(bl, bb, color='k', lw=0.8, transform=pc, zorder=5.8)
axa.text(0.006, 0.03, 'GraphCast (0.25°): effect of the TC on 2-m temperature near 14 LT', transform=axa.transAxes, fontsize=7,
         ha='left', va='bottom', zorder=8, path_effects=HALO)
at = lambda lo_, la_: inv.transform(axa.transData.transform(proj.transform_point(lo_, la_, pc)))
(cx0, cy0), (cx1, cy1) = at(-152, -9), at(-98, -5.5)
cax = fig.add_axes([cx0, cy0, cx1 - cx0, cy1 - cy0])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.ax.tick_params(labelsize=6.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels(['−2', '−1', '0', '1', '2'])
cb.set_label('With the TC minus TC removed (K)', fontsize=6.5, labelpad=1.5)

# ---------------------------------------------------------------- b: WRF maps, one per coast
EW = E[E.pair == 'ctl-widel'].copy(); EW['grp'] = EW.coast.map(GRP)
Qv = None
for j, (coast, nm) in enumerate(COASTS):
    e, a_ = EXT[coast]; bw = R2H / a_
    ax = fig.add_axes(box(BXS[j], YB2, bw, R2H), projection=MERC)
    p = PB(e); ax.set_xlim(p[0, 0], p[1, 0]); ax.set_ylim(p[0, 1], p[1, 1]); ax.set_aspect('auto')
    if ZC is not None and f'{coast}__T2' in ZC:
        mx_, my_, D = ZC[f'{coast}__lon'], ZC[f'{coast}__lat'], ZC[f'{coast}__T2']
        ax.contourf(mx_, my_, D, levels=bounds, cmap=cmap, norm=norm, extend='both', transform=pc, zorder=1)
        wl, wa = ZC[f'{coast}__wlon'], ZC[f'{coast}__wlat']
        kk = (np.round((wl - wl.min()) / 0.75) % 2 == 0) & (np.round((wa - wa.min()) / 0.75) % 2 == 0)   # every 1.5 deg
        kk &= (wl > e[0] + 0.3) & (wl < e[1] - 0.3) & (wa > e[2] + 0.3) & (wa < e[3] - 0.3)
        du_, dv_ = ZC[f'{coast}__du'][kk], ZC[f'{coast}__dv'][kk]; sp_ = np.hypot(du_, dv_)
        cap = np.where(sp_ > 5.0, 5.0 / np.maximum(sp_, 1e-6), 1.0)            # arrows longer than 5 m/s drawn at 5 m/s
        Qv = ax.quiver(wl[kk], wa[kk], du_ * cap, dv_ * cap, transform=pc, scale=20.0, scale_units='inches', units='inches',
                       width=0.0125, headwidth=3.0, headlength=3.4, headaxislength=3.0, color='#222222', zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.55, edgecolor='k', zorder=3)
    if coast == 'GD':
        ax.plot(ST.lon, ST.lat, 'k.', ms=1.6, transform=pc, zorder=5)
    else:
        cl = pd.read_csv(f'{WRFIN}/cells_{coast}.csv')
        for a, b in zip(cl.lat, cl.lon):
            ax.add_patch(Rectangle((b - 0.5, a - 0.5), 1, 1, fc='none', ec='k', lw=0.35, transform=pc, zorder=5))
    for s_ in ax.spines.values():
        s_.set_linewidth(0.8)                                    # same weight as the boxes in a
    ax.text(0.04, 0.96, 'GD' if coast == 'GD' else NUM(coast), fontsize=6.5, fontweight='bold', ha='left', va='top',   # label as in a
            transform=ax.transAxes, zorder=8, path_effects=HALO)
    w = EW.loc[EW.coast == coast, 'Tmax']
    fig.text((BXS[j] + bw / 2) / W, (YB2 + R2H + 0.8) / H, f'{nm}\n{w.mean():+.1f} K ({len(w)})'.replace('-', '−') if len(w) else nm,
             fontsize=6.4, ha='center', va='bottom', linespacing=1.05, color=BLUE if coast == 'C16' else 'k')
if Qv is not None:                                               # key under the last map, flush right
    fig.text((AX0 + MW) / W, (YB2 - 1.6) / H, '5 m/s', fontsize=6.5, ha='right', va='center')
    fig.axes[-1].quiverkey(Qv, (AX0 + MW - 7.0) / W, (YB2 - 1.6) / H, 5.0, '', labelpos='E', coordinates='figure', labelsep=0.0)

# ---------------------------------------------------------------- c: WRF against GraphCast, case by case
CX, CW = 13.0, 48.0
axc = fig.add_axes(box(CX, YB3, CW, R3H))
xx = np.array([-4.5, 4.5]); axc.plot(xx, xx, color='#9E9E9E', lw=0.6, ls=(0, (3, 2)), zorder=1)
axc.axhline(0, color='#555555', lw=0.5, zorder=1); axc.axvline(0, color='#555555', lw=0.5, zorder=1)
STY = {'wl': dict(marker='o', mfc=RED, mec=RED, mew=0.6), 'wm': dict(marker='o', mfc='white', mec=RED, mew=1.0),
       'co': dict(marker='s', mfc='white', mec=BLUE, mew=1.0), 'nu': dict(marker='D', mfc=GREY, mec='k', mew=0.4)}
pts = []
for _, e in E.iterrows():
    c = int(e.case); null = e.pair == 'ctl-nullw'
    if e.pair not in ('ctl-widel', 'ctl-nullw'):
        continue
    g = 'nu' if null else GRP[e.coast]
    if e.coast == 'GD':
        lo_, hi_ = gd[c]; x = 0.5 * (lo_ + hi_)
        axc.plot([lo_, hi_], [e.Tmax, e.Tmax], color=RED, lw=0.8, zorder=2)
    elif c in gx.index:
        x = gx.loc[c, 'd_null' if null else 'd_free']
    else:
        continue
    axc.plot(x, e.Tmax, ms=4.2 if g != 'nu' else 3.4, ls='none', zorder=4, **STY[g])
    if not null:
        pts.append((x, e.Tmax, e.coast))
if pts:
    A = np.array([(p[0], p[1]) for p in pts]); same = int((np.sign(A[:, 0]) == np.sign(A[:, 1])).sum())
    r_ = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    from scipy.stats import spearmanr
    rho = spearmanr(A[:, 0], A[:, 1])[0]
    axc.text(0.03, 0.97, f'same sign: {same} of {len(A)}\nrank correlation {rho:.2f}'.replace('-', '−'), transform=axc.transAxes,
             fontsize=6.5, ha='left', va='top')
    print('panel c: cases', len(A), 'same sign', same, 'r', round(r_, 2), 'spearman', round(rho, 2))
lim = 4.2
axc.set_xlim(-lim, lim); axc.set_ylim(-lim, lim); axc.set_xticks([-4, -2, 0, 2, 4]); axc.set_yticks([-4, -2, 0, 2, 4])
axc.set_xticklabels(['−4', '−2', '0', '2', '4']); axc.set_yticklabels(['−4', '−2', '0', '2', '4'])
axc.set_xlabel('GraphCast, same day (K)', labelpad=1); axc.set_ylabel('WRF, daily maximum (K)', labelpad=1)
axc.text(3.2, 3.85, '1:1', fontsize=6.0, color='#7A7A7A', ha='right', va='center')
hd = [mlines.Line2D([], [], ls='none', ms=4.0, **STY[k]) for k in ('wl', 'wm', 'co', 'nu')]
axc.legend(hd, ['warm, <30°', 'warm, ≥30°', 'cool, ≥30°', 'null edit'], loc='lower right', frameon=False, fontsize=6.8, handletextpad=0.2,
           borderaxespad=0.1, labelspacing=0.2)

# ---------------------------------------------------------------- d: pooled effects by group
def pooled(d, col, n=2000):
    d = d[np.isfinite(d[col])]
    regs = {r: [g[col].values for _, g in q.groupby('sid')] for r, q in d.groupby('region')}
    m = np.mean([np.mean([s.mean() for s in v]) for v in regs.values()])
    bs = []
    for _ in range(n):
        bs.append(np.mean([np.mean([v[i].mean() for i in rng.integers(0, len(v), len(v))]) for v in regs.values()]))
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


CU = CO[CO.region.isin(U.index)]
GROUPS = [('wl', 'Warm regions, <30°', (CU.kind == 'warm') & (CU.band == 'low'), RED),
          ('wm', 'Warm regions, ≥30°', (CU.kind == 'warm') & (CU.band == 'mid'), RED),
          ('co', 'Cool regions', CU.kind == 'cool', BLUE)]
BARS = [('d_core', 'core', 0.45), ('d_free', 'region-free', 0.75), ('d_full', 'full', 1.0), ('d_null', 'null edit', None)]
DX, DW = 76.0, 101.0
YL = (-1.6, 3.0)
axd = fig.add_axes(box(DX, YB3, DW, R3H))
nb = len(BARS) + 1; gw = nb + 1.2


def shade(col, a):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(col)); return tuple(1 - a * (1 - c))


for gi, (key, gname, sel, col) in enumerate(GROUPS):
    x0 = gi * gw
    d = CU[sel]
    mp = d.groupby(['region', 'sid']).anom.mean().groupby('region').mean().mean()
    for bi, (cname, lab, a) in enumerate(BARS):
        m, l_, h_ = pooled(d, cname)
        fc = GREY if a is None else shade(col, a)
        axd.bar(x0 + bi, m, 0.8, color=fc, edgecolor='none', zorder=2)
        axd.errorbar(x0 + bi, m, yerr=[[m - l_], [h_ - m]], fmt='none', ecolor='k', elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3)
        print('panel d GraphCast', key, cname, round(m, 2), round(l_, 2), round(h_, 2), 'regions', d.region.nunique())
    axd.plot([x0 - 0.45, x0 + len(BARS) - 0.55], [mp, mp], color='k', lw=0.8, ls=(0, (2, 1.2)), zorder=4)
    w = EW[EW.grp == key]
    if len(w):
        xw = x0 + len(BARS)
        axd.bar(xw, w.Tmax.mean(), 0.8, color='white', edgecolor=col, lw=1.0, hatch='////', zorder=2)
        jit = (np.arange(len(w)) - (len(w) - 1) / 2) * 0.08
        axd.plot(xw + jit, w.Tmax.clip(lower=YL[0] + 0.08), marker='D', ms=2.2, mfc='white', mec='k', mew=0.45, ls='none', zorder=4)
        for xj, v_ in zip(xw + jit, w.Tmax):
            if v_ < YL[0] + 0.08:                                # off scale below: value next to the clipped diamond
                axd.text(xj, YL[0] + 0.32, f'{v_:.1f}'.replace('-', '−'), fontsize=6.5, ha='center', va='center')
        wa = w.case.map(CASES.anom).dropna()
        if len(wa):
            ma = wa.mean()
            if ma < YL[0] + 0.15:                                  # off scale below: arrow and value
                axd.annotate('', xy=(xw - 0.75, YL[0] + 0.02), xytext=(xw - 0.75, YL[0] + 0.42), zorder=4,
                             arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', lw=0.8, color='k'))
                axd.text(xw - 0.95, YL[0] + 0.24, f'WRF days\nin the map: {ma:+.1f} K'.replace('-', '−'), fontsize=6.5, ha='right', va='center', linespacing=1.0)
            elif ma < YL[1] - 0.15:
                axd.plot([xw - 0.45, xw + 0.45], [ma, ma], color='k', lw=0.8, ls=(0, (2, 1.2)), zorder=4)
            else:                                              # off scale: arrow and value
                axd.annotate('', xy=(xw, YL[1] - 0.02), xytext=(xw, YL[1] - 0.4), zorder=4,
                             arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', lw=0.8, color='k'))
                axd.text(xw + 0.3, YL[1] - 0.2, f'WRF days\nin the map: {ma:+.1f} K', fontsize=6.5, ha='left', va='center', linespacing=1.0)
        print('panel d WRF', key, 'n', len(w), 'mean', round(w.Tmax.mean(), 2), 'map anomaly of these days', round(wa.mean(), 2) if len(wa) else None)
    axd.text(x0 + (nb - 1) / 2, 1.0, gname, transform=axd.get_xaxis_transform(), ha='center', va='bottom', fontsize=7.0, color=col)
axd.axhline(0, color='#555555', lw=0.5)
axd.set_xticks([]); axd.set_ylim(*YL); axd.set_yticks([-1, 0, 1, 2, 3]); axd.set_yticklabels(['−1', '0', '1', '2', '3'])
axd.set_xlim(-0.7, (len(GROUPS) - 1) * gw + nb - 0.3); axd.set_ylabel('Effect of the TC (K)', labelpad=1)
for s_ in ('top', 'right', 'bottom'):
    axd.spines[s_].set_visible(False)
nW = int((EW.Tmax.notna()).sum())
hd = [Rectangle((0, 0), 1, 1, fc=shade(RED, a) if a else GREY, ec='none') for _, _, a in BARS]
hd += [Rectangle((0, 0), 1, 1, fc='white', ec='#555555', hatch='////'), mlines.Line2D([], [], color='k', lw=0.8, ls=(0, (2, 1.2)))]
axd.legend(hd, ['GraphCast: core removed', 'region-free removal', 'full removal', 'null edit', f'WRF, {nW} days', 'change in the map'],
           loc='upper left', bbox_to_anchor=(0.0, -0.03), frameon=False, fontsize=7.0, ncol=3, handlelength=1.3, handletextpad=0.4,
           borderaxespad=0.1, labelspacing=0.25, columnspacing=1.0)

# panel letters
fig.text(1.0 / W, (H - 1.0) / H, 'a', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text(1.0 / W, (YB2 + R2H + 5.5) / H, 'b', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text(1.0 / W, (YB3 + R3H + 6.0) / H, 'c', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text((DX - 11.0) / W, (YB3 + R3H + 6.0) / H, 'd', fontsize=9, fontweight='bold', va='top', ha='left')
print('size mm', W, round(H, 1))
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/Fig2.png', dpi=600); fig.savefig(f'{OUT}/Fig2.pdf')
