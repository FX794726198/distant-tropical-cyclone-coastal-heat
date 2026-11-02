"""v10 Fig. 3: the storm's outer winds decide which coasts warm (from v09 Fig. 6, with stations and the six-coast WRF runs).
a: map of the 41 regions (colour = change in the map, Fig. 1a,b) with the idealized outer-circulation wind of a storm in the
   region's direction (brown: blows off the land; teal: onto the land);
b: that wind's onshore part against the change in the map (41 regions);
c: the same, from the storm positions of the GraphCast cases, against the GraphCast effect (usable regions);
d: weather stations on all coasts: change in daily maximum temperature binned by the onshore part of the outer wind
   (<30 deg and 30-45 deg; boxes of 2.5 deg count equally; 90% intervals from resampling years; ISD-Lite, 12_isd_hourly.py);
e: the same stations: measured afternoon onshore wind and dew point;
f: what the storm changes over the regions in the six-coast WRF runs (3 km, 08-18 LT means, rain summed; sinking at 500 hPa
   was dropped: at 3 km its regional mean is as large in the null edits; bars: group means, diamonds:
   single cases) for warm coasts <30 deg (W17, W3, W5, Guangdong), warm >=30 deg (W22), cool (C16) and the null edits;
   v09's GraphCast group bars (Fig. 6d) go to the SI;
g: storm-relative composite of the GraphCast changes for the warm regions <30 deg (storm on the +x axis, Southern
   Hemisphere mirrored).  The >=30 deg and cool composites go to the SI.
Inputs: MECH (P9/data/mech: geometry_map_regions.csv, geometry_cases_oper.csv, geometry_flagged_cases.csv,
coast_normals.csv, stormrel_oper.npz), ISD (P9/data/isd/isd_coast_bins.csv), WRFEFF (wrfnc_effects.csv of the multi-coast
runs; 11-02: early analysis of all 17 cases, wrfnc_out/early; final run adds the 419 null edit).
Usage: python3 fig3_coastal_orientation.py  (environment: P9, MECH, ISD, WRFEFF, FIGOUT)"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
P9 = os.environ.get('P9', '.')
from ggc_common import CEN, NUM
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter

MECH = os.environ.get('MECH', f'{P9}/data/mech')
ISD = os.environ.get('ISD', f'{P9}/data/isd')
WRFEFF = os.environ.get('WRFEFF', 'data/wrfnc/wrfnc_effects.csv')
print('WRF input', WRFEFF)
OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
OFF, ON = '#8C510A', '#01665E'                      # outer circulation blows off the land / onto the land
G = pd.read_csv(f'{MECH}/geometry_map_regions.csv').set_index('region')
flag = set(pd.read_csv(f'{MECH}/geometry_flagged_cases.csv').case)
GC = pd.read_csv(f'{MECH}/geometry_cases_oper.csv'); GC = GC[~GC.case.isin(flag)]
GC['lat_r'] = GC.region.map(G.lat); GC['band'] = np.where(GC.lat_r.abs() < 30, 'low', 'mid')
rng = np.random.default_rng(7)
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
RHO500 = 0.69                                       # kg m-3, to turn WRF w (m/s) into omega (Pa/s) at 500 hPa
WGROUP = {'W17': 'wl', 'W3': 'wl', 'W5': 'wl', 'GD': 'wl', 'W22': 'wm', 'C16': 'co'}


def wind_dir(sector, lat):
    """unit vector (east, north) of the idealized outer-circulation wind at the region (tangential + 0.4 inflow)"""
    b = np.radians(SECT.index(sector) * 45.0); r = np.array([np.sin(b), np.cos(b)])
    t = np.array([r[1], -r[0]]) if lat >= 0 else np.array([-r[1], r[0]])
    v = t + 0.4 * r
    return v / np.hypot(*v)


def region_means(d, cols):
    return d.groupby(['region', 'sid'])[cols].mean().groupby('region').mean()


def pooled_ci(d, col, n=2000):
    """region-equal mean and 95% interval from resampling TCs within regions"""
    regs = {r: [g[col].values for _, g in q.groupby('sid')] for r, q in d.groupby('region')}
    m = np.mean([np.mean([s.mean() for s in v]) for v in regs.values()])
    bs = []
    for _ in range(n):
        vals = []
        for v in regs.values():
            pick = rng.integers(0, len(v), len(v)); vals.append(np.mean([v[i].mean() for i in pick]))
        bs.append(np.mean(vals))
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ---------------------------------------------------------------- layout (mm)
W = 180.0
pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
MW = 174.0; MH = MW * (Y1 - Y0) / (X1 - X0)
YB3, R3H = 11.0, 37.0                                # third row: processes and composite
YB2, R2H = YB3 + R3H + 20.5, 33.0                    # second row: the rule in four data sets
H = YB2 + R2H + 6.0 + MH + 2.5
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()
fmt = lambda v: f'{v:g}'.replace('-', '−')

# ---------------------------------------------------------------- a: map
bounds = np.arange(-2.0, 2.01, 0.5); cmap, norm = binned('RdBu_r', bounds)
axa = fig.add_axes(box(3.5, H - 2.5 - MH, MW, MH), projection=proj)
axa.set_xlim(X0, X1); axa.set_ylim(Y0, Y1); axa.set_xticks([]); axa.set_yticks([])
axa.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND, edgecolor='none', zorder=0)
axa.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#555555', zorder=1)
axa.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=(0, (2, 2)), xlocs=range(-180, 181, 30), ylocs=[-30, 0, 30], zorder=0.5)
for la_ in (-30, 0, 30):
    x, y = proj.transform_point(-168, la_, pc)
    axa.text(x, y, {-30: '30° S', 0: '0°', 30: '30° N'}[la_], fontsize=6.5, color='#666666', ha='left', va='bottom', zorder=6)
for s in axa.spines.values():
    s.set_linewidth(0.5)
L = 6.0                                              # arrow length (degrees of latitude)
for r, (lo_, la_) in CEN.items():
    g = G.loc[r]; v = wind_dir(g.sector, g.lat); col = OFF if g.imp04 < 0 else ON
    x0, y0 = proj.transform_point(lo_, la_, pc)
    x1, y1 = proj.transform_point(lo_ + L * v[0] / np.cos(np.radians(la_)), la_ + L * v[1], pc)
    axa.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=5.5, lw=0.9, color=col, zorder=5,
                                  shrinkA=2.0, shrinkB=0))
    warm = r[0] == 'W'
    axa.plot(lo_, la_, marker='o' if warm else 's', ms=4.6 if warm else 4.0, mfc=cmap(norm(g.ex_map)), mec='k', mew=0.35,
             ls='none', transform=pc, zorder=6)
for r, dlo, dla in (('W13', 4.5, -3.5), ('W22', -5.5, 2.5), ('W23', 3.5, 3.0)):   # mid-latitude coasts that break the rule
    lo_, la_ = CEN[r]
    axa.text(lo_ + dlo, la_ + dla, NUM(r), fontsize=6.3, color=RED, ha='center', va='center', transform=pc, zorder=8, path_effects=HALO)
at = lambda lo_, la_: inv.transform(axa.transData.transform(proj.transform_point(lo_, la_, pc)))
(cx0, cy0), (cx1, cy1) = at(-152, -9), at(-98, -5.5)
cax = fig.add_axes([cx0, cy0, cx1 - cx0, cy1 - cy0])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.ax.tick_params(labelsize=6.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels(['−2', '−1', '0', '1', '2'])
cb.set_label('TC days minus TC-free days (K)', fontsize=6.5, labelpad=1.5)
for la_, txt, col in ((-20.5, 'outer wind blows off the land', OFF), (-25, 'outer wind blows onto the land', ON)):
    kx, ky = at(-152, la_); kx2, _ = at(-145, la_)
    fig.add_artist(FancyArrowPatch((kx, ky), (kx2, ky), arrowstyle='-|>', mutation_scale=5.5, lw=0.9, color=col,
                                   transform=fig.transFigure))
    fig.text(kx2 + 0.004, ky, txt, fontsize=6.5, ha='left', va='center', color=col)
kx, ky = at(-152, -29.5)
fig.text(kx, ky, 'circles: warm regions; squares: cool regions', fontsize=6.5, ha='left', va='center', color='#333333')


def xaxis_onshore(ax, label=True):
    ax.set_xlim(-1.12, 1.12); ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['−1\noff land', '0', '1\nonto land'], linespacing=0.95)
    tl = ax.get_xticklabels(); tl[0].set_color(OFF); tl[2].set_color(ON)
    if label:
        ax.set_xlabel('Outer wind, onshore part', labelpad=1, fontsize=7)


def corr_text(ax, items, x, y, ha):
    for k, (txt, col) in enumerate(items):
        ax.text(x, y - 0.09 * k, txt.replace('-', '−'), transform=ax.transAxes, ha=ha, va='top', fontsize=6.5, color=col)


def mk(sg, low, ms=3.4):
    col = RED if sg == 'warm' else BLUE
    return dict(marker='o' if sg == 'warm' else 's', ms=ms, mfc=col if low else 'white', mec=col, mew=0.6 if low else 0.9, ls='none')


# ---------------------------------------------------------------- b: the map
PW, PX = 30.5, [11.0, 54.5, 98.0, 141.5]            # second-row panels
axb = fig.add_axes(box(PX[0], YB2, PW, R2H))
for (sg, bd), g in G.groupby(['sign', 'band']):
    axb.plot(g.imp04, g.ex_map, zorder=3, **mk(sg, bd == 'low'))
axb.axhline(0, color='#555555', lw=0.5, zorder=1); axb.axvline(0, ymin=0.22, color='#555555', lw=0.5, zorder=1)
lo = G[G.band == 'low']; mi = G[G.band == 'mid']
corr_text(axb, [(f'<30°: $r$ = {lo.imp04.corr(lo.ex_map):.2f}', 'k'), (f'≥30°: $r$ = {mi.imp04.corr(mi.ex_map):.2f}', '#555555')],
          0.03, 0.19, 'left')
for r, dx, dy in (('W13', 0.0, 0.32), ('W22', 0.0, 0.32), ('W23', -0.17, 0.0)):
    g = G.loc[r]
    axb.text(g.imp04 + dx, g.ex_map + dy, NUM(r), fontsize=6.0, color=RED, ha='center', va='center', zorder=5)
xaxis_onshore(axb); axb.set_ylim(-2.35, 2.25); axb.set_yticks([-2, -1, 0, 1, 2])
axb.set_ylabel('Change in the map (K)', labelpad=1)
axb.set_title('Reanalysis, 41 regions', fontsize=6.8, pad=2, loc='left')

# ---------------------------------------------------------------- c: GraphCast cases
axc = fig.add_axes(box(PX[1], YB2, PW, R2H))
Q = region_means(GC, ['t2m', 'imp04', 'onshore']); Q['kind'] = GC.groupby('region').kind.first(); Q['band'] = GC.groupby('region').band.first()
for (k, bd), g in Q.groupby(['kind', 'band']):
    axc.plot(g.imp04, g.t2m, zorder=3, **mk(k, bd == 'low'))
axc.axhline(0, color='#555555', lw=0.5, zorder=1); axc.axvline(0, ymin=0.22, color='#555555', lw=0.5, zorder=1)
ql = Q[Q.band == 'low']
corr_text(axc, [(f'<30°: $r$ = {ql.imp04.corr(ql.t2m):.2f}', 'k'), (f'all: $r$ = {Q.imp04.corr(Q.t2m):.2f}', '#555555')], 0.03, 0.19, 'left')
xaxis_onshore(axc); axc.set_ylim(-1.15, 1.5); axc.set_yticks([-1, 0, 1])
axc.set_ylabel('Effect of the TC, 2 m (K)', labelpad=1)
axc.set_title(f'GraphCast, {len(Q)} regions', fontsize=6.8, pad=2, loc='left')
print('panel c regions', len(Q), 'corr low', round(ql.imp04.corr(ql.t2m), 2), 'all', round(Q.imp04.corr(Q.t2m), 2))

# ---------------------------------------------------------------- d, e: weather stations on all coasts
B = pd.read_csv(f'{ISD}/isd_coast_bins.csv')
XB = {'-1 to -0.6': -0.8, '-0.6 to -0.2': -0.4, '-0.2 to 0.2': 0.0, '0.2 to 0.6': 0.4, '0.6 to 1': 0.8}
B['x'] = B.bin.map(XB)
BANDS = [('lat < 30', '<30°', '#222222', True), ('lat 30-45', '30–45°', '#8C8C8C', False)]
NST = {'lat < 30': 761, 'lat 30-45': 606}           # stations (isd_result.txt)


def station_line(ax, var, scale=1.0, vline=True):
    for sub, lab, col, fill in BANDS:
        b = B[(B.subset == sub) & (B['var'] == var)].sort_values('x')
        ax.fill_between(b.x, b.lo * scale, b.hi * scale, color=col, alpha=0.18, lw=0, zorder=1)
        ax.plot(b.x, b.est * scale, color=col, lw=1.0, zorder=2)
        ax.plot(b.x, b.est * scale, marker='o', ms=3.0, mfc=col if fill else 'white', mec=col, mew=0.8, ls='none', zorder=3)
    ax.axhline(0, color='#555555', lw=0.5, zorder=0)
    if vline:
        ax.axvline(0, color='#555555', lw=0.5, zorder=0)


axd = fig.add_axes(box(PX[2], YB2, PW, R2H))
station_line(axd, 'Tmax', vline=False)
xaxis_onshore(axd); axd.set_ylim(-0.98, 0.85); axd.set_yticks([-0.5, 0, 0.5]); axd.set_yticklabels(['−0.5', '0', '0.5'])
axd.set_ylabel('Station max. temperature (K)', labelpad=1)
axd.set_title('Stations, all coasts', fontsize=6.8, pad=2, loc='left')
hd = [mlines.Line2D([], [], color=c, lw=1.0, marker='o', ms=3.0, mfc=c if f else 'white', mec=c, mew=0.8) for _, _, c, f in BANDS]
axd.legend(hd, [f'{l} ({NST[s]} stations)' for s, l, _, _ in BANDS], loc='lower left', frameon=False, fontsize=6.0, handlelength=1.4,
           handletextpad=0.3, borderaxespad=0.2, labelspacing=0.2)
EH = (R2H - 3.0) / 2
axe1 = fig.add_axes(box(PX[3], YB2 + EH + 3.0, PW, EH))
station_line(axe1, 'on_pm'); xaxis_onshore(axe1, label=False); axe1.set_xticklabels([])
axe1.set_ylim(-1.6, 1.6); axe1.set_yticks([-1, 0, 1]); axe1.set_yticklabels(['−1', '0', '1'])
axe1.set_ylabel('Onshore\nwind (m s$^{-1}$)', labelpad=1, fontsize=6.5, linespacing=0.95)
axe1.set_title('Same stations, afternoon', fontsize=6.8, pad=2, loc='left')
axe2 = fig.add_axes(box(PX[3], YB2, PW, EH))
station_line(axe2, 'Td_pm'); xaxis_onshore(axe2)
axe2.set_ylim(-0.5, 1.45); axe2.set_yticks([0, 1])
axe2.set_ylabel('Dew point\n(K)', labelpad=1, fontsize=6.5, linespacing=0.95)

# ---------------------------------------------------------------- f: what the storm changes in WRF
E = pd.read_csv(WRFEFF)
E['grp'] = np.where(E.pair == 'ctl-nullw', 'nu', E.coast.map(WGROUP))
E = E[E.pair.isin(['ctl-widel', 'ctl-nullw'])].copy()
day = lambda a, b: E[a] if a in E else E[b]                     # daytime means (v3 analysis) or values at V (v2)
E['f_tmax'] = E['Tmax']; E['f_t850'] = day('T850_day', 'T850_V')
E['f_rain'] = day('rain_day', 'rainacc_V')                      # rain over 08-18 LT (v3) or 24-h rain to V (v2), mm
E['f_onshore'] = day('onshore_day', 'onshore_V'); E['f_q2'] = day('Q2_day', 'Q2_V')
E['f_sw'] = day('SWDOWN_day', 'SW_V'); E['f_pbl'] = day('PBLH_day', 'PBLH_V')
grp = [('wl', 'warm, <30°', RED, True), ('wm', 'warm, ≥30°', RED, False), ('co', 'cool, ≥30°', BLUE, False), ('nu', 'null edit', GREY, True)]
VARS = [('tmax', 'Max. 2-m T\n(K)'), ('t850', '850-hPa T\n(K)'), ('rain', 'Rain\n(mm)'),
        ('onshore', 'Onshore\nwind (m s$^{-1}$)'), ('q2', '2-m humidity\n(g kg$^{-1}$)'), ('sw', 'Sunshine\n(W m$^{-2}$)'),
        ('pbl', 'Mixed\nlayer (m)')]
FX0, FW, FG = 11.5, 9.9, 4.8
NICE = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
for j, (v, lab) in enumerate(VARS):
    ax = fig.add_axes(box(FX0 + j * (FW + FG), YB3, FW, R3H))
    vals = [0.0]
    for i, (key, nm, col, fill) in enumerate(grp):
        w = E.loc[E.grp == key, f'f_{v}'].dropna()
        if not len(w):
            continue
        ax.bar(i, w.mean(), 0.66, color=col if fill else 'white', edgecolor=col, lw=0.8, alpha=0.85 if fill else 1, zorder=2)
        jit = (np.arange(len(w)) - (len(w) - 1) / 2) * 0.09
        ax.plot(i + jit, w, marker='D', ms=2.2, mfc='white', mec='k', mew=0.45, ls='none', zorder=4)
        vals += list(w)
        print('panel f WRF', v, nm, 'n', len(w), 'mean', round(w.mean(), 3), 'values', np.round(w.values, 2))
    ax.axhline(0, color='#555555', lw=0.5)
    lo_, hi_ = min(vals), max(vals); pad_ = 0.1 * (hi_ - lo_)
    lo_, hi_ = lo_ - pad_, hi_ + pad_                              # limits follow the data (one cool case reaches -3.7 K)
    tk = [x for x in NICE if x <= max(-lo_, hi_) * 0.6][-1]
    tks = [t for t in (-2 * tk, -tk, 0, tk, 2 * tk) if lo_ <= t <= hi_]
    ax.set_ylim(lo_, hi_); ax.set_xlim(-0.6, len(grp) - 0.4); ax.set_xticks([])
    ax.set_yticks(tks); ax.set_yticklabels([fmt(t) if t else '0' for t in tks])
    ax.tick_params(axis='y', labelsize=6.3, pad=1)
    ax.set_title(lab, fontsize=6.5, pad=2, linespacing=0.95)
    for s_ in ('top', 'right', 'bottom'):
        ax.spines[s_].set_visible(False)
from matplotlib.legend_handler import HandlerTuple                # one legend for b-f: symbols of b-e and bars of f
bar_ = lambda c, f: Rectangle((0, 0), 1, 1, fc=c if f else 'white', ec=c, lw=0.8)
mk_ = lambda sg, low: mlines.Line2D([], [], **mk(sg, low, ms=3.6))
hd = [(mk_('warm', True), bar_(RED, True)), (mk_('warm', False), bar_(RED, False)), mk_('cool', True), (mk_('cool', False), bar_(BLUE, False)),
      bar_(GREY, True), mlines.Line2D([], [], marker='D', ms=2.4, mfc='white', mec='k', mew=0.45, ls='none')]
fig.legend(hd, ['warm, <30°', 'warm, ≥30°', 'cool, <30°', 'cool, ≥30°', 'null edit', 'single WRF day'], loc='upper left',
           bbox_to_anchor=box(FX0 - 1.0, YB3 - 2.0, 10, 5)[:2], frameon=False, fontsize=6.5, ncol=6, handlelength=1.6, handleheight=0.8,
           columnspacing=0.8, handletextpad=0.3, borderaxespad=0, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})

# ---------------------------------------------------------------- g: storm-relative composite, warm regions <30 deg
Z = np.load(f'{MECH}/stormrel_oper.npz'); xg, yg = Z['x'], Z['y']; VO = list(Z['vars'])
tb = np.arange(-1.0, 1.01, 0.2); tcm, tnm = binned('RdBu_r', tb)
xs, ys = (xg >= -1500) & (xg <= 2000), (yg >= -1500) & (yg <= 1500)
sub = np.ix_(ys, xs)
YY, XX = np.meshgrid(yg, xg, indexing='ij')
RMASK, CLIP, SEAC = 750.0, 1.5, '#1F78B4'


def smooth(a, s):
    m = np.isfinite(a); a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, s); den = gaussian_filter(m.astype(float), s)
    return np.where(m, num / np.maximum(den, 1e-6), np.nan)


NR = pd.read_csv(f'{MECH}/coast_normals.csv').set_index('region')


def sea_dir(d):
    rows = []
    for _, c in d.iterrows():
        lo0, la0 = NR.loc[c.region, 'lon'], NR.loc[c.region, 'lat']
        r = np.array([((c.lon - lo0 + 180) % 360 - 180) * np.cos(np.radians(la0)), c.lat - la0]); r /= np.hypot(*r)
        n = np.array([NR.loc[c.region, 'nx'], NR.loc[c.region, 'ny']])
        ny = n @ np.array([-r[1], r[0]])
        rows.append((c.region, n @ r, -ny if la0 < 0 else ny))
    q = pd.DataFrame(rows, columns=['region', 'x', 'y']).groupby('region').mean().mean()
    return q.x, q.y


GW = R3H * 3500.0 / 3000.0; GX = W - 10.5 - GW
key = 'warm_low'
ax = fig.add_axes(box(GX, YB3, GW, R3H))
F = Z[key]; agree = Z[key + '_agree']; nreg = Z[key + '_nreg']; nc = Z[key + '_n']; d0 = float(Z[key + '_dist'][0])
ok = nreg >= max(3, int(np.ceil(0.5 * nc[2])))
t2 = np.where(ok, F[VO.index('t2m')], np.nan)
ax.pcolormesh(xg[xs], yg[ys], t2[sub], cmap=tcm, norm=tnm, shading='nearest', rasterized=True, zorder=1)
st = (agree[VO.index('t2m')] >= 0.8) & ok
m_ = st[sub] & ((np.arange(ys.sum())[:, None] % 3 == 0) & (np.arange(xs.sum())[None, :] % 3 == 0))
ax.plot(XX[sub][m_], YY[sub][m_], 'k.', ms=0.8, zorder=3)
w5 = smooth(np.where(ok, F[VO.index('w500')], np.nan), 2.0)[sub]
ax.contour(xg[xs], yg[ys], w5, levels=[0.01], colors='#6A3D9A', linewidths=0.9, zorder=4)
GCW = GC[(GC.kind == 'warm') & (GC.band == 'low')]
sx, sy = sea_dir(GCW); sl = np.hypot(sx, sy)
ax.annotate('', xy=(650 * sx / sl, 650 * sy / sl), xytext=(0, 0), zorder=7.5,
            arrowprops=dict(arrowstyle='-|>,head_length=0.5,head_width=0.25', lw=1.6, color=SEAC, shrinkA=3, shrinkB=0))
ax.text(870 * sx / sl + 120, 870 * sy / sl, 'sea', fontsize=6.5, color=SEAC, ha='center', va='center', zorder=8, path_effects=HALO)
k = 6
u, vv = np.where(ok, F[VO.index('u10')], np.nan), np.where(ok, F[VO.index('v10')], np.nan)
far = np.hypot(XX - d0, YY) > RMASK
sp = np.hypot(u, vv); cl = np.minimum(1.0, CLIP / np.maximum(sp, 1e-9))
u, vv = np.where(far, u * cl, np.nan), np.where(far, vv * cl, np.nan)
ii, jj = np.where(ys)[0][::k], np.where(xs)[0][::k]
qv = ax.quiver(xg[jj], yg[ii], u[np.ix_(ii, jj)], vv[np.ix_(ii, jj)], scale=9.0, scale_units='inches', width=0.0065,
               headwidth=3.4, headlength=3.8, headaxislength=3.4, color='#222222', zorder=6)
ax.plot(d0, 0, marker='$◎$', ms=8, color='k', zorder=7)
ax.plot(0, 0, marker='o', ms=4.5, mfc='none', mec='k', mew=0.9, zorder=7)
ax.set_xlim(-1500, 2000); ax.set_ylim(-1500, 1500); ax.set_aspect('equal')
ax.set_xticks([-1000, 0, 1000, 2000]); ax.set_yticks([-1000, 0, 1000])
ax.set_xticklabels(['−1,000', '0', '1,000', '2,000']); ax.set_yticklabels(['−1,000', '0', '1,000'])
ax.tick_params(labelsize=6.3)
ax.set_title(f'GraphCast, warm <30° ({nc[2]} regions)', loc='left', fontsize=6.5, pad=2)
ax.set_ylabel('Across (km)', labelpad=0.5, fontsize=6.5)
ax.set_xlabel('Towards the TC (km)', labelpad=1, fontsize=6.5)
print('panel g n', nc, 'storm distance', round(d0), 'seaward', round(sx, 2), round(sy, 2))
ax.quiverkey(qv, (GX + GW / 2 - 0.6) / W, (YB3 - 9.0) / H, 1.0, '1 m/s', labelpos='E', coordinates='figure',
             fontproperties={'size': 6.5}, labelsep=0.05)
cax = fig.add_axes(box(GX + GW + 1.2, YB3 + 3, 1.8, R3H - 6))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=tnm, cmap=tcm), cax=cax, orientation='vertical', extend='both', ticks=[-1, 0, 1])
cb.ax.tick_params(labelsize=6.3, length=2, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_yticklabels(['−1', '0', '1'])
cb.set_label('2-m T (K)', fontsize=6.3, labelpad=0.5)

# panel letters
fig.text(1.0 / W, (H - 1.0) / H, 'a', fontsize=9, fontweight='bold', va='top', ha='left')
y2 = (YB2 + R2H + 5.0) / H
for s_, x0 in zip('bcde', PX):
    fig.text((x0 - 10.5) / W, y2, s_, fontsize=9, fontweight='bold', va='top', ha='left')
y3 = (YB3 + R3H + 7.0) / H
fig.text((FX0 - 10.5) / W, y3, 'f', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text((GX - 9.5) / W, y3, 'g', fontsize=9, fontweight='bold', va='top', ha='left')
print('size mm', W, round(H, 1))
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/Fig3.png', dpi=600); fig.savefig(f'{OUT}/Fig3.pdf')
