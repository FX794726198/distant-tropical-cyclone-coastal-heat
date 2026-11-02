"""v10 SI figure: v09 Fig. 6 d, f, g (relabelled a-c); a-c and e are in v10 Fig. 3."""
"""Fig. 6 (draft, v08): one rule behind the warm and cool coasts, and where it stops.
a: map of the 41 regions (colour = change in the map, Fig. 1a,b) with the idealized outer-circulation wind of a storm in the
   region's direction (brown: blows off the land; teal: onto the land);
b: that wind's onshore component against the change in the map (41 regions);
c: the same component, from the storm positions of the GraphCast cases, against the GraphCast effect (usable regions);
d: GraphCast changes caused by the storm for three groups of regions (2-m and 850-hPa temperature, 500-hPa sinking,
   onshore wind, rain), with 95% intervals from resampling storms within regions;
e-g: storm-relative composites of the GraphCast changes (storm on the +x axis, Southern Hemisphere mirrored) for warm regions
   equatorward of 30 deg, warm regions at 30 deg and beyond, and cool regions.
Days on which the removed storm lay two or more sectors from the region's direction (four in C8, one in W8;
geometry_flagged_cases.csv) are left out of c-g.
Inputs (MECH, default data/mech): geometry_map_regions.csv, geometry_cases_oper.csv, geometry_flagged_cases.csv,
mech_cases_oper.csv, stormrel_oper.npz (from nc_upgrade/scripts/mech: mech_gc.py, mech_win.py, geometry.py, mech_explore.py).
Usage: MECH=... FIGOUT=... python3 figS13_outer_winds.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
from ggc_common import CEN, NUM
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch

MECH = os.environ.get('MECH', 'data/mech')
OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
OFF, ON = '#8C510A', '#01665E'                      # outer circulation blows off the land / onto the land
G = pd.read_csv(f'{MECH}/geometry_map_regions.csv').set_index('region')
flag = set(pd.read_csv(f'{MECH}/geometry_flagged_cases.csv').case)
GC = pd.read_csv(f'{MECH}/geometry_cases_oper.csv'); GC = GC[~GC.case.isin(flag)]
GC['lat_r'] = GC.region.map(G.lat); GC['band'] = np.where(GC.lat_r.abs() < 30, 'low', 'mid')
rng = np.random.default_rng(7)
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def wind_dir(sector, lat):
    """unit vector (east, north) of the idealized outer-circulation wind at the region (tangential + 0.4 inflow)"""
    b = np.radians(SECT.index(sector) * 45.0); r = np.array([np.sin(b), np.cos(b)])
    t = np.array([r[1], -r[0]]) if lat >= 0 else np.array([-r[1], r[0]])
    v = t + 0.4 * r
    return v / np.hypot(*v)


def region_means(d, cols):
    """storm-equal means within regions"""
    return d.groupby(['region', 'sid'])[cols].mean().groupby('region').mean()


def pooled_ci(d, col, n=2000):
    """region-equal mean and 95% interval from resampling storms within regions"""
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
from scipy.ndimage import gaussian_filter
W = 180.0
pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
MW = 174.0; MH = MW * (Y1 - Y0) / (X1 - X0)
EX, EW, EG = 92.0, 36.0, 4.0                        # composites to the right of the bars, one row
R3H = EW * 3000.0 / 3500.0
R2H = R3H
YB3 = 19.0; YB2 = YB3
H = YB3 + R3H + 12.0
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()
fmt = lambda v: f'{v:g}'.replace('-', '−')

# ---------------------------------------------------------------- d: fingerprints of three groups
GC['w500c'] = GC.w500 * 100.0
grp = [('warm, <30°', (GC.kind == 'warm') & (GC.band == 'low'), RED, True), ('warm, ≥30°', (GC.kind == 'warm') & (GC.band == 'mid'), RED, False),
       ('cool', GC.kind == 'cool', BLUE, True)]
VARS = [('t2m', '2-m T\n(K)'), ('t850', '850-hPa\nT (K)'), ('w500c', 'Sinking\n(10$^{-2}$\nPa s$^{-1}$)'), ('onshore', 'Onshore\nwind\n(m s$^{-1}$)'),
        ('tp6', 'Rain\n(mm per\n6 h)')]
DX0, DWi, DG = 10.0, 8.4, 6.6
for j, (v, lab) in enumerate(VARS):
    ax = fig.add_axes(box(DX0 + j * (DWi + DG), YB2, DWi, R2H))
    vals = []
    for i, (nm, sel, col, fill) in enumerate(grp):
        m, l_, h_ = pooled_ci(GC[sel], v); vals += [l_, h_]
        ax.bar(i, m, 0.72, color=col if fill else 'white', edgecolor=col, lw=0.8, zorder=2)
        ax.errorbar(i, m, yerr=[[m - l_], [h_ - m]], fmt='none', ecolor='k', elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=3)
        print('panel d', v, nm, round(m, 3), round(l_, 3), round(h_, 3), 'regions', GC[sel].region.nunique())
    ax.axhline(0, color='#555555', lw=0.5)
    lim = max(abs(min(vals)), abs(max(vals))) * 1.1
    tk = {'t2m': 0.5, 't850': 0.5, 'w500c': 2, 'onshore': 0.5, 'tp6': 0.5}[v]
    ax.set_ylim(-lim, lim); ax.set_xlim(-0.6, 2.6); ax.set_xticks([])
    ax.set_yticks([-tk, 0, tk]); ax.set_yticklabels([fmt(-tk), '0', fmt(tk)])
    ax.tick_params(axis='y', labelsize=6.3, pad=1)
    ax.set_title(lab, fontsize=6.5, pad=2, linespacing=0.95)
    for s_ in ('top', 'right', 'bottom'):
        ax.spines[s_].set_visible(False)
hd = [matplotlib.patches.Rectangle((0, 0), 1, 1, fc=c if f else 'white', ec=c, lw=0.8) for _, _, c, f in grp]
fig.legend(hd, [g[0] for g in grp], loc='upper left', bbox_to_anchor=box(DX0 - 4.0, YB2 - 1.5, 70, 5)[:2], frameon=False, fontsize=6.4,
           ncol=3, handlelength=1.0, handleheight=0.8, columnspacing=1.0, handletextpad=0.4, borderaxespad=0)

# ---------------------------------------------------------------- e-g: storm-relative composites
Z = np.load(f'{MECH}/stormrel_oper.npz'); xg, yg = Z['x'], Z['y']; VO = list(Z['vars'])
tb = np.arange(-1.0, 1.01, 0.2); tcm, tnm = binned('RdBu_r', tb)
xs, ys = (xg >= -1500) & (xg <= 2000), (yg >= -1500) & (yg <= 1500)
sub = np.ix_(ys, xs)
YY, XX = np.meshgrid(yg, xg, indexing='ij')
RMASK = 750.0                                       # no arrows within this distance of the storm centre
CLIP = 1.5                                          # longer arrows are drawn at this length (m/s)
SEAC = '#1F78B4'


def smooth(a, s):
    m = np.isfinite(a); a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, s); den = gaussian_filter(m.astype(float), s)
    return np.where(m, num / np.maximum(den, 1e-6), np.nan)


NR = pd.read_csv(f'{MECH}/coast_normals.csv').set_index('region')


def sea_dir(d):
    """region-equal mean of the seaward coast normal in the storm-relative frame (x towards the storm, SH mirrored)"""
    rows = []
    for _, c in d.iterrows():
        lo0, la0 = NR.loc[c.region, 'lon'], NR.loc[c.region, 'lat']
        r = np.array([((c.lon - lo0 + 180) % 360 - 180) * np.cos(np.radians(la0)), c.lat - la0]); r /= np.hypot(*r)
        n = np.array([NR.loc[c.region, 'nx'], NR.loc[c.region, 'ny']])
        ny = n @ np.array([-r[1], r[0]])
        rows.append((c.region, n @ r, -ny if la0 < 0 else ny))
    q = pd.DataFrame(rows, columns=['region', 'x', 'y']).groupby('region').mean().mean()
    return q.x, q.y


SEL = {'warm_low': (GC.kind == 'warm') & (GC.band == 'low'), 'warm_mid': (GC.kind == 'warm') & (GC.band == 'mid'), 'cool': GC.kind == 'cool'}
panels = [('warm_mid', 'Warm, ≥30°'), ('cool', 'Cool')]
for j, (key, ttl) in enumerate(panels):
    ax = fig.add_axes(box(EX + j * (EW + EG), YB3, EW, R3H))
    F = Z[key]; agree = Z[key + '_agree']; nreg = Z[key + '_nreg']; nc = Z[key + '_n']; d0 = float(Z[key + '_dist'][0])
    ok = nreg >= max(3, int(np.ceil(0.5 * nc[2])))
    t2 = np.where(ok, F[VO.index('t2m')], np.nan)
    ax.pcolormesh(xg[xs], yg[ys], t2[sub], cmap=tcm, norm=tnm, shading='nearest', rasterized=True, zorder=1)
    st = (agree[VO.index('t2m')] >= 0.8) & ok
    m_ = st[sub] & ((np.arange(ys.sum())[:, None] % 3 == 0) & (np.arange(xs.sum())[None, :] % 3 == 0))
    ax.plot(XX[sub][m_], YY[sub][m_], 'k.', ms=0.8, zorder=3)
    w5 = smooth(np.where(ok, F[VO.index('w500')], np.nan), 2.0)[sub]
    ax.contour(xg[xs], yg[ys], w5, levels=[0.01], colors='#6A3D9A', linewidths=0.9, zorder=4)
    sx, sy = sea_dir(GC[SEL[key]]); sl = np.hypot(sx, sy)
    ax.annotate('', xy=(650 * sx / sl, 650 * sy / sl), xytext=(0, 0), zorder=7.5,
                arrowprops=dict(arrowstyle='-|>,head_length=0.5,head_width=0.25', lw=1.6, color=SEAC, shrinkA=3, shrinkB=0))
    ax.text(820 * sx / sl, 820 * sy / sl, 'sea', fontsize=6.5, color=SEAC, ha='center', va='center', zorder=8, path_effects=HALO)
    print('panel', key, 'seaward direction (x, y)', round(sx, 2), round(sy, 2), 'length', round(sl, 2))
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
    ax.set_xticklabels(['−1,000', '0', '1,000', '2,000']); ax.set_yticklabels(['−1,000', '0', '1,000'] if j == 0 else [])
    ax.tick_params(labelsize=6.5)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(True)
    ax.set_title(f'{ttl} ({nc[2]} regions, {nc[0]} days)', loc='left', fontsize=6.8, pad=2.5)
    print('panel', key, 'n', nc, 'storm distance', round(d0))
    if j == 0:
        ax.set_ylabel('Across the storm direction (km)', labelpad=1)
    ax.set_xlabel('Towards the storm (km)', labelpad=1)
ax.quiverkey(qv, (EX + 2 * EW + EG - 2.0) / W, (YB3 - 14.6) / H, 1.0, '1 m s$^{-1}$', labelpos='W', coordinates='figure',
             fontproperties={'size': 6.3}, labelsep=0.04)
fig.text(DX0 / W, (YB3 - 11.2) / H, 'Arrows: change in 10-m wind (none within 750 km of the storm, capped at 1.5 m s$^{-1}$); purple line: '
         'faster sinking at 500 hPa (0.01 Pa s$^{-1}$)', fontsize=6.3, color='#333333', va='center', ha='left')
fig.text(DX0 / W, (YB3 - 14.6) / H, 'Blue arrow: mean direction of the sea from the region; dots: at least 80% of regions agree in sign',
         fontsize=6.3, color='#333333', va='center', ha='left')
cax = fig.add_axes(box(EX + 2 * EW + EG + 1.2, YB3 + 4, 2.0, R3H - 8))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=tnm, cmap=tcm), cax=cax, orientation='vertical', extend='both', ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.tick_params(labelsize=6.3, length=2, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_yticklabels(['−1', '−0.5', '0', '0.5', '1'])
cb.set_label('Effect on 2-m T (K)', fontsize=6.5, labelpad=1)

# panel letters
fig.text(1.0 / W, (YB2 + R2H + 10.5) / H, 'a', fontsize=9, fontweight='bold', va='top', ha='left')
for j, s_ in enumerate('bc'):
    fig.text((EX + j * (EW + EG) - (10.5 if j == 0 else 3.8)) / W, (YB3 + R3H + 10.5) / H, s_, fontsize=9, fontweight='bold', va='top', ha='left')
print('size mm', W, round(H, 1))
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/FigS13.png', dpi=600); fig.savefig(f'{OUT}/FigS13.pdf')
