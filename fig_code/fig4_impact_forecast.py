"""Fig. 4 (NC v10): heat from distant storms reaches many people and is forecast days ahead.
Top: impact (impact.py + impact_ext.py v3 on DelftBlue; 2020 population, warm regions, f = 0.50).
Bottom: forecasts (fc_test.py / fc_track.py, WeatherBench2 1.5 deg HRES 2016-2022 and Pangu 2018-2022) and one example day
(fc_example.py, 0.25 deg HRES and ERA5, Mitag 2019-09-30 06 UTC).
Env: IMPD (impact CSVs), FCD (fctest_out), FIGOUT.  Run: python3 fig4_impact_forecast.py"""
import os, sys
import numpy as np
import matplotlib.lines as mlines, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.io import shapereader
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import shapely.geometry as sgeom
from shapely.prepared import prep

HERE = os.path.dirname(os.path.abspath(__file__))
IMPD = os.environ.get('IMPD', os.path.join(HERE, '..', 'data', 'impact'))
FCD = os.environ.get('FCD', os.path.join(HERE, '..', 'data', 'forecast'))
OUT = os.environ.get('FIGOUT', os.path.join(HERE, '..', 'fig'))
EXF = os.path.join(FCD, 'examples', 'fc_example_2019093006_2019268N10155.npz')
os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(IMPD, f))
have = lambda f: os.path.exists(os.path.join(IMPD, f))

CONT = [('E and SE Asia', [7, 9, 10, 12, 13], '#C44E52'), ('S Asia', [5, 6], '#E69F00'),
        ('Americas', [17, 18, 19, 20, 21, 22, 23], '#4C72B0'), ('Africa', [1, 2, 3, 4], '#009E73'),
        ('Australia, N Guinea', [8, 11, 14, 15, 16], '#8C8C8C')]
RNAME = {'W1': 'NE South Africa', 'W2': 'Inland N Mozambique', 'W3': 'Coastal N Mozambique', 'W4': 'Madagascar', 'W5': 'SE India',
         'W6': 'Bangladesh', 'W7': 'Southern China, N Vietnam', 'W8': 'W Australia', 'W9': 'E Borneo', 'W10': 'Korea, NE China',
         'W11': 'N Australia', 'W12': 'W Japan', 'W13': 'Central Japan', 'W14': 'S New Guinea', 'W15': 'Central Queensland',
         'W16': 'NE Queensland', 'W17': 'NW Mexico', 'W18': 'E Mexico', 'W19': 'SE United States', 'W20': 'Belize, Yucatan',
         'W21': 'Nicaragua, Honduras', 'W22': 'NE United States', 'W23': 'Maritime Canada'}
LIGHT = '#CFCFCF'

# ---------------- layout (mm)
W = 180.0
pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(20, 0, pc)[0], proj.transform_point(300, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37, pc)[1], proj.transform_point(160, 51, pc)[1]
MW = 110.0; MH = MW * (Y1 - Y0) / (X1 - X0)
SH = 33.0                                   # stat panel height
EH = 28.0                                   # example map height (39.8 mm wide, 30 x 21 deg)
rows = [4.5, MH, 12.0, SH, 12.0, SH, 11.0, EH, 10.5]
H = sum(rows)
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, ytop, w, h: fig.add_axes([x0 / W, 1 - (ytop + h) / H, w / W, h / H])
yA = rows[0]; yB = yA + MH + rows[2]; yC = yB + SH + rows[4]; yD = yC + SH + rows[6]
CX = [11.0, 72.0, 133.0]; CW = 45.0

def L(ax, s, dx=-10.0, dy=4.2):
    bb = ax.get_position(); fig.text(bb.x0 + dx / W, bb.y1 + dy / H, s, fontsize=9, fontweight='bold', va='top', ha='left')

def clean(ax):
    ax.spines['left'].set_linewidth(0.5); ax.spines['bottom'].set_linewidth(0.5)

# ---------------- a: map of storm-caused hot days per year (colour) and person-days per region (bubbles)
axA = fig.add_axes([1.0 / W, 1 - (yA + MH) / H, MW / W, MH / H], projection=proj)
axA.set_xlim(X0, X1); axA.set_ylim(Y0, Y1)
axA.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#E6E6E6', edgecolor='none', zorder=0)
axA.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#666666', zorder=3)
axA.spines['geo'].set_linewidth(0.4); axA.spines['geo'].set_edgecolor('#999999')
dbounds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]
dcm, dnorm = binned('OrRd', dbounds)
dcm = ListedColormap(plt.get_cmap('OrRd')(np.linspace(0.18, 0.95, len(dbounds) - 1))); dcm.set_under('#FBE9DF'); dcm.set_over('#7F0000')
dnorm = BoundaryNorm(dbounds, dcm.N)
if have('cells_warm.csv'):                                       # warm-region grid points with any storm-caused hot day
    Cu = rd('cells_warm.csv'); vcol = 'att_f0.50_per_yr'           # (impact_cells.csv, one row per grid point)
    for _, r in Cu.iterrows():
        v = r[vcol]
        axA.add_patch(Rectangle((r.lon - 0.5, r.lat - 0.5), 1, 1, transform=pc, lw=0,
                                facecolor=dcm(dnorm(v)) if v >= dbounds[0] else '#FBE9DF', zorder=2))
else:
    axA.text(0.5, 0.5, 'impact_cells.csv pending', transform=axA.transAxes, ha='center')
RG = rd('impact_ext_regions.csv')
qmax = RG.Q_M_per_yr.max()
sz = lambda q: 4 + 260 * np.sqrt(q / qmax)
lab = {'W7': (-1, 9.5), 'W5': (-8, -3), 'W10': (-6, 6), 'W13': (8, 3), 'W6': (-9, 2.5), 'W19': (-6, 5.5), 'W12': (3, -6.5),
       'W18': (-8, -2), 'W22': (-8, 3.5)}
for _, r in RG.iterrows():
    axA.scatter(r.lon, r.lat, s=sz(r.Q_M_per_yr), transform=pc, facecolor='none', edgecolor='#222222', lw=0.5, alpha=0.75, zorder=5)
    if r.region in lab:
        dx, dy = lab[r.region]
        t = f'{r.Q_M_per_yr:.0f}' if r.Q_M_per_yr >= 10 else f'{r.Q_M_per_yr:.1f}'
        axA.text(r.lon + dx, r.lat + dy, t, transform=pc, fontsize=6.5, ha='center', va='center', path_effects=HALO, zorder=6)
# bubble legend in the empty South Pacific
for k, (q, s_) in enumerate([(200, '200'), (20, '20'), (2, '2')]):
    x, y = 68 + 14 * k, -22                                     # clear of the bottom frame
    axA.scatter(x, y, s=sz(q), transform=pc, facecolor='none', edgecolor='#222222', lw=0.6, zorder=5)
    axA.text(x, y - 8.0, s_, transform=pc, fontsize=6.5, ha='center', va='top', zorder=6)
axA.text(76, -9.0, 'Million person-days\nper year (region)', transform=pc, fontsize=6.5, ha='center', va='bottom', zorder=6)
# colour bar for the days per year, in the southern Indian Ocean
def fig_xy(lo_, la_):
    xp, yp = proj.transform_point(lo_, la_, pc); d = axA.transData.transform((xp, yp)); return fig.transFigure.inverted().transform(d)
c0, c1 = fig_xy(185, -26.0), fig_xy(269, -26.0)                 # tick labels clear of the bottom frame
cax = fig.add_axes([c0[0], c0[1], c1[0] - c0[0], 1.6 / H])
cb = plt.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap=dcm), cax=cax, orientation='horizontal', extend='both', ticks=dbounds)
cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=6.5, length=1.5, width=0.3, pad=1)
cb.ax.set_xticklabels(['0.05', '0.1', '0.15', '0.2', '0.3', '0.4', '0.6'])
fig.text((c0[0] + c1[0]) / 2, c0[1] + 2.4 / H, 'Storm-attributed hot days per year', fontsize=6.5, ha='center', va='bottom')
L(axA, 'a', dx=-0.5, dy=4.2)

# ---------------- b: chance of a hot day when a distant storm is in position (observed, no model), against other days
axB = box(141.0, yA, 37.5, MH); clean(axB)
RK = rd('impact_risk.csv')
sub = RK[RK.region != 'all'].merge(RG[['region', 'pop2020_M']], on='region')
sub = sub[sub.pop2020_M >= 2].sort_values('exposure_pd_M_per_yr', ascending=False).head(10)
sub = pd.concat([RK[RK.region == 'all'], sub])
yy = np.arange(len(sub))[::-1]
for y_, (_, r) in zip(yy, sub.iterrows()):
    p1, p0 = 100 * r.p_hot_exposure, 100 * r.p_hot_other_days
    axB.plot([p0, p1], [y_, y_], color='#BBBBBB', lw=1.4, solid_capstyle='butt', zorder=1)
    axB.scatter([p0], [y_], s=14, facecolor='white', edgecolor='#555555', lw=0.7, zorder=2)
    axB.scatter([p1], [y_], s=16, color=RED, lw=0, zorder=3)
    axB.text(35.5, y_, f'{p1 / p0:.1f}', fontsize=6.5, va='center', ha='right', color=RED)
axB.set_yticks(yy); axB.set_yticklabels(['All warm coasts' if r == 'all' else RNAME.get(r, r) for r in sub.region], fontsize=6.5)
axB.set_ylim(-2.9, len(sub) - 0.4)
axB.set_xlim(0, 36); axB.set_xticks([0, 10, 20, 30])
axB.set_xlabel('Chance of a hot day (%)')
axB.text(35.5, 1.0, 'Ratio', transform=axB.get_xaxis_transform(), fontsize=6.5, ha='right', va='bottom', color=RED)
axB.legend(handles=[Line2D([], [], marker='o', ls='', mfc=RED, mec=RED, ms=3.6, label='Storm in position'),
                    Line2D([], [], marker='o', ls='', mfc='white', mec='#555555', ms=3.4, label='Other days')],
           loc='lower left', bbox_to_anchor=(0.0, 0.0), ncol=1, fontsize=6.5, handletextpad=0.1, borderaxespad=0.1)
L(axB, 'b', dx=-30.0)

# ---------------- c: year by year (2020 population), stacked by continent
axC = box(CX[0], yB, CW, SH); clean(axC)
Y = rd('impact_years.csv'); yrs = Y.year.values
if have('cont_years.csv'):                                       # impact_region_years.csv summed by continent
    CY = rd('cont_years.csv').set_index('year').reindex(yrs).fillna(0); base = np.zeros(len(yrs))
    for (name, ids, col), key in zip(CONT, ('EA', 'SA', 'AM', 'AF', 'AU')):
        v = CY[key].values / 1e3
        axC.bar(yrs, v, bottom=base, width=0.8, color=col, lw=0, label=name); base += v
    assert np.allclose(base, Y.Q_pop2020.values / 1e9, atol=2e-3), 'continent sum differs from the yearly total'
else:
    axC.bar(yrs, Y.Q_pop2020 / 1e9, width=0.8, color=RED, lw=0)
axC.plot(yrs, Y.Q_popvarying / 1e9, color='black', lw=0.8, label="Each year's population")
mus = []
for a_, b_ in ((1980, 2000), (2001, 2021)):
    m = (yrs >= a_) & (yrs <= b_); mu = Y.Q_pop2020[m].mean() / 1e9; mus.append(mu)
    axC.plot([a_ - 0.4, b_ + 0.4], [mu, mu], color='#333333', lw=0.7, ls=(0, (3, 1.5)))
    axC.text((a_ + b_) / 2, 0.565, f'Mean {mu:.2f}' if len(mus) == 1 else f'Mean {mu:.2f} (+{100 * (mu / mus[0] - 1):.0f}%)',
             fontsize=6.5, ha='center', va='bottom', color='#333333')          # above the tallest bars, below the legend
axC.set_xlim(1979, 2022); axC.set_ylim(0, 0.86); axC.set_xticks([1980, 1990, 2000, 2010, 2020]); axC.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
axC.set_ylabel('Billion person-days per year')
axC.legend(loc='upper left', ncol=2, fontsize=6.5, handlelength=0.9, columnspacing=0.6, borderaxespad=0.0)
L(axC, 'c')

# ---------------- d: season
axD = box(CX[1], yB, CW, SH); clean(axD)
SS = rd('impact_season.csv'); mo = np.arange(1, 13)
GRP = [('southern China (W7)', 'Southern China, N Vietnam', RED), ('Northern Hemisphere, other', 'Other N Hemisphere', '#E69F00'),
       ('Southern Hemisphere', 'S Hemisphere', '#4C72B0')]
base = np.zeros(12)
for g, nm, col in GRP:
    v = SS[SS.group == g].set_index('month').storm_caused.reindex(mo).fillna(0).values / 1e6
    axD.bar(mo, v, bottom=base, width=0.75, color=col, lw=0, label=nm); base += v
axD.set_xlim(0.4, 12.6); axD.set_xticks(mo); axD.set_xticklabels(list('JFMAMJJASOND'))
axD.set_ylabel('Million person-days per year')
w7 = SS[SS.group == 'southern China (W7)'].set_index('month').share_of_hot
axD.set_ylim(0, 92)
axD.text(0.02, 0.64, f'Aug: {100 * w7.loc[8]:.1f}% of hot\ndays in S China,\nN Vietnam', transform=axD.transAxes, fontsize=6.5, va='top', ha='left')
axD.legend(loc='upper left', fontsize=6.5, handlelength=0.9, borderaxespad=0.0)
L(axD, 'd')

# ---------------- e: heat, then the storm?
axE = box(CX[2], yB, CW, SH); clean(axE)
SQ = rd('impact_sequence.csv'); dd = np.arange(1, 6)
for kind, nm, col, ls in (('storm-caused hot', 'Storm-attributed hot days', RED, '-'), ('hot, not storm-caused', 'Other hot exposure days', '#555555', (0, (3, 1.5))),
                          ('all exposure', 'All exposure days', '#AAAAAA', '-')):
    r = SQ[(SQ.kind == kind) & (SQ.region == 'all')]
    if len(r):
        r = r.iloc[0]; v = [100 * r[f'same_by_{k}d'] for k in dd]
        axE.plot(np.r_[0, dd], np.r_[0, v], color=col, ls=ls, lw=1.0, marker='o', ms=2.2, label=nm)
axE.set_xlim(0, 5.3); axE.set_xticks(range(0, 6)); axE.set_ylim(0, 25)
axE.set_xlabel('Days after the hot day'); axE.set_ylabel('Storm within 300 km (%)')
axE.text(0.97, 0.08, '9 in 10: no passage within\n300 km in the next 5 days', transform=axE.transAxes, fontsize=6.5, ha='right', va='center')
axE.legend(loc='upper left', fontsize=6.5, handlelength=1.4, borderaxespad=0.0)
L(axE, 'e')

# ---------------- f: share of the observed heat that forecasts capture
axF = box(CX[0], yC, CW, SH); clean(axF)
lead = np.arange(0, 10)
for model, ls in (('hres', '-'), ('pangu', (0, (3, 1.5)))):
    P = pd.read_csv(os.path.join(FCD, f'fc_{model}_pooled.csv'))
    for sign, col in (('warm', RED), ('cool', BLUE)):
        s = P[P.sign == sign].set_index('series')
        v = np.array([s.loc[f'f{k}', 'frac_of_obs'] for k in lead]) * 100
        if model == 'hres':
            lo = np.array([s.loc[f'f{k}', 'frac_lo'] for k in lead]) * 100; hi = np.array([s.loc[f'f{k}', 'frac_hi'] for k in lead]) * 100
            axF.fill_between(lead, lo, hi, color=col, alpha=0.15, lw=0)
        axF.plot(lead, v, color=col, ls=ls, lw=1.0)
axF.axhline(100, color='#999999', lw=0.5, ls=(0, (2, 2)))
axF.set_xlim(0, 9); axF.set_ylim(0, 130); axF.set_xticks(range(0, 10))
axF.set_xlabel('Forecast lead (days)'); axF.set_ylabel('Heat (cooling) captured (%)')
axF.legend(handles=[Line2D([], [], color=RED, lw=1, label='Warm coasts'), Line2D([], [], color=BLUE, lw=1, label='Cool coasts'),
                    Line2D([], [], color='#444444', lw=1, label='HRES'), Line2D([], [], color='#444444', lw=1, ls=(0, (3, 1.5)), label='Pangu')],
           loc='lower left', fontsize=6.5, ncol=2, handlelength=1.4, columnspacing=0.8, borderaxespad=0.1)
L(axF, 'f')

# ---------------- g: captured heat by forecast storm position error (HRES, warm coasts)
axG = box(CX[1], yC, CW, SH); clean(axG)
T = pd.read_csv(os.path.join(FCD, 'fc_hres_by_track.csv')); T = T[(T.sign == 'warm') & (T.lead >= 2)]
for b_, nm, col in (('< 250 km', '< 250 km', '#7F0000'), ('250-500 km', '250–500 km', '#E3826B'),
                    ('>= 500 km or no storm', 'Farther, missed', '#9E9E9E')):
    s = T[T.bin == b_].sort_values('lead'); axG.plot(s.lead, 100 * s.frac, color=col, lw=1.0, marker='o', ms=2.2, label=nm)
axG.axhline(100, color='#999999', lw=0.5, ls=(0, (2, 2)))
axG.set_xlim(1.7, 9.3); axG.set_xticks(range(2, 10)); axG.set_ylim(-100, 120); axG.set_yticks([-50, 0, 50, 100])
axG.set_xlabel('Forecast lead (days)'); axG.set_ylabel('Heat captured (%)')
axG.legend(loc='lower left', fontsize=6.5, handlelength=1.2, borderaxespad=0.1, title='Storm position error', title_fontsize=6.5)
L(axG, 'g')

# ---------------- h: hit rate for hot days
axH = box(CX[2], yC, CW, SH); clean(axH)
EH_ = pd.read_csv(os.path.join(FCD, 'fc_hres_error_hits.csv'))
for kind, nm, col in (('exp', 'Hot days with a storm in position', RED), ('ref', 'Other hot days', '#8C8C8C')):
    r = EH_[(EH_.kind == kind) & (EH_.sign == 'warm')].iloc[0]
    axH.plot(lead, [100 * r[f'hit{k}'] for k in lead], color=col, lw=1.0, marker='o', ms=2.2, label=nm)
axH.set_xlim(-0.3, 9.3); axH.set_xticks(range(0, 10)); axH.set_ylim(20, 85)
axH.set_xlabel('Forecast lead (days)'); axH.set_ylabel('Hot days forecast as hot (%)')
axH.legend(loc='lower left', fontsize=6.5, handlelength=1.2, borderaxespad=0.1)
L(axH, 'h')

# ---------------- i: one day: Mitag, 30 September 2019, 14 LT in southern China
Z = np.load(EXF); lon, lat = Z['lon'], Z['lat']
LO, LA = np.meshgrid(lon, lat, indexing='ij')
land = prep(sgeom.MultiPolygon([g for g in shapereader.Reader(shapereader.natural_earth('50m', 'physical', 'land')).geometries()
                                for g in (g.geoms if hasattr(g, 'geoms') else [g])]))
BX = (106, 120, 20, 27)
inb = (LO >= BX[0]) & (LO <= BX[1]) & (LA >= BX[2]) & (LA <= BX[3])
lm = np.zeros_like(inb)
for i, j in zip(*np.where(inb)):
    lm[i, j] = land.contains(sgeom.Point(float(LO[i, j]), float(LA[i, j])))
abounds = np.arange(-4, 4.01, 1.0)
acm = ListedColormap(plt.get_cmap('RdBu_r')(np.linspace(0.08, 0.92, len(abounds) - 1))); acm.set_under('#053061'); acm.set_over('#67001F')
anorm = BoundaryNorm(abounds, acm.N)
leads = list(Z['leads']); show = [('Observed (ERA5)', None)] + [(f'{d}-day forecast', leads.index(d)) for d in (3, 5, 7)]
gap = 2.6; ex0 = CX[0]; EW = (CX[2] + CW - ex0 - 3 * gap) / 4      # maps flush with the frames of f-h
dlo = np.r_[lon - 0.125, lon[-1] + 0.125]; dla = np.r_[lat - 0.125, lat[-1] + 0.125]
vals = []
for k, (nm, li) in enumerate(show):
    ax = fig.add_axes([(ex0 + k * (EW + gap)) / W, 1 - (yD + EH) / H, EW / W, EH / H], projection=pc)
    ax.set_extent([102, 132, 12, 33], crs=pc)
    A_ = Z['obs_anom'] if li is None else Z['fc_anom'][li]; P_ = Z['obs_mslp'] if li is None else Z['fc_mslp'][li]
    ax.pcolormesh(dlo, dla, A_.T, cmap=acm, norm=anorm, transform=pc, rasterized=True, zorder=1)
    ax.contour(lon, lat, P_.T, levels=np.arange(980, 1009, 4), colors='#222222', linewidths=0.35, transform=pc, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.35, edgecolor='#333333', zorder=3)
    ax.add_patch(Rectangle((BX[0], BX[2]), BX[1] - BX[0], BX[3] - BX[2], transform=pc, fill=False, lw=0.6, ls=(0, (2, 1)), edgecolor='black', zorder=4))
    ax.plot(Z['track_lon'], Z['track_lat'], color='black', lw=0.7, transform=pc, zorder=4)
    tt = pd.to_datetime(Z['track_time']); now = tt == pd.Timestamp(str(Z['valid']))
    ax.plot(Z['track_lon'][now], Z['track_lat'][now], marker='o', ms=3.2, color='black', transform=pc, zorder=5, ls='')
    if li is not None:
        fx, fy, fe = Z['fc_pos'][li]
        ax.plot(fx, fy, marker='X', ms=5.5, mew=0.6, mfc='#FFE14D', mec='#1F4E9E', transform=pc, zorder=6, ls='')
    v = float(A_[lm].mean()); vals.append(v)
    ax.text(0.03, 0.97, nm, transform=ax.transAxes, fontsize=7.0, va='top', ha='left', path_effects=HALO, zorder=7)
    ax.text(0.03, 0.03, f'{v:+.1f} K', transform=ax.transAxes, fontsize=7.0, va='bottom', ha='left', path_effects=HALO, zorder=7)
    ax.spines['geo'].set_linewidth(0.4)
    if k == 0:
        L(ax, 'i', dx=-10.0, dy=4.2)
yl = yD + EH + 5.0                                               # legend row under i
tl = fig.text(ex0 / W, 1 - yl / H, '2-m temperature anomaly (K)', fontsize=7.5, ha='left', va='center')
fig.canvas.draw(); wl = tl.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi / MM
cx0 = ex0 + wl + 2.5
cax = fig.add_axes([cx0 / W, 1 - (yl + 0.9) / H, 46.0 / W, 1.8 / H])
cb = plt.colorbar(plt.cm.ScalarMappable(norm=anorm, cmap=acm), cax=cax, orientation='horizontal', extend='both', ticks=abounds)
cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=7.0, length=1.5, width=0.3, pad=1)
cb.ax.set_xticklabels([f'{t:g}'.replace('-', '−') for t in abounds])
hs = [mlines.Line2D([], [], marker='o', ms=4.0, color='black', ls='none'),
      mlines.Line2D([], [], marker='X', ms=6.0, mew=0.6, mfc='#FFE14D', mec='#1F4E9E', ls='none'),
      mlines.Line2D([], [], color='black', lw=0.7)]
fig.legend(hs, ['observed storm', 'forecast storm', 'observed track'], loc='center right', bbox_to_anchor=((CX[2] + CW) / W, 1 - yl / H),
           ncol=3, fontsize=7.5, frameon=False, handlelength=1.2, handletextpad=0.35, columnspacing=1.0, borderaxespad=0, borderpad=0)

print('example land-box means (obs, 3, 5, 7 d):', [round(v, 2) for v in vals])
bad = overlap_check(fig)
print('QA:', 'clean' if not bad else bad)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT, f'Fig4.{ext}'), dpi=600 if ext == 'pdf' else 300)
print('saved', os.path.join(OUT, 'Fig4_impact_forecast.png'), f'{W:.0f} x {H:.0f} mm')
