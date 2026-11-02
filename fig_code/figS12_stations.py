"""v10 SI figure: v09 Fig. S8 a, f, g (relabelled a-c); b-d are v10 Fig. 3d,e and e is v10 Fig. 1e."""
"""Fig. S8: hourly station records (NOAA ISD) and the outer-wind rule.
a: coastal stations (within 50 km of the coast, |lat| <= 45) coloured by how much warmer the daily maximum is when the
   idealized outer wind of the nearest storm blows off the land (minus the least-squares slope of Tmax on the implied
   onshore component, K per unit);
b-d: daily maximum temperature, onshore wind at 12-16 LT and dew point at 12-16 LT on exposure days, binned by the implied
   onshore component (2.5-degree boxes count equally; 90% intervals from resampling years);
e: stations against the ERA5 map in the regions of Fig. 1 (storm in the region's direction, the region's season);
f: change in the onshore wind measured at the stations of each region against the implied outer wind (Fig. 6b);
g: slope of Tmax on the implied component within storm-direction sectors, with the storm turning the right way and the
   wrong way round.
Inputs (ISD, default data/isd): isd_station_slopes.csv, isd_coast_bins.csv, isd_regions.csv, isd_coast_slopes.csv
(nc_upgrade/scripts/isd: isd_list.py, isd_select.py, isd_fetch.py, isd_analyze.py).
Usage: ISD=... FIGOUT=... python3 figS12_stations.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
from ggc_common import NUM
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.lines as mlines

ISD = os.environ.get('ISD', 'data/isd')
OUT = os.environ.get('FIGOUT', 'fig'); os.makedirs(OUT, exist_ok=True)
OFF, ON = '#8C510A', '#01665E'
SL = pd.read_csv(f'{ISD}/isd_station_slopes.csv')
BN = pd.read_csv(f'{ISD}/isd_coast_bins.csv')
RT = pd.read_csv(f'{ISD}/isd_regions.csv'); RT = RT[(RT.n_storm >= 20)].copy()
FE = pd.read_csv(f'{ISD}/isd_coast_slopes.csv')
fmt = lambda v: f'{v:g}'.replace('-', '−')

W = 180.0
pc = ccrs.PlateCarree(); proj = ccrs.Robinson(central_longitude=160)
X0, X1 = proj.transform_point(22, 0, pc)[0], proj.transform_point(302, 0, pc)[0]
Y0, Y1 = proj.transform_point(160, -37.5, pc)[1], proj.transform_point(160, 46.5, pc)[1]
MW = 174.0; MH = MW * (Y1 - Y0) / (X1 - X0)
R2H, R3H = 36.0, 38.0
YB3 = 10.0
H = YB3 + R3H + 9.0 + MH + 2.0
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
inv = fig.transFigure.inverted()

# ---------------------------------------------------------------- a: map of station slopes
bounds = np.arange(-1.5, 1.51, 0.5); cmap, norm = binned('RdBu_r', bounds)
axa = fig.add_axes(box(3.5, H - 2.0 - MH, MW, MH), projection=proj)
axa.set_xlim(X0, X1); axa.set_ylim(Y0, Y1)
axa.add_feature(cfeature.LAND.with_scale('110m'), facecolor=LAND, edgecolor='none', zorder=0)
axa.add_feature(cfeature.COASTLINE.with_scale('110m'), lw=0.3, edgecolor='#555555', zorder=1)
axa.gridlines(draw_labels=False, lw=0.25, color='#BDBDBD', ls=(0, (2, 2)), xlocs=range(-180, 181, 30), ylocs=[-30, 0, 30], zorder=0.5)
for la_ in (-30, 0, 30):
    x, y = proj.transform_point(-168, la_, pc)
    axa.text(x, y, {-30: '30° S', 0: '0°', 30: '30° N'}[la_], fontsize=6.5, color='#666666', ha='left', va='bottom', zorder=6)
for s in axa.spines.values():
    s.set_linewidth(0.5)
m = SL.dropna(subset=['Tmax_slope']).copy(); m['v'] = -m.Tmax_slope; m = m.reindex(m.v.abs().sort_values().index)
axa.scatter(m.lon, m.lat, c=m.v, cmap=cmap, norm=norm, s=4.0, lw=0.15, edgecolors='#333333', transform=pc, zorder=5)
lo_ = m[m.lat.abs() < 30]; mi_ = m[m.lat.abs() >= 30]
tx, ty = proj.transform_point(-148, -4, pc)
axa.text(tx, ty, f'Warmer when the outer wind\nblows off the land:\n{int((lo_.v > 0).sum())} of {len(lo_)} stations within 30°\n'
         f'{int((mi_.v > 0).sum())} of {len(mi_)} stations beyond 30°', fontsize=7, ha='left', va='top', zorder=8, linespacing=1.15)
at = lambda lo, la: inv.transform(axa.transData.transform(proj.transform_point(lo, la, pc)))
(cx0, cy0), (cx1, cy1) = at(-147, -27), at(-95, -23.5)
cax = fig.add_axes([cx0, cy0, cx1 - cx0, cy1 - cy0])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.ax.tick_params(labelsize=6.5, length=2, pad=1); cb.outline.set_linewidth(0.4)
cb.ax.set_xticklabels([fmt(v) for v in bounds[::2]])
cb.set_label('Daily maximum: off land minus onto land (K per unit)', fontsize=6.5, labelpad=1.5)


def xaxis_onshore(ax, label=True):
    ax.set_xlim(-1.12, 1.12); ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['−1\noff land', '0', '1\nonto land'], linespacing=0.95)
    tl = ax.get_xticklabels(); tl[0].set_color(OFF); tl[2].set_color(ON)
    if label:
        ax.set_xlabel('Outer wind, onshore part', labelpad=1, fontsize=7.5)


# ---------------------------------------------------------------- f: measured onshore change against the implied outer wind
FX, FWd = 24.0, 55.0
axf = fig.add_axes(box(FX, YB3, FWd, R3H))
axf.axhline(0, color='#555555', lw=0.5, zorder=1); axf.axvline(0, color='#555555', lw=0.5, zorder=1)
for _, g in RT.iterrows():
    col = RED if g.sign == 'warm' else BLUE; low = abs(g.lat) < 30
    axf.plot(g.imp04, g.on_pm, marker='o' if g.sign == 'warm' else 's', ms=3.4, mfc=col if low else 'white', mec=col,
             mew=0.6 if low else 0.9, ls='none', zorder=3)
for r, dx, dy in (('W13', -0.15, 0.0), ('W22', -0.13, 0.22), ('W23', 0.12, 0.25)):
    g = RT.set_index('region').loc[r]
    axf.text(g.imp04 + dx, g.on_pm + dy, NUM(r), fontsize=6.0, color=RED, ha='center', va='center', zorder=5)
fl = RT[RT.lat.abs() < 30]
axf.text(0.03, 0.97, f'<30°: $r$ = {fl.imp04.corr(fl.on_pm):.2f}', transform=axf.transAxes, ha='left', va='top', fontsize=6.3)
axf.text(0.03, 0.88, f'all: $r$ = {RT.imp04.corr(RT.on_pm):.2f}', transform=axf.transAxes, ha='left', va='top', fontsize=6.3, color='#555555')
xaxis_onshore(axf); axf.set_ylim(-2.6, 1.6); axf.set_yticks([-2, -1, 0, 1]); axf.set_yticklabels([fmt(v) for v in (-2, -1, 0, 1)])
axf.set_ylabel('Measured onshore change (m s$^{-1}$)', labelpad=1)
he = [mlines.Line2D([], [], marker='o', ms=3.4, mfc=RED, mec=RED, ls='none'), mlines.Line2D([], [], marker='o', ms=3.4, mfc='white', mec=RED, mew=0.9, ls='none'),
      mlines.Line2D([], [], marker='s', ms=3.4, mfc=BLUE, mec=BLUE, ls='none'), mlines.Line2D([], [], marker='s', ms=3.4, mfc='white', mec=BLUE, mew=0.9, ls='none')]
lgf = axf.legend(he, ['warm, <30°', 'warm, ≥30°', 'cool, <30°', 'cool, ≥30°'], loc='lower left', bbox_to_anchor=(0.52, 0.0), frameon=False,
                 fontsize=6.0, handletextpad=0.1, borderaxespad=0.15, labelspacing=0.15, ncol=2, columnspacing=0.6)

# ---------------------------------------------------------------- g: right and wrong rotation
GX, GWd = 112.0, 55.0
axg = fig.add_axes(box(GX, YB3, GWd, R3H))
rows = [('lat < 30', '<30°'), ('lat < 30, Southern Hemisphere', '<30°, S.H.'), ('lat 30-45', '30–45°')]
for i, (sub, lab) in enumerate(rows):
    f = FE[(FE.subset == sub) & (FE['var'] == 'Tmax')].iloc[0]
    axg.bar(i - 0.19, f['within-sector a=0.4'], 0.36, color='#444444', zorder=3)
    axg.errorbar(i - 0.19, f['within-sector a=0.4'], yerr=[[f['within-sector a=0.4'] - f.ws_lo], [f.ws_hi - f['within-sector a=0.4']]],
                 color='k', lw=0.7, capsize=1.5, zorder=4)
    axg.bar(i + 0.19, f['within-sector wrong rotation'], 0.36, color='white', edgecolor='#444444', lw=0.7, zorder=3)
axg.axhline(0, color='#555555', lw=0.5, zorder=1)
axg.set_xticks(range(3)); axg.set_xticklabels([r[1] for r in rows]); axg.set_xlim(-0.6, 2.6)
axg.set_ylim(-0.85, 0.75); axg.set_yticks([-0.5, 0, 0.5]); axg.set_yticklabels([fmt(v) for v in (-0.5, 0, 0.5)])
axg.set_ylabel('Slope of daily maximum (K)', labelpad=1)
hg = [matplotlib.patches.Patch(facecolor='#444444'), matplotlib.patches.Patch(facecolor='white', edgecolor='#444444', lw=0.7)]
lg = axg.legend(hg, ['same storm direction,\nright rotation', 'wrong rotation'], loc='upper left', frameon=True, framealpha=1.0, edgecolor='none',
                facecolor='white', fontsize=6.2, handlelength=1.2, borderpad=0.2, labelspacing=0.2, borderaxespad=0.2)
lg.set_zorder(4)

# ---------------------------------------------------------------- letters
ya = H - 2.0
fig.text(1.0 / W, (ya - 0.3) / H, 'a', fontsize=9, fontweight='bold', va='top', ha='left')
for ax, s in zip([axf, axg], 'bc'):
    x0 = ax.get_position().x0
    fig.text(x0 - 15.0 / W, ax.get_position().y1 + 3.2 / H, s, fontsize=9, fontweight='bold', va='top', ha='left')
print('QA overlap:', overlap_check(fig)); print('QA occlusion:', occlusion_check(fig))
fig.savefig(f'{OUT}/FigS12.png', dpi=600); fig.savefig(f'{OUT}/FigS12.pdf')
print('stations on map', len(m), 'within 30', len(lo_), 'warmer off land', int((lo_.v > 0).sum()), 'beyond', len(mi_), int((mi_.v > 0).sum()))
