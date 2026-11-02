"""Fig. 3: removing the storm from GraphCast initial states (input: gc_export.py output dir GCX, case and track tables)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figcommon import *
import cartopy.crs as ccrs, cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

GCX = os.environ.get('GCX', 'data/figpack')
DB = os.environ.get('DB', 'data/db')
OUT = os.environ.get('FIGOUT', 'fig')
EXT = [106, 128, 16, 29]
CONF = [('small', 'NOTC', 'NULL', 'WEAK', 'small_std', '1°, core'), ('small', 'WIDE', 'NULLW', 'WIDEH', 'small_wide', '1°, full'),
        ('oper', 'NOTC', 'NULL', 'WEAK', 'oper_std', '0.25°, core'), ('oper', 'WIDE', 'NULLW', 'WIDEH', 'oper_wide', '0.25°, full')]
REV = os.environ.get('REV', 'data/revision/figure_inputs')          # revision pack with effects_case.csv: station-free (WIDEL) and 1,000-km (W1000) removals
# panel c: removal definitions ordered by how far the edit reaches, each followed by its null edit where one was run
DEFS = [('NOTC', 'NULL', 'Core'), ('WIDEL', None, 'Station-\nfree'), ('W1000', None, '1,000\nkm'), ('WIDE', 'NULLW', 'Full')]
SHADE = {'small': {'NOTC': '#F2A65A', 'WIDEL': '#E2873A', 'W1000': '#D57427', 'WIDE': '#C8601C'},
         'oper': {'NOTC': '#B7A2D6', 'WIDEL': '#9C82C6', 'W1000': '#8463B3', 'WIDE': '#6A3D9A'}}


get, grids = load_maps(f'{GCX}/fp_maps.npz' if os.path.exists(f'{GCX}/fp_maps.npz') else f'{GCX}/gcx_maps.npz')
CF = f'{GCX}/fp_case.csv' if os.path.exists(f'{GCX}/fp_case.csv') else f'{GCX}/gcx_case.csv'
C = pd.read_csv(CF, keep_default_na=False, na_values=[''])
C['variant'] = C.variant.fillna('NULL')                 # 'NULL' is read as missing by default; a blank variant can only be NULL
cases = pd.read_csv(f'{DB}/gc_cases.csv'); trk = pd.read_csv(f'{DB}/gc_tracks.csv')
stn = pd.read_csv(f'{DB}/gd_stations.csv')
pos = trk.merge(cases[['case', 'V', 'kind']], left_on=['case', 'time'], right_on=['case', 'V'])

# per-case effect of the storm: d = CTL - variant at the 19 stations, lead mean
P = C.pivot_table(index=['model', 'case', 'kind', 'tid'], columns='variant', values='t2m').reset_index()
if REV and os.path.exists(f'{REV}/effects_case.csv'):
    R = pd.read_csv(f'{REV}/effects_case.csv', keep_default_na=False, na_values=[''])
    R['model'] = np.where(R.dir.str.contains('oper'), 'oper', 'small')
    for v in ('WIDEL', 'W1000'):
        r = R[R.variant == v].set_index(['model', 'case']).d_t2m
        key = pd.MultiIndex.from_frame(P[['model', 'case']])
        P[v] = P['CTL'].values - r.reindex(key).values          # stored so that CTL minus the column is the effect of the storm

bounds = np.arange(-3, 3.01, 0.5)
cols = plt.get_cmap('RdBu_r')(np.linspace(0.06, 0.94, len(bounds) + 1))
cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1]); norm = BoundaryNorm(bounds, cmap.N)

W, H = 180, 108
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
MW = 70.0; MH = MW * (EXT[3] - EXT[2]) / (EXT[1] - EXT[0])
pc = ccrs.PlateCarree()
la, lo = grids['oper']
for j, kind in enumerate(('heat', 'placebo')):
    ax = fig.add_axes(box(9 + 80 * j, H - 3.5 - MH, MW, MH), projection=pc)
    ax.set_extent(EXT, crs=pc); ax.set_xticks([]); ax.set_yticks([])
    f = get(f'oper|{kind}|WIDE|t2m|mean'); ag = get(f'oper|{kind}|WIDE|t2m|agree')
    ax.contourf(lo, la, f, levels=bounds, cmap=cmap, norm=norm, extend='both', transform=pc, zorder=1)
    LO, LA = np.meshgrid(lo, la); sel = (np.abs(ag - 0.5) >= 0.3) & (np.round(LO * 2) % 2 == 0) & (np.round(LA * 2) % 2 == 0)
    ax.scatter(LO[sel], LA[sel], s=0.5, color='k', lw=0, transform=pc, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.45, edgecolor='#333333', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.3, edgecolor='#666666', zorder=3)
    p = pos[pos.kind == kind]
    ax.scatter(p.lon, p.lat, s=10, facecolor='none', edgecolor='k', lw=0.5, transform=pc, zorder=5)
    ax.scatter(stn.lon, stn.lat, s=7, marker='^', color='k', lw=0, transform=pc, zorder=5)
    gl = ax.gridlines(draw_labels=True, lw=0.25, color='#9E9E9E', ls=':', xlocs=range(105, 131, 5), ylocs=range(15, 31, 5), zorder=4)
    gl.top_labels = gl.right_labels = False; gl.left_labels = (j == 0)
    gl.xlabel_style = gl.ylabel_style = dict(size=7); gl.xpadding = gl.ypadding = 2
    n = int((P.model.eq('oper') & P.kind.eq(kind)).sum())
    ax.text(0.015, 0.975, f'{n} cases', transform=ax.transAxes, ha='left', va='top', fontsize=7, zorder=6, path_effects=HALO)
    fig.text((1 + 82.5 * j) / W, 1 - 2.5 / H, 'ab'[j], fontsize=9, fontweight='bold', va='top', ha='left')
cax = fig.add_axes(box(162.5, H - 3.5 - MH + 3, 2.2, MH - 6))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend='both', ticks=bounds[::2])
cb.set_label('Temperature with storm minus\nwithout storm, 14 LT (K)', fontsize=7, labelpad=2); cb.ax.tick_params(labelsize=7, length=2)
cb.outline.set_linewidth(0.4)

# c: per-case effects for each removal definition and model; filled, storm east on hot days; open, storm southwest
axc = fig.add_axes(box(15, 13, 96, 35))
rng = np.random.default_rng(3); xt, xl, summ = [], [], []
x = 0.0
for i, m in enumerate(('small', 'oper')):
    x0m = x
    for v, nv, lab in DEFS:
        if v not in P or P.loc[P.model == m, v].isna().all():
            continue
        pad = 0.13 if v == 'WIDEL' else 0.0                 # room for the two-line label next to its neighbours
        x += pad
        for var, fill in ((v, True), (nv, None)):
            if var is None or var not in P:
                continue
            for kind in (('heat', 'placebo') if fill else ('heat',)):
                q = P[(P.model == m) & (P.kind == kind)].dropna(subset=['CTL', var])
                d = (q['CTL'] - q[var]).values
                if len(d) == 0: continue
                col = SHADE[m][v] if fill else GREY
                face = col if (kind == 'heat') else 'white'
                axc.scatter(x + rng.uniform(-0.27, 0.27, len(d)), d, s=6, facecolor=face, edgecolor='none' if kind == 'heat' else col,
                            lw=0.55, zorder=3, alpha=0.9 if fill else 0.7)
                mu, lo_, hi_ = boot(d, q.tid.values, seed=len(summ))
                axc.plot([x - 0.34, x + 0.34], [mu, mu], color='k', lw=1.1, zorder=4, solid_capstyle='butt')
                axc.plot([x, x], [lo_, hi_], color='k', lw=0.8, zorder=4)
                summ.append(dict(model=m, removal=v, kind=kind, var=var, n=len(d), mean=mu, lo=lo_, hi=hi_, frac_pos=np.mean(d > 0)))
                if fill and kind == 'heat':
                    axc.text(x, d.max() + 0.12, f'{mu:+.2f}', ha='center', va='bottom', fontsize=6.5)
            xt.append(x); xl.append(lab if fill else 'Null'); x += 1.0
        x += pad
    axc.annotate('', xy=(x0m - 0.4, -0.195), xytext=(x - 0.6, -0.195), xycoords=('data', 'axes fraction'),
                 arrowprops=dict(arrowstyle='-', lw=0.5, color='#555555'), annotation_clip=False)
    axc.text((x0m + x - 1.0) / 2, -0.205, ('1° model', '0.25° model')[i], transform=axc.get_xaxis_transform(), ha='center', va='top', fontsize=7)
    x += 0.6
axc.axhline(0, color='#555555', lw=0.5, zorder=1)
axc.set_xticks(xt); axc.set_xticklabels(xl, fontsize=6.5, linespacing=0.95)
axc.tick_params(axis='x', length=0, pad=2)
axc.set_xlim(-0.6, x - 0.9)
axc.set_ylabel('Effect of the storm on\n14 LT temperature (K)')
S = pd.DataFrame(summ)
ylo, yhi = axc.get_ylim(); axc.set_ylim(ylo - 0.13 * (yhi - ylo), yhi + 0.06 * (yhi - ylo))
h = [Line2D([], [], ls='none', marker='o', ms=3.4, mfc='#7F7F7F', mec='none', label='Storm east, hot days'),
     Line2D([], [], ls='none', marker='o', ms=3.4, mfc='white', mec='#7F7F7F', label='Storm southwest'),
     Line2D([], [], ls='none', marker='o', ms=3.4, mfc=GREY, mec='none', alpha=0.7, label='Same edit, no storm')]
axc.legend(handles=h, loc='lower left', ncol=3, frameon=False, handletextpad=0.1, columnspacing=0.9, borderaxespad=0.1)
fig.text(1 / W, 51.5 / H, 'c', fontsize=9, fontweight='bold', va='top', ha='left')

# d: graded removal (dose), hot days with the storm to the east
axd = fig.add_axes(box(131, 13, 47, 35))
for i, (m, v, nv, hv, ck, lab) in enumerate(CONF):
    q = P[(P.model == m) & (P.kind == 'heat')]
    xs, ys, lo_s, hi_s = [0.0], [0.0], [0.0], [0.0]
    for frac, var in ((0.5, hv), (1.0, v)):
        qq = q.dropna(subset=['CTL', var])
        if len(qq) == 0: continue
        mu, a, b = boot((qq['CTL'] - qq[var]).values, qq.tid.values, seed=50 + i)
        xs.append(frac); ys.append(mu); lo_s.append(a); hi_s.append(b)
    off = (i - 1.5) * 0.022
    axd.errorbar(np.array(xs) + off, ys, yerr=[np.array(ys) - lo_s, np.array(hi_s) - ys], color=CFG[ck], marker='o', ms=3.2, lw=1.0,
                 elinewidth=0.8, capsize=0, label=lab, zorder=3)
    qq = q.dropna(subset=['CTL', nv])
    if len(qq):
        mu, a, b = boot((qq['CTL'] - qq[nv]).values, qq.tid.values, seed=80 + i)
        axd.errorbar(1.08 + off, mu, yerr=[[mu - a], [b - mu]], color=GREY, marker='o', ms=3.2, mfc='white', mec=CFG[ck], lw=0, elinewidth=0.8, zorder=3)
axd.axhline(0, color='#555555', lw=0.5, zorder=1)
axd.set_xticks([0, 0.5, 1.0]); axd.set_xticklabels(['0', '½', '1'])
axd.set_xlim(-0.08, 1.16)
axd.set_xlabel('Share of the storm removed')
axd.set_ylabel('Effect of the storm (K)')
hd, ld = axd.get_legend_handles_labels()
hd.append(Line2D([], [], ls='none', marker='o', ms=3.2, mfc='white', mec='#7F7F7F')); ld.append('Same edit, no storm')
axd.legend(hd, ld, loc='upper left', frameon=False, handlelength=1.2, borderaxespad=0.1, labelspacing=0.3)
yl = axd.get_ylim(); axd.set_ylim(min(yl[0], -0.12), yl[1] + 0.25 * (yl[1] - yl[0]))   # room for the legend above the lines
fig.text(117 / W, 51.5 / H, 'd', fontsize=9, fontweight='bold', va='top', ha='left')

os.makedirs(OUT, exist_ok=True)
S.to_csv(f'{OUT}/FigS6_summary.csv', index=False, float_format='%.3f')
print(S.to_string(index=False))
print('QA:', overlap_check(fig))
fig.savefig(f'{OUT}/FigS6.png', dpi=600); fig.savefig(f'{OUT}/FigS6.pdf')
