"""Supplementary GraphCast figures.
Fig. S3 (FigS3.png): effect of the storm on 14 LT 2-m temperature for the other removal variants, the checks and the control bias (hot cases).
Fig. S5 (FigS5.png): effect of the storm and control bias against forecast lead time.
Inputs: GCX (fp_maps.npz, fp_case.csv, fp_lead.csv), DB (gc_cases.csv, gc_tracks.csv, gd_stations.csv)."""
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
pc = ccrs.PlateCarree()
get, grids = load_maps(f'{GCX}/fp_maps.npz')
cases = pd.read_csv(f'{DB}/gc_cases.csv'); trk = pd.read_csv(f'{DB}/gc_tracks.csv'); stn = pd.read_csv(f'{DB}/gd_stations.csv')
pos = trk.merge(cases[['case', 'V', 'kind']], left_on=['case', 'time'], right_on=['case', 'V'])
pos = pos[pos.kind == 'heat']

bounds = np.arange(-3, 3.01, 0.5)
cols = plt.get_cmap('RdBu_r')(np.linspace(0.06, 0.94, len(bounds) + 1))
cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1]); norm = BoundaryNorm(bounds, cmap.N)

# ---------------- maps
PAN = [('small|heat|NOTC|t2m|mean', None, 'small', '1°, core removal'),
       ('small|heat|WIDE|t2m|mean', 'small|heat|WIDE|t2m|agree', 'small', '1°, full removal'),
       ('oper|heat|NOTC|t2m|mean', 'oper|heat|NOTC|t2m|agree', 'oper', '0.25°, core removal'),
       ('oper|heat|WIDEH|t2m|mean', None, 'oper', '0.25°, half of full removal'),
       ('oper|heat|NULLW|t2m|mean', None, 'oper', '0.25°, full edit, no TC'),
       ('oper|heat|SHIFTN|t2m|mean', None, 'oper', '0.25°, TC moved southwest'),
       ('small|heat|CTLminusERA5|t2m|mean', None, 'small', '1°, control minus ERA5'),
       ('oper|heat|CTLminusERA5|t2m|mean', None, 'oper', '0.25°, control minus ERA5')]
W = 180; MW = 41.5; MH = MW * (EXT[3] - EXT[2]) / (EXT[1] - EXT[0]); GX = 2.3; GY = 7.0
H = 4.0 + 2 * MH + GY + 15.0
fig = plt.figure(figsize=(W * MM, H * MM))
box = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
x0s = [4.5 + i * (MW + GX) for i in range(4)]
for n, (k, ak, grid, title) in enumerate(PAN):
    r, c = divmod(n, 4)
    y0 = H - 4.0 - MH - r * (MH + GY)
    ax = fig.add_axes(box(x0s[c], y0, MW, MH), projection=pc)
    ax.set_extent(EXT, crs=pc); ax.set_xticks([]); ax.set_yticks([])
    la, lo = grids[grid]; f = get(k)
    if grid == 'small':
        ax.pcolormesh(lo, la, f, cmap=cmap, norm=norm, transform=pc, zorder=1, shading='nearest', rasterized=True)
    else:
        ax.contourf(lo, la, f, levels=bounds, cmap=cmap, norm=norm, extend='both', transform=pc, zorder=1)
    if ak is not None:
        ag = get(ak); LO, LA = np.meshgrid(lo, la)
        sel = (np.abs(ag - 0.5) >= 0.3) & (np.round(LO * 2) % 2 == 0) & (np.round(LA * 2) % 2 == 0)
        ax.scatter(LO[sel], LA[sel], s=0.4, color='k', lw=0, transform=pc, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.4, edgecolor='#333333', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.25, edgecolor='#666666', zorder=3)
    if not any(t in k for t in ('ERA5', 'NULLW', 'SHIFTN')):
        ax.scatter(pos.lon, pos.lat, s=6, facecolor='none', edgecolor='k', lw=0.4, transform=pc, zorder=5)
    ax.scatter(stn.lon, stn.lat, s=4, marker='^', color='k', lw=0, transform=pc, zorder=5)
    ax.text(0.025, 0.965, title, transform=ax.transAxes, ha='left', va='top', fontsize=6.5, path_effects=HALO, zorder=7)
    fig.text((x0s[c] - 0.2) / W, (y0 + MH + 3.6) / H, 'abcdefgh'[n], fontsize=9, fontweight='bold', va='top', ha='left')
cax = fig.add_axes(box(52, 8.0, 76, 1.8))
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both', ticks=bounds[::2])
cb.set_label('2-m temperature difference at 14 LT (K)', fontsize=7, labelpad=1)
cb.ax.tick_params(labelsize=7, length=2, pad=1); cb.outline.set_linewidth(0.4)
os.makedirs(OUT, exist_ok=True)
print('QA maps:', overlap_check(fig))
fig.savefig(f'{OUT}/FigS7.png', dpi=600); fig.savefig(f'{OUT}/FigS7.pdf'); plt.close(fig)

# ---------------- lead time
L = pd.read_csv(f'{GCX}/fp_lead.csv', keep_default_na=False, na_values=['']); L['variant'] = L.variant.fillna('NULL')
C = pd.read_csv(f'{GCX}/fp_case.csv', keep_default_na=False, na_values=['']); C['variant'] = C.variant.fillna('NULL')
kind = C.drop_duplicates('case').set_index('case').kind; tid = C.drop_duplicates('case').set_index('case').tid
L['kind'] = L.case.map(kind); L['tid'] = L.case.map(tid)
Pl = L.pivot_table(index=['model', 'case', 'kind', 'tid', 'lead'], columns='variant', values='T').reset_index()
Tt = L[L.variant == 'CTL'].set_index(['model', 'case', 'lead']).T_truth
Pl['truth'] = [Tt.get((m, c, l), np.nan) for m, c, l in zip(Pl.model, Pl.case, Pl.lead)]
CONF = [('small', 'NOTC', 'small_std', '1°, core'), ('small', 'WIDE', 'small_wide', '1°, full'),
        ('oper', 'NOTC', 'oper_std', '0.25°, core'), ('oper', 'WIDE', 'oper_wide', '0.25°, full')]
W2, H2 = 180, 62
fig = plt.figure(figsize=(W2 * MM, H2 * MM))
box2 = lambda x0, y0, w, h: [x0 / W2, y0 / H2, w / W2, h / H2]
axa = fig.add_axes(box2(14, 10, 72, 48)); axb = fig.add_axes(box2(104, 10, 72, 48))
rows = []
for i, (m, v, ck, lab) in enumerate(CONF):
    for kd, ls, mfc in (('heat', '-', CFG[ck]), ('placebo', (0, (3, 1.5)), 'white')):
        q = Pl[(Pl.model == m) & (Pl.kind == kd)].dropna(subset=['CTL', v])
        xs, ys, lo_, hi_ = [], [], [], []
        for ld, g in q.groupby('lead'):
            mu, a, b = boot((g['CTL'] - g[v]).values, g.tid.values, n=2000, seed=ld + i)
            xs.append(ld + (i - 1.5) * 0.7); ys.append(mu); lo_.append(a); hi_.append(b)
            rows.append(dict(conf=lab, kind=kd, lead=ld, n=len(g), effect=mu, lo=a, hi=b))
        axa.errorbar(xs, ys, yerr=[np.subtract(ys, lo_), np.subtract(hi_, ys)], color=CFG[ck], ls=ls, marker='o', ms=2.8, mfc=mfc,
                     mec=CFG[ck], lw=0.9, elinewidth=0.6, label=lab if kd == 'heat' else None, zorder=3)
axa.axhline(0, color='#555555', lw=0.5, zorder=1)
axa.set_xticks([24, 30, 36, 42, 48]); axa.set_xlabel('Forecast lead time (h)', labelpad=1)
axa.set_ylabel('Effect of the TC on\n14 LT temperature (K)')
h, l = axa.get_legend_handles_labels()
h += [Line2D([], [], color='#7F7F7F', ls='-', marker='o', ms=2.8, lw=0.9), Line2D([], [], color='#7F7F7F', ls=(0, (3, 1.5)), marker='o', ms=2.8, mfc='white', lw=0.9)]
l += ['TC east, hot days', 'TC southwest']
yl = axa.get_ylim(); axa.set_ylim(yl[0], yl[1] + 0.3 * (yl[1] - yl[0]))
axa.legend(h, l, loc='upper left', ncol=3, frameon=False, handlelength=1.6, columnspacing=0.8, borderaxespad=0.1)
for i, (m, ck) in enumerate((('small', 'small_wide'), ('oper', 'oper_wide'))):
    for kd, ls, mfc in (('heat', '-', CFG[ck]), ('placebo', (0, (3, 1.5)), 'white')):
        q = Pl[(Pl.model == m) & (Pl.kind == kd)].dropna(subset=['CTL', 'truth'])
        xs, ys, lo_, hi_ = [], [], [], []
        for ld, g in q.groupby('lead'):
            mu, a, b = boot((g['CTL'] - g['truth']).values, g.tid.values, n=2000, seed=ld)
            xs.append(ld + (i - 0.5) * 0.8); ys.append(mu); lo_.append(a); hi_.append(b)
            rows.append(dict(conf=('1°' if m == 'small' else '0.25°') + ' control minus ERA5', kind=kd, lead=ld, n=len(g), effect=mu, lo=a, hi=b))
        axb.errorbar(xs, ys, yerr=[np.subtract(ys, lo_), np.subtract(hi_, ys)], color=CFG[ck], ls=ls, marker='o', ms=2.8, mfc=mfc, mec=CFG[ck],
                     lw=0.9, elinewidth=0.6, zorder=3, label=('1° model' if m == 'small' else '0.25° model') if kd == 'heat' else None)
axb.axhline(0, color='#555555', lw=0.5, zorder=1)
axb.set_xticks([24, 30, 36, 42, 48]); axb.set_xlabel('Forecast lead time (h)', labelpad=1)
axb.set_ylabel('Control forecast minus ERA5,\n19 stations (K)')
yl = axb.get_ylim(); axb.set_ylim(yl[0], yl[1] + 0.15 * (yl[1] - yl[0]))
axb.legend(loc='upper right', frameon=False, ncol=2, handlelength=1.6, borderaxespad=0.1)
fig.text(1 / W2, 61 / H2, 'a', fontsize=9, fontweight='bold', va='top', ha='left')
fig.text(91 / W2, 61 / H2, 'b', fontsize=9, fontweight='bold', va='top', ha='left')
print('QA lead:', overlap_check(fig))
pd.DataFrame(rows).to_csv(f'{OUT}/FigS9_summary.csv', index=False, float_format='%.3f')
fig.savefig(f'{OUT}/FigS9.png', dpi=600); fig.savefig(f'{OUT}/FigS9.pdf'); plt.close(fig)
