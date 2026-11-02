# Shared style and checks for the TC dry-heat paper figures (adapted from the rain/fire paper figcommon.py).
import numpy as np, pandas as pd, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nature.mplstyle'))
# CEE: one sans typeface (Arial; Liberation Sans is metric-compatible), optimum 8 pt; ticks/annotations 7 pt, panel letters 9 pt bold
import matplotlib.font_manager as _fm
_SANS = 'Arial' if any(f.name == 'Arial' for f in _fm.fontManager.ttflist) else 'Liberation Sans'
plt.rcParams.update({'font.sans-serif': [_SANS, 'DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.labelpad': 2, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'legend.handletextpad': 0.4,
                     'legend.borderaxespad': 0.3, 'legend.labelspacing': 0.3, 'pdf.fonttype': 42})
FS = dict(tick=7, label=8, ann=7, letter=9)
def letter(ax, s, x, y, fig=None, **k):
    (fig or ax.figure).text(x, y, s, fontsize=9, fontweight='bold', va='top', ha='left', **k) if fig is not None else ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='left', **k)
MM = 1 / 25.4
# Okabe-Ito
OI = dict(orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442', blue='#0072B2', verm='#D55E00', purple='#CC79A7', grey='#999999', black='#000000')
import matplotlib.patheffects as _pe
HALO = [_pe.withStroke(linewidth=2.2, foreground='white')]
# paper colour identity: storm-made heat = brick red, cooling/control = steel blue, null/procedure = grey
RED, BLUE, GREY, LAND, SEA = '#C44E52', '#4C72B0', '#8C8C8C', '#E3E3E3', '#FFFFFF'
CFG = {'small_std': '#F2A65A', 'small_wide': '#C8601C', 'oper_std': '#B7A2D6', 'oper_wide': '#6A3D9A'}

def binned(base, bounds):
    # discrete classes for change maps: labelled bin edges, arrow ends for values beyond the last bin
    cols = plt.get_cmap(base)(np.linspace(0.01, 0.99, len(bounds) + 1))
    cm = matplotlib.colors.ListedColormap(cols[1:-1]); cm.set_under(cols[0]); cm.set_over(cols[-1])
    return cm, matplotlib.colors.BoundaryNorm(bounds, len(bounds) - 1)

def label(ax, s, x=-0.02, y=1.0, fig=None):
    (fig or ax.figure).text(x, y, s, fontsize=8, fontweight='bold', va='bottom', ha='left', transform=ax.transAxes if fig is None else None)

def overlap_check(fig):
    """Report overlapping text boxes (axes labels, ticks, annotations) in a rendered figure."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); boxes = []
    undrawn = set()
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            drawn = set(id(k) for k in axis._update_ticks())
            for k in axis.get_major_ticks() + axis.get_minor_ticks():
                if id(k) not in drawn: undrawn.update([id(k.label1), id(k.label2)])
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in undrawn: continue
        try: bb = t.get_window_extent(r)
        except Exception: continue
        if bb.width < 1: continue
        boxes.append((t.get_text()[:30], bb, t.get_rotation() not in (0, 90, 270) and not t.get_text().startswith('\n')))
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if boxes[i][2] and boxes[j][2]: continue   # two rotated tick labels: their upright boxes overlap, the glyphs do not
            if a.overlaps(b) and min(a.x1, b.x1) - max(a.x0, b.x0) > 1.5 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1.5: bad.append((boxes[i][0], boxes[j][0]))
    W, H = fig.bbox.width, fig.bbox.height
    out = [('OUTSIDE FIGURE', s) for s, bb, _ in boxes if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5]
    return bad + out + [('OCCLUDED', h) for h in occlusion_check(fig)]

def occlusion_check(fig, pad=0.5):
    """Report text (annotations, legends, labels) crossed by data lines, error bars or scatter markers in any axes."""
    from matplotlib.collections import PathCollection, LineCollection
    from matplotlib.legend import Legend
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi
    boxes = []
    for leg in fig.findobj(Legend):
        if leg.get_visible(): boxes.append(('legend:' + ','.join(t.get_text()[:12] for t in leg.get_texts())[:40], leg.get_window_extent(r), leg))
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip(): continue
        if any(isinstance(p, Legend) for p in [t.axes.get_legend() if t.axes else None] + fig.legends if p is not None and t in p.get_texts()): continue
        if getattr(t, '_is_tick', False): continue
        try: bb = t.get_window_extent(r)
        except Exception: continue
        if bb.width < 1: continue
        boxes.append((t.get_text()[:30], bb, t))
    def pts_line(ln):
        xy = ln.get_transform().transform(ln.get_xydata()); xy = xy[np.isfinite(xy).all(1)]
        if len(xy) < 2: return xy
        out = [xy[:1]]
        for a, b in zip(xy[:-1], xy[1:]):
            n = int(max(2, np.hypot(*(b - a)) / 1.0)); out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
        return np.vstack(out)
    hits = []
    for ax in fig.axes:
        if not ax.axison: art_lines = []; colls = [c for c in ax.collections if isinstance(c, PathCollection)]
        else: art_lines = [l for l in ax.lines if l.get_visible() and l.get_linestyle() not in ('None', '', ' ')]; colls = [c for c in ax.collections if isinstance(c, (PathCollection, LineCollection))]
        ctx = ax.get_window_extent(r)
        geoms = []
        for l in art_lines: geoms.append(('line', pts_line(l), 0.5 * l.get_linewidth() * dpi / 72))
        for l in ax.lines:
            if l.get_marker() not in (None, 'None', '', ' ') and l.get_visible():
                xy = l.get_transform().transform(l.get_xydata()); geoms.append(('marker', xy[np.isfinite(xy).all(1)], 0.5 * l.get_markersize() * dpi / 72))
        for c in colls:
            if isinstance(c, PathCollection):
                off = c.get_offsets()
                if len(off) == 0: continue
                xy = c.get_offset_transform().transform(off); s = c.get_sizes(); rad = (np.sqrt(s.max() if len(s) else 20) / 2) * dpi / 72
                geoms.append(('scatter', xy[np.isfinite(xy).all(1)], rad))
            else:
                for seg in c.get_segments():
                    if len(seg) < 2: continue
                    xy = c.get_transform().transform(seg); out = []
                    for a, b in zip(xy[:-1], xy[1:]):
                        n = int(max(2, np.hypot(*(b - a)) / 1.0)); out.append(a + (b - a) * np.linspace(0, 1, n)[:, None])
                    lw = (c.get_linewidths()[0] if len(c.get_linewidths()) else 1) * dpi / 72 / 2; geoms.append(('errbar', np.vstack(out), lw))
        for name, bb, art in boxes:
            if art in (ax.xaxis.label, ax.yaxis.label) or art in ax.get_xticklabels() + ax.get_yticklabels(): continue
            if not bb.overlaps(ctx): continue
            halo = any('Stroke' in type(pe).__name__ for pe in (art.get_path_effects() if hasattr(art, 'get_path_effects') else []))
            for kind, xy, rad in geoms:
                if len(xy) == 0 or (halo and kind in ('line', 'errbar')): continue
                inside = (xy[:, 0] > bb.x0 - rad + pad) & (xy[:, 0] < bb.x1 + rad - pad) & (xy[:, 1] > bb.y0 - rad + pad) & (xy[:, 1] < bb.y1 + rad - pad)
                if inside.any(): hits.append((name, kind)); break
    return sorted(set(hits))


# ---- data helpers shared by the GraphCast figures
def load_maps(path):
    z = np.load(path); keys = list(z['keys']); sc = z['scales']
    def get(key):
        i = keys.index(key); q = z['m%03d' % i].astype(np.float32)
        if key.endswith('|agree'): return np.where(q < 0, np.nan, q * 0.01)
        return np.where(q == -128, np.nan, q * sc[i])
    return get, {m: (z[f'lat_{m}'], z[f'lon_{m}']) for m in ('oper', 'small') if f'lat_{m}' in z}


def boot(d, groups, n=4000, seed=0):
    """Mean and 95% interval, resampling whole storms."""
    rng = np.random.default_rng(seed); g = pd.Series(d).groupby(np.asarray(groups)); s, c = g.sum().values, g.count().values
    k = rng.integers(0, len(s), (n, len(s))); m = s[k].sum(1) / c[k].sum(1)
    return np.mean(d), *np.percentile(m, [2.5, 97.5])
