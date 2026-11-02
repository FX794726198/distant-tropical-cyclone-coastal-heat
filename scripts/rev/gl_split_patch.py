"""Make split-sample versions of gl_map.py and gl_stage2.py (independent review, 2026-10-29).

YEARSET=odd|even restricts the statistics to the storm seasons of odd or even years.  Shared with the all-year run: the
harmonic and trend fit of each grid point and the storm-season months (season_mask from all years' storm codes).  Restricted
to the half: the two reference-day corrections in matched(), the exposure days, the stage-2 analogue pool and the bootstrap
over years or storms.  GLOUT is the output directory and, for stage 2, GLIN the directory of the gl_map.py run whose regions
are tested.  Discovery on one half and estimation on the other: no region or direction is chosen with the years that test it.
Usage: python gl_split_patch.py DIR_WITH_gl_map.py
"""
import os, sys
d = sys.argv[1]
s = open(os.path.join(d, 'gl_map.py')).read()
for a, b in (("OUT = os.path.join(GL, 'out')", "OUT = os.environ.get('GLOUT', os.path.join(GL, 'out'))"),
             ("    seas_d, seas_m = season_mask(code, days)\n",
              "    seas_d, seas_m = season_mask(code, days)\n    YS = os.environ.get('YEARSET', '')\n"
              "    if YS:\n        seas_d &= ((days.year.values % 2) == (1 if YS == 'odd' else 0))[:, None]\n"),
             ("    W = np.stack([np.bincount(rng.integers(0, ny, ny), minlength=ny) for _ in range(a.nboot)]).astype(float)",
              "    ys_ = np.array([k for k in range(ny) if not YS or (y0 + k) % 2 == (1 if YS == 'odd' else 0)])\n"
              "    W = np.stack([np.bincount(rng.choice(ys_, len(ys_)), minlength=ny) for _ in range(a.nboot)]).astype(float)")):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(os.path.join(d, 'gl_map_split.py'), 'w').write(s)
s = open(os.path.join(d, 'gl_stage2.py')).read()
for a, b in (("OUT = os.path.join(GL, 'out')", "OUT = os.environ.get('GLOUT', os.path.join(GL, 'out')); IN_ = os.environ.get('GLIN', OUT)"),
             ("pd.read_csv(os.path.join(OUT, 'gl_cells.csv'))", "pd.read_csv(os.path.join(IN_, 'gl_cells.csv'))"),
             ("np.load(os.path.join(OUT, 'gl_codes.npz'))", "np.load(os.path.join(IN_, 'gl_codes.npz'))"),
             ("pd.read_csv(os.path.join(OUT, 'gl_regions.csv'))", "pd.read_csv(os.path.join(IN_, 'gl_regions.csv'))"),
             ("        sd = mon_ok[mon - 1]\n",
              "        sd = mon_ok[mon - 1]\n        YS = os.environ.get('YEARSET', '')\n"
              "        if YS:\n            sd = sd & ((yr % 2) == (1 if YS == 'odd' else 0))\n")):
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(os.path.join(d, 'gl_stage2_split.py'), 'w').write(s)
print('wrote gl_map_split.py and gl_stage2_split.py in', d)
