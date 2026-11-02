"""Table S5 from the full-precision WRF source table (fourth review, 2026-10-30).
Usage: python -I tabS5_wrf.py PARTS_DIR WRF_DIR
Reads PARTS_DIR/tabS5_base.md (case, storm, day, distance, start, edited run, new low) and WRF_DIR/wrf_tabS5.csv
(scripts/model/wrf/wrf_tabS5.py); writes PARTS_DIR/tabS5.md with the three differences (control minus edited run)."""
import os, sys
import pandas as pd

parts, wrf = sys.argv[1:3]
W = pd.read_csv(os.path.join(wrf, 'wrf_tabS5.csv'))
W['kind'] = W.edited.map(lambda e: 'Null edit' if e.startswith('null') else 'Storm removed')
W = W.set_index(['case', 'start_h', 'kind'])
fmt = lambda x: '0.00' if abs(x) < 0.005 else f'{x:+.2f}'.replace('-', '−')
out = []
for line in open(os.path.join(parts, 'tabS5_base.md')).read().strip().split('\n'):
    c = [x.strip() for x in line.strip().strip('|').split('|')]
    if not c[0].isdigit():                                       # header and rule lines are kept as they are
        out.append(line); continue
    r = W.loc[(int(c[0]), int(c[4]), c[5])]
    c[6:9] = [fmt(r.dT14), fmt(r.dTmax), fmt(r.dQ14)]
    out.append('| ' + ' | '.join(c) + ' |')
open(os.path.join(parts, 'tabS5.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
