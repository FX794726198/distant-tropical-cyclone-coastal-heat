"""Add the station-free removal to Table S3 (revision v02).  Usage: python -I tabS3_addsf.py PARTS_DIR REVPACK_DIR
Reads PARTS_DIR/tabS3_v01cols.md (case table with the full-removal effects) and REVPACK_DIR/effects_case.csv; writes PARTS_DIR/tabS3.md."""
import os, sys
import pandas as pd

parts, rev = sys.argv[1:3]
E = pd.read_csv(os.path.join(rev, 'effects_case.csv'), keep_default_na=False, na_values=[''])
E['model'] = E.dir.str.contains('oper').map({True: 'oper', False: 'small'})
sf = E[E.variant == 'WIDEL'].set_index(['model', 'case']).d_t2m
f = lambda m, c: f'{sf[(m, c)]:+.1f}' if (m, c) in sf.index else '–'
out = []
for line in open(os.path.join(parts, 'tabS3_v01cols.md')).read().strip().split('\n'):
    c = [x.strip() for x in line.strip('|').split('|')]
    if c[0] == 'Case':
        c = c[:5] + ['1° full (K)', '1° station-free (K)', '0.25° full (K)', '0.25° station-free (K)']
    elif set(c[0]) <= set('-'):
        c = ['---'] * 9
    elif c[0].startswith('*'):
        c = c[:5] + [''] * 4
    else:
        k = int(c[0]); c = c[:5] + [c[5], f('small', k), c[6], f('oper', k)]
        c = [('0.0' if x in ('+0.0', '-0.0') else x) for x in c]          # no sign on a value that rounds to zero
    out.append('| ' + ' | '.join(c) + ' |')
open(os.path.join(parts, 'tabS3.md'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out[:6]))
