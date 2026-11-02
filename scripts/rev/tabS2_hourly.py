"""Add the hourly wet-bulb measures and their sample coverage to Table S2 (second review, 2026-10-30).

Run after 91_tables.py.  Reads OBS/fingerprints.csv and OBS/coverage.csv (10_obs_gd_v2.py, the second with --coverage)
and rewrites parts/tabS2.md as two blocks: the daily maximum temperature variants and the hourly wet-bulb measures.
Usage: python -I tabS2_hourly.py OBS PARTS
"""
import os, sys
import numpy as np, pandas as pd

obs, parts = sys.argv[1:3]
fm = lambda v: (f'{np.sign(v) * np.floor(abs(v) * 100 + 0.5) / 100:+.2f}').replace('-', '−')
ci = lambda lo, hi: f"{fm(lo).lstrip('+')} to {fm(hi).lstrip('+')}" if lo < 0 else f"{fm(lo).lstrip('+')}–{fm(hi).lstrip('+')}"
src = open(f'{parts}/tabS2.md').read().split('\n')
head, body = src[:2], [l for l in src[2:] if l.startswith('| ') and not l.startswith('| *')]
body = [l for l in body if 'wet-bulb' not in l.lower() and 'reports' not in l and 'local time (' not in l]
f = pd.read_csv(f'{obs}/fingerprints.csv'); f = f[f.group == 'east'].set_index('var')
cov = pd.read_csv(f'{obs}/coverage.csv'); cov = cov[cov.group == 'east'].set_index('measure')
rows = head + ['| *Daily maximum temperature (GSOD)* | | | | |'] + body + ['| *Wet-bulb temperature from hourly reports (ISD-Lite)* | | | | |']
for var, meas, lab in (('Tw_14LT', '14 LT', 'At 14 local time'), ('Tw_12_17LT', '12-17 LT mean', 'Mean over 12–17 local time'),
                       ('Tw_at_Tmax', 'hour of highest temperature', 'At the hour of the highest temperature'),
                       ('Tw_daily_max', 'highest of the reports, >= 4 reports', 'Highest of the day’s reports (at least four)'),
                       ('Tw_daily_max_8obs', 'highest of the reports, >= 8 reports', 'Highest of the day’s reports (at least eight)')):
    r, c = f.loc[var], cov.loc[meas]
    rows.append(f'| {lab}; {100 * c.share_station_days:.0f}% of station-days | {int(r.n_days)} | {int(r.n_storms)} | {fm(r.excess)} | {ci(r.lo, r.hi)} |')
open(f'{parts}/tabS2.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
