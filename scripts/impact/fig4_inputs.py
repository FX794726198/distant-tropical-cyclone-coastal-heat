"""Two drawing tables of Fig. 4 made from the impact outputs (impact.py, impact_ext.py):
  cells_warm.csv  grid points of the warm regions with any storm-attributed hot day: lon, lat, att_f0.50_per_yr
                  (storm-attributed hot days per year with half of the mapped warming removed, from impact_cells.csv)
  cont_years.csv  storm-attributed hot person-days per year (millions, 2020 population) summed over the warm regions of
                  each continent group, from impact_region_years.csv (Fig. 4c)
Usage: python -I fig4_inputs.py data/impact   (reads and writes in that folder)"""
import os, sys
import pandas as pd

D = sys.argv[1]
C = pd.read_csv(os.path.join(D, 'impact_cells.csv'))
W = C[C['att_f0.50_per_yr'] > 0][['lon', 'lat', 'att_f0.50_per_yr']].copy()
W['att_f0.50_per_yr'] = W['att_f0.50_per_yr'].round(3); W = W.sort_values(['lon', 'lat'])
W.to_csv(os.path.join(D, 'cells_warm.csv'), index=False)
# continent groups of the 23 warm regions (AF Africa, AM Americas, AU Australia and New Guinea, EA East and Southeast Asia,
# SA South Asia)
GROUP = {**{r: 'AF' for r in ('W1', 'W2', 'W3', 'W4')}, **{r: 'SA' for r in ('W5', 'W6')},
         **{r: 'EA' for r in ('W7', 'W9', 'W10', 'W12', 'W13')}, **{r: 'AU' for r in ('W8', 'W11', 'W14', 'W15', 'W16')},
         **{f'W{i}': 'AM' for i in range(17, 24)}}
Y = pd.read_csv(os.path.join(D, 'impact_region_years.csv'))
assert set(Y.region) == set(GROUP), sorted(set(Y.region) ^ set(GROUP))
Y['group'] = Y.region.map(GROUP)
P = Y.pivot_table(index='year', columns='group', values='persondays_M', aggfunc='sum')[['AF', 'AM', 'AU', 'EA', 'SA']].round(2)
P.to_csv(os.path.join(D, 'cont_years.csv'))
print(len(W), 'grid points;', len(P), 'years')
