#!/usr/bin/env bash
# Redraw every figure of the manuscript from the data in this repository (run from the repository root).
# Output: fig/Fig1-4 and fig/FigS1-S13 (PNG and PDF); set FIGOUT to write elsewhere.
set -euo pipefail
export FIGOUT="${FIGOUT:-fig}"; mkdir -p "$FIGOUT"
for f in fig1_global_map fig2_storm_removal fig3_coastal_orientation fig4_impact_forecast \
         figS01_storm_direction figS02_split_sample figS03_guangdong_stations figS04_graphcast_regions \
         figS05_graphcast_cells figS06_guangdong_graphcast figS07_S09_graphcast_checks figS08_graphcast_circulation \
         figS10_storm_edit figS11_guangdong_wrf figS12_stations figS13_outer_winds; do
  echo "== $f"; python fig_code/$f.py
done
