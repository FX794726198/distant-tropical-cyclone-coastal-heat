# A3 impact: how to run on DelftBlue

Root `/scratch/$USER/tcheat/impact/` (new). `/scratch/$USER/tcheat/global` is read only.

1. Copy `pop_get.sh pop_agg.py impact.py impact.slurm` to `$IMP/scripts/`, and `gcg_regions.json`, `all_oper_regions.csv`
   (A2 region table, from `ggc_all/out_oper/` or the paper's `data/ggc/`) to `$IMP/`.
2. Login node: `python3 -m venv $IMP/venv && $IMP/venv/bin/pip install numpy pandas rasterio`.
3. Login node: `bash $IMP/scripts/pop_get.sh` (five GHS-POP epochs, about 2–3 GB, public JRC download).
4. `sbatch $IMP/scripts/impact.slurm` (aggregates the population once, then runs `impact.py`, < 1 h).
5. Results in `$IMP/out/`: `impact_result.txt` (main numbers), `impact_regions.csv`, `impact_years.csv`,
   `impact_cells.csv`; copy them to the local archive `nc_upgrade/impact_out/` with `pop/pop_1deg.npz` and `pop/pop_md5.txt`.

## Fig. 4 extension (2026-11-01, NC v10)
6. Copy `impact_ext.py` and `impact_ext.slurm` to `$IMP/scripts/` and run `sbatch $IMP/scripts/impact_ext.slurm` (~10 min).
   Needs `$GL/ibtracs.since1980.list.v04r01.csv` (read only). Outputs in `$IMP/out/`: `impact_ext_result.txt`,
   `impact_season.csv`, `impact_severity.csv`, `impact_sequence.csv`, `impact_cities.csv`, `impact_events.csv`.
   The first line of the result checks the storm-caused total against impact.py (0.267 bn person-days per year).
