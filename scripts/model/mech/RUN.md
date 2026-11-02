# A4 multi-coast mechanism: how to run on DelftBlue (exploratory, no pass/fail)

Reads the A2 GraphCast case files READ-ONLY: /scratch/$USER/tcheat/ggc_all/out_oper and out_small (case_*.nc).
Never write, move or delete anything under ggc_all or tcheat/global. Root for this analysis: /scratch/$USER/tcheat/mech/.

1. Copy this folder (the local archive, nc_upgrade/scripts/mech/) to DelftBlue:
   scripts -> /scratch/$USER/tcheat/mech/scripts/ (coast_normals.py, mech_gc.py, mech_win.py, mech_era5.py, mech.slurm)
   inputs/ -> /scratch/$USER/tcheat/mech/inputs/ (all_oper_regions.csv, all_oper_cases.csv, all_small_cases.csv,
   coast_normals.csv, mech_era5.csv). Check md5 against MD5.txt.
2. sbatch /scratch/$USER/tcheat/mech/scripts/mech.slurm   (CPU, ~10-30 min; log /scratch/$USER/tcheat/logs/a4_mech_<job>.log)
3. Copy /scratch/$USER/tcheat/mech/out/* and the log to the local archive, nc_upgrade/mech_out/gc/.
   Outputs: mech_cases_{oper,small}.csv, mech_regions_{oper,small}.csv, mech_gc_result.txt, win_oper.npz, win_small.npz.
