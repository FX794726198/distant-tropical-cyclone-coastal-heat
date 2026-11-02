#!/bin/bash
#SBATCH --job-name=a4_mech --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=2 --mem-per-cpu=7900M --time=01:30:00 --output=/scratch/%u/tcheat/logs/a4_mech_%j.log
# A4 multi-coast mechanism (exploratory): reads the A2 case files READ-ONLY.
module load 2025 python/3.11.9 2>/dev/null
export MECH=/scratch/$USER/tcheat/mech GA=/scratch/$USER/tcheat/ggc_all PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
PY=~/venvs/gc/bin/python
cd $MECH/scripts; mkdir -p $MECH/out
date; md5sum mech_gc.py mech_win.py ../inputs/*.csv
$PY -I mech_gc.py $GA/out_oper $GA/out_small $MECH/inputs/all_oper_regions.csv $MECH/inputs/coast_normals.csv $MECH/out $MECH/inputs/mech_era5.csv
$PY -I mech_win.py $GA/out_oper oper $MECH/out/win_oper.npz
$PY -I mech_win.py $GA/out_small small $MECH/out/win_small.npz
date; ls -la $MECH/out; md5sum $MECH/out/*
