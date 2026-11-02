#!/bin/bash
#SBATCH --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3996M --time=02:00:00 --job-name=wrfnc_nl --output=/scratch/%u/tcheat/logs/wrfnc_newlows_%j.log
# New lows in the edited multi-coast WRF runs (wrfnc_newlows.py; every 3rd hour of d01).
# usage: sbatch --export=ALL,PY=/path/to/python wrfnc_newlows.slurm [CASES...]     (no cases = all cases of wrfnc_cases.csv)
PY=${PY:?set PY to a python with numpy, pandas and netCDF4}
cd /scratch/$USER/tcheat/wrfnc/scripts; date
TCH=/scratch/$USER/tcheat $PY wrfnc_newlows.py "$@"
date
