#!/bin/bash
#SBATCH --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3996M --time=02:00:00 --job-name=wrfnc_an --output=/scratch/%u/tcheat/logs/wrfnc_analyse_%j.log
# Analysis of the multi-coast WRF runs (wrfnc_analyse.py; about 1 s per hourly frame, about 45 min for all 35 runs).
# usage: sbatch --export=ALL,PY=/path/to/python wrfnc_analyse.slurm [CASES...]     (no cases = all cases of wrfnc_cases.csv)
PY=${PY:?set PY to a python with numpy, pandas and netCDF4}
cd /scratch/$USER/tcheat/wrfnc/scripts; date
TCH=/scratch/$USER/tcheat $PY wrfnc_analyse.py "$@"
date
