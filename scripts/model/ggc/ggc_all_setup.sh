#!/bin/bash
# Analysis A2 (design analysis plan A2 (internal, written before the runs)): select all usable exposure days 2016-2021, <= 2 per storm.
# usage: bash ggc_all_setup.sh SRC_DIR   (SRC_DIR = folder with the A2 scripts copied from the local archive, e.g. /scratch/$USER/tcheat/ggc_src)
# Creates /scratch/$USER/tcheat/ggc_all beside ggc: run scripts copied from ggc/scripts (md5-checked against analysis A),
# gc and era5 symlinked to ggc's, selection written to ggc_all/inputs.  Does NOT start the ERA5 staging (gcg_stage_all.sh).
set -eo pipefail
SRC=${1:?SRC_DIR}
ROOT=/scratch/$USER/tcheat; G=$ROOT/ggc; A=$ROOT/ggc_all
[ -d $G/scripts ] && [ -d $G/era5 ] || { echo "analysis A root $G missing: stop"; exit 1; }
[ -e $A/inputs/gc_cases.csv ] && { echo "$A/inputs/gc_cases.csv exists already: stop (selection is done once)"; exit 1; }
mkdir -p $A/scripts $A/inputs $A/logs
[ -e $A/gc ] || ln -s $G/gc $A/gc
[ -e $A/era5 ] || ln -s $G/era5 $A/era5
echo "--- analysis A scripts unchanged?"; (cd $G/scripts && md5sum -c $G/logs/ggc_md5.txt)
cp $G/scripts/gcg_common.py $G/scripts/gcg_run.py $G/scripts/gcg_eval.py $G/scripts/gc_build.py $G/scripts/gc_run.py \
   $G/scripts/gc_stage.py $G/scripts/tcfilter.py $G/scripts/symvortex.py $G/scripts/rev_build.py $A/scripts/
cp $SRC/gcg_select_all.py $SRC/gcg_eval_all.py $SRC/gcg_run_all.slurm $SRC/gcg_stage_all.sh $SRC/ggc_cells.py $A/scripts/
cp $G/inputs/gd_stations.csv $A/inputs/
cd $A/scripts; md5sum gcg_select_all.py gcg_eval_all.py gcg_run_all.slurm gcg_stage_all.sh ggc_cells.py gcg_common.py gcg_run.py gcg_eval.py \
   gc_build.py gc_stage.py rev_build.py symvortex.py tcfilter.py | tee $A/logs/ggc_all_md5.txt
echo "--- selection (A2: all usable days 2016-2021, <= 2 per storm, seed 20261031, cases from 300)"; date
TCH=$ROOT GLS=$ROOT/scripts/global ~/venvs/gc/bin/python -I gcg_select_all.py $A/inputs 2>&1 | tee $A/logs/gcg_select_all_run.log
md5sum $A/inputs/gc_cases.csv | tee $A/logs/gc_cases_md5.txt
echo "--- cases: $(($(wc -l < $A/inputs/gc_cases.csv) - 1)); free space on /scratch:"; df -h $ROOT | tail -1
echo "--- ERA5 times needed vs already staged in $G/era5 (files YYYYmmddHH.nc):"
~/venvs/gc/bin/python -I - <<'PY'
import pandas as pd, os
A='/scratch/$USER/tcheat/ggc_all'; G='/scratch/$USER/tcheat/ggc'
c=pd.read_csv(f'{A}/inputs/gc_cases.csv'); need=set()
for V in pd.to_datetime(c.V):
    for h in (54,48,42,36,30,24,0): need.add((V-pd.Timedelta(hours=h)).strftime('%Y%m%d%H'))
have={f[:-3] for f in os.listdir(f'{G}/era5') if f.endswith('.nc')}
print(f'{len(need)} times needed, {len(need & have)} already staged, {len(need - have)} to download (~{0.345*len(need-have):.0f} GB)')
PY
date
