#!/bin/bash
# Supplement to the first set: select the cases of the four large regions with the quarter-of-cells rule.
# usage: bash ggc_supp_setup.sh SRC_DIR   (SRC_DIR = the ggc folder copied from the local archive, e.g. /scratch/$USER/tcheat/ggc_src)
# Creates /scratch/$USER/tcheat/ggc_supp beside ggc: scripts copied from ggc/scripts (already md5-checked), gc and era5
# symlinked to ggc's, selection written to ggc_supp/inputs.  Does NOT start the ERA5 staging (see gcg_stage_supp.sh).
set -eo pipefail
SRC=${1:?SRC_DIR}
ROOT=/scratch/$USER/tcheat; G=$ROOT/ggc; S=$ROOT/ggc_supp
[ -d $G/scripts ] && [ -d $G/era5 ] || { echo "main run root $G missing: stop"; exit 1; }
mkdir -p $S/scripts $S/inputs $S/logs
[ -e $S/gc ] || ln -s $G/gc $S/gc
[ -e $S/era5 ] || ln -s $G/era5 $S/era5
cp $G/scripts/gcg_common.py $G/scripts/gcg_run.py $G/scripts/gcg_eval.py $G/scripts/gc_build.py $G/scripts/gc_run.py \
   $G/scripts/gc_stage.py $G/scripts/tcfilter.py $G/scripts/symvortex.py $G/scripts/rev_build.py $S/scripts/
cp $SRC/gcg_select_supp.py $SRC/gcg_run_supp.slurm $SRC/gcg_stage_supp.sh $S/scripts/
cp $G/inputs/gd_stations.csv $S/inputs/
cd $S/scripts; md5sum gcg_select_supp.py gcg_common.py gcg_run.py gcg_eval.py gc_build.py rev_build.py symvortex.py tcfilter.py gcg_run_supp.slurm | tee $S/logs/ggc_supp_md5.txt
echo "--- main-run scripts unchanged?"; (cd $G/scripts && md5sum -c $G/logs/ggc_md5.txt)
echo "--- selection (supplement: W7,W11,C10,C14, cell fraction 0.25, cases from 200)"; date
TCH=$ROOT GLS=$ROOT/scripts/global ~/venvs/gc/bin/python -I gcg_select_supp.py $S/inputs 2>&1 | tee $S/logs/gcg_select_supp_run.log
md5sum $S/inputs/gc_cases.csv | tee $S/logs/gc_cases_md5.txt
echo "--- cases: $(($(wc -l < $S/inputs/gc_cases.csv) - 1)); free space on /scratch:"; df -h $ROOT | tail -1
echo "--- ERA5 times needed vs already staged (file names <time>.nc in $G/era5):"
~/venvs/gc/bin/python -I - <<'PY'
import pandas as pd, os
S='/scratch/$USER/tcheat/ggc_supp'; G='/scratch/$USER/tcheat/ggc'
c=pd.read_csv(f'{S}/inputs/gc_cases.csv'); need=set()
for V in pd.to_datetime(c.V):
    for h in (54,48,42,36,30,24,0): need.add((V-pd.Timedelta(hours=h)).strftime('%Y-%m-%dT%H'))
have={f[:-3] for f in os.listdir(f'{G}/era5') if f.endswith('.nc')}
print(f'{len(need)} times needed, {len(need & have)} already staged, {len(need - have)} to download (~{0.345*len(need-have):.0f} GB)')
PY
date
