#!/bin/bash
# Global storm-removal test (analysis A): steps 0-2 of RUN.md on the DelftBlue login node.
# usage: bash ggc_setup.sh SRC_DIR     (SRC_DIR = the copied ggc folder, e.g. /scratch/$USER/tcheat/ggc_src)
# Creates /scratch/$USER/tcheat/ggc, copies the scripts, selects the cases, freezes gc_cases.csv (md5), starts ERA5 staging.
set -eo pipefail
SRC=${1:?SRC_DIR}
ROOT=/scratch/$USER/tcheat; G=$ROOT/ggc
mkdir -p $G/scripts $G/inputs $G/era5 $G/logs
[ -e $G/gc ] || ln -s $ROOT/gc $G/gc
cp $SRC/gcg_select.py $SRC/gcg_common.py $SRC/gcg_run.py $SRC/gcg_eval.py $SRC/gcg_stage.sh $SRC/gcg_run.slurm $G/scripts/
cp $ROOT/scripts/gc_build.py $ROOT/scripts/gc_run.py $ROOT/scripts/gc_stage.py $ROOT/scripts/tcfilter.py $ROOT/scripts/symvortex.py $G/scripts/
# WIDEL patch: the pilot's copy lives in tcheat/rev (not tcheat/scripts); it must equal the local copy shipped in SRC_DIR
[ "$(md5sum < $ROOT/rev/rev_build.py)" = "$(md5sum < $SRC/rev_build.py)" ] || { echo 'rev_build.py differs between tcheat/rev and T7: stop'; exit 1; }
cp $ROOT/rev/rev_build.py $G/scripts/
cp $ROOT/inputs/gd_stations.csv $G/inputs/
cd $G/scripts; md5sum gcg_select.py gcg_common.py gcg_run.py gcg_eval.py gcg_run.slurm gcg_stage.sh | tee $G/logs/ggc_md5.txt
echo "--- selection"; date
TCH=$ROOT GLS=$ROOT/scripts/global ~/venvs/gc/bin/python -I gcg_select.py $G/inputs 2>&1 | tee $G/logs/gcg_select_run.log
md5sum $G/inputs/gc_cases.csv | tee $G/logs/gc_cases_md5.txt
echo "--- cases: $(($(wc -l < $G/inputs/gc_cases.csv) - 1)); free space on /scratch:"; df -h $ROOT | tail -1
echo "--- staging (background)"
nohup bash gcg_stage.sh > $ROOT/logs/gcg_stage.log 2>&1 &
echo "staging started, pid $!"; date
