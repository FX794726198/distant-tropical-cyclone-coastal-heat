#!/bin/bash
#SBATCH --partition=compute-p1 --account=research-ceg-wm --ntasks=144 --cpus-per-task=1 --mem-per-cpu=3900M --time=12:00:00 --output=/scratch/%u/tcheat/logs/wrfcase_%x_%j.log
# usage: sbatch --job-name=wrf044 run_case.slurm 44 2021-09-10_00 2021-09-13_00 "ctl wide notc"
c=$1; C=$(printf %03d $c); S=$2; E=$3; RUNS=${4:-"ctl wide notc"}
T=/scratch/$USER/tcheat; W=$T/wrf/case_$C; F="$T/venv_gc/bin/python $T/scripts/metem_filter.py"
export TCH=$T PYTHONWARNINGS=ignore
cd $W
[ -d metem_notc ] && [ $(ls metem_notc | wc -l) -eq $(ls metem_ctl | wc -l) ] || (srun -n1 --exclusive $F --indir metem_ctl --outdir metem_notc --case $c --amp 0 > $T/logs/metem_filter_case$C.log 2>&1) &
[ -d metem_wide ] && [ $(ls metem_wide | wc -l) -eq $(ls metem_ctl | wc -l) ] || (srun -n1 --exclusive $F --indir metem_ctl --outdir metem_wide --case $c --amp 0 --wide > $T/logs/metem_filter_wide_case$C.log 2>&1) &
wait
source "${WPS_ENV:-/scratch/$USER/wps_env.sh}" >/dev/null 2>&1
for v in $RUNS; do $T/scripts/mkrun.sh $c $v $S $E >/dev/null; done
for v in $RUNS; do (cd $W/$v && srun -n 48 --exclusive ./real.exe > real.out 2>&1; tail -1 rsl.error.0000 > real.status) & done; wait
for v in $RUNS; do (cd $W/$v && t0=$(date +%s) && srun -n 48 --exclusive ./wrf.exe > wrf.out 2>&1; echo "$v wrf.exe $(( $(date +%s)-t0 )) s: $(tail -1 rsl.error.0000)") & done; wait
date
