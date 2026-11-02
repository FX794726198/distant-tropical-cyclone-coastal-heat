#!/bin/bash
#SBATCH --partition=compute-p1 --account=research-ceg-wm --nodes=1 --ntasks=48 --ntasks-per-node=48 --exclusive --mem=0 --time=10:00:00 --output=/scratch/%u/tcheat/logs/wrfone_%x_%j.log
# usage: sbatch --job-name=w016wide run_one.slurm /scratch/$USER/tcheat/wrf/case_016/wide
source "${WPS_ENV:-/scratch/$USER/wps_env.sh}" >/dev/null 2>&1
cd $1; date
if [ ! -s wrfinput_d01 ] || [ ! -s wrfinput_d02 ] || [ ! -s wrfbdy_d01 ]; then srun -n 48 ./real.exe > real.out 2>&1; tail -1 rsl.error.0000; fi
rm -f rsl.* wrfout_*
t0=$(date +%s); srun -n 48 ./wrf.exe > wrf.out 2>&1
echo "wrf.exe $(( $(date +%s)-t0 )) s: $(tail -1 rsl.error.0000)"; date
