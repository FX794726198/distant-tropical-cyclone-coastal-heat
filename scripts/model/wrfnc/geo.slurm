#!/bin/bash
#SBATCH --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=1 --mem-per-cpu=3900M --time=02:00:00 --output=/scratch/%u/tcheat/logs/wrfnc_geo_%x_%j.log
# geogrid for one coast.  usage: sbatch --job-name=geoW17 geo.slurm W17
set -e
T=/scratch/$USER/tcheat; R=$T/wrfnc; export TCH=$T WRFNC=$R
source "${WPS_ENV:-/scratch/$USER/wps_env.sh}" >/dev/null 2>&1
nl=$($T/venv_gc/bin/python $R/scripts/wrfnc_setup.py wps $1); cd $(dirname $nl)
$WPS_DIR/geogrid.exe > geogrid.out 2>&1; tail -2 geogrid.out; ls -la geo_em.d0*.nc
