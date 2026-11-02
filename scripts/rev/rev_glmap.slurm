#!/bin/bash
#SBATCH --job-name=glsplit --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=8 --mem-per-cpu=3996M --time=06:00:00 --output=/scratch/%u/tcheat/logs/rev_glmap_%x_%j.log
# usage: sbatch --job-name=glmap_odd --export=ALL,Y=odd rev_glmap.slurm
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat PYTHONWARNINGS=ignore OMP_NUM_THREADS=8
mkdir -p $TCH/global/out_$Y; cd $TCH/scripts/global
date; YEARSET=$Y GLOUT=$TCH/global/out_$Y ~/venvs/gc/bin/python -W ignore gl_map_split.py --years 1980-2021 --nboot 1000 --placebo 0 --min-exp 50; date
ls -la $TCH/global/out_$Y/
