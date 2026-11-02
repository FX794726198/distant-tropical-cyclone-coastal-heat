#!/bin/bash
#SBATCH --job-name=gls2split --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3996M --time=03:00:00 --output=/scratch/%u/tcheat/logs/rev_gls2_%x_%j.log
# usage: sbatch --job-name=gls2_odd_test_even --export=ALL,Y=even,FROM=odd rev_gls2.slurm   (regions of FROM half, tested on years Y)
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
O=$TCH/global/out_${FROM}_test_${Y}
mkdir -p $O; cd $TCH/scripts/global
date; YEARSET=$Y GLIN=$TCH/global/out_$FROM GLOUT=$O ~/venvs/gc/bin/python -W ignore gl_stage2_split.py --years 1980-2021 --nboot 1000; date
ls -la $O/
