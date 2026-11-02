#!/bin/bash
#SBATCH --job-name=a3_ext --partition=compute-p1,compute-p2 --account=research-ceg-wm --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3996M --time=01:00:00 --output=/scratch/%u/tcheat/logs/a3_ext_%j.log
# Fig. 4 extension of A3 (season, severity, heat then storm, cities, events); same inputs as impact.slurm, read only.
module load 2025 python/3.11.9 2>/dev/null
export IMP=/scratch/$USER/tcheat/impact GL=/scratch/$USER/tcheat/global PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
cd $IMP/scripts
date; md5sum impact_ext.py ../gcg_regions.json ../all_oper_regions.csv
$IMP/venv/bin/python -I impact_ext.py; date
ls -la $IMP/out
