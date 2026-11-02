#!/bin/bash
#SBATCH --job-name=a3_impact --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=4 --mem-per-cpu=3996M --time=02:00:00 --output=/scratch/%u/tcheat/logs/a3_impact_%j.log
# A3 impact (frozen design plan A3 (internal)): population to 1 degree (pop_agg.py, once), then impact.py.
module load 2025 python/3.11.9 2>/dev/null
export IMP=/scratch/$USER/tcheat/impact GL=/scratch/$USER/tcheat/global PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
cd $IMP/scripts
date; md5sum pop_agg.py impact.py ../gcg_regions.json ../all_oper_regions.csv
[ -s $IMP/pop/pop_1deg.npz ] || $IMP/venv/bin/python -I pop_agg.py
$IMP/venv/bin/python -I impact.py --nboot 1000; date
ls -la $IMP/out
