#!/bin/bash
#SBATCH --job-name=revdiag --partition=compute-p1 --account=research-ceg-wm --array=0-3 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=3996M --time=08:00:00 --output=/scratch/%u/tcheat/logs/rev_diag_%A_%a.log
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat PYTHONWARNINGS=ignore OMP_NUM_THREADS=2
cd $TCH/rev
date; TASK=$SLURM_ARRAY_TASK_ID NTASK=4 ~/venvs/gc/bin/python rev_diag.py; date
