#!/bin/bash
#SBATCH --job-name=revsmall --partition=gpu-v100 --account=innovation --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem-per-cpu=5000M --time=06:00:00 --output=/scratch/%u/tcheat/logs/rev_small_%j.log
# rev13 split: the 1-degree part of rev_gc.slurm (job 1142557), same commands and output paths.
# While it runs, rev/small_split.claimed makes rev_run.py / rev_steps.py skip the small model in any other job;
# if it fails, the claim is removed so 1142557 does the small model itself as originally planned.
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONWARNINGS=ignore REV_SMALL_OWNER=1
cd $TCH/rev
if squeue -h -j 1142557 -t R 2>/dev/null | grep -q .; then echo "1142557 is already running; leaving the small model to it"; exit 0; fi
C=$TCH/rev/small_split.claimed
echo "claimed by job $SLURM_JOB_ID at $(date)" > $C
trap 'rc=$?; if [ $rc -ne 0 ]; then rm -f $C; echo "failed (rc=$rc), claim removed"; fi' EXIT
trap 'exit 143' TERM
nvidia-smi --query-gpu=name,memory.total --format=csv
set -e
date; ~/venvs/gc/bin/python rev_run.py --model small --variants CTL,WIDEL,W1000 --tag _rev
date; echo "=== rev_steps small ==="; ~/venvs/gc/bin/python rev_steps.py --model small
echo "finished at $(date)" >> $C
date
