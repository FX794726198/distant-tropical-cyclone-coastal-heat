#!/bin/bash
#SBATCH --job-name=revopertest --partition=gpu-v100 --account=innovation --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem-per-cpu=5000M --time=01:00:00 --output=/scratch/%u/tcheat/logs/rev_oper_v100test_%j.log
# rev13: single-forecast feasibility test of the 0.25-degree model on gpu-v100 (no preallocation, so nvidia-smi shows real use)
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONWARNINGS=ignore
cd $TCH/rev
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > $TCH/logs/rev_oper_v100test_mem_$SLURM_JOB_ID.txt &
date; ~/venvs/gc/bin/python rev_oper_v100test.py; rc=$?; date
echo "exit $rc; nvidia-smi peak MiB: $(sort -n $TCH/logs/rev_oper_v100test_mem_$SLURM_JOB_ID.txt | tail -1)"
exit $rc
