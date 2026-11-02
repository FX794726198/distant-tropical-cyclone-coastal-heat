#!/bin/bash
#SBATCH --job-name=revgc --partition=gpu-a100 --account=innovation --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem-per-cpu=7800M --time=08:00:00 --output=/scratch/%u/tcheat/logs/rev_gc_%j.log
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONWARNINGS=ignore
cd $TCH/rev
nvidia-smi --query-gpu=name,memory.total --format=csv
set -e
date; ~/venvs/gc/bin/python rev_run.py --model small --variants CTL,WIDEL,W1000 --tag _rev
date; ~/venvs/gc/bin/python rev_run.py --model oper --variants CTL,WIDEL,W1000 --leads 24,36,48 --tag _rev
date; echo "=== rev_steps small ==="; ~/venvs/gc/bin/python rev_steps.py --model small
date; echo "=== rev_steps oper ==="; ~/venvs/gc/bin/python rev_steps.py --model oper
date
