#!/bin/bash
#SBATCH --job-name=gcoper --partition=gpu-a100 --account=innovation --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem-per-cpu=7800M --time=02:00:00 --output=/scratch/%u/tcheat/logs/gcoper_%j.log
# usage: sbatch [--time=..] gc_oper.slurm CASES(e.g. 24-25 or 0-60)
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONWARNINGS=ignore
cd $TCH/scripts
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 10 > $TCH/logs/gcoper_mem_$SLURM_JOB_ID.txt &
date; ~/venvs/gc/bin/python gc_run.py --model oper --variants CTL,NOTC,WIDE,NULLW --leads 24,36,48 --tag _v2 --cases $1; date
