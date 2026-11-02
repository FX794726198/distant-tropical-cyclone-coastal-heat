#!/bin/bash
#SBATCH --job-name=gcg --partition=gpu-a100 --account=innovation --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem-per-cpu=7800M --time=08:00:00 --output=/scratch/%u/tcheat/logs/gcg_%j.log
# Global storm-removal test (analysis A).  usage: sbatch gcg_run.slurm MODEL(oper|small) CASES(e.g. 0-64)
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat/ggc XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONWARNINGS=ignore
cd $TCH/scripts
nvidia-smi --query-gpu=name,memory.total --format=csv
date; md5sum gcg_run.py gcg_common.py gc_build.py rev_build.py symvortex.py tcfilter.py
~/venvs/gc/bin/python gcg_run.py --model $1 --cases $2; date
ls $TCH/out_$1 | wc -l
