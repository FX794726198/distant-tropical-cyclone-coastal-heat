#!/bin/bash
#SBATCH --job-name=gcwide --partition=gpu-a100 --account=innovation --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 --mem-per-cpu=7800M --time=06:00:00 --output=/scratch/%u/tcheat/logs/gcwide_%j.log
module load 2025 python/3.11.9 2>/dev/null
export TCH=/scratch/$USER/tcheat XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 PYTHONWARNINGS=ignore
cd $TCH/scripts
date; ~/venvs/gc/bin/python gc_run.py --model small --variants CTL,WIDE,WIDEH,NULLW --tag _wide ; date
ls -la $TCH/out_small/
~/venvs/gc/bin/python gc_analyse.py --model small --tag _wide --main WIDE
