#!/bin/bash
#SBATCH --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=2 --mem-per-cpu=3900M --time=06:00:00 --output=/scratch/%u/tcheat/logs/wrfnc_prep_%x_%j.log
# usage: sbatch --job-name=prep393 prep_case.slurm 393 "widel nullw"
bash /scratch/$USER/tcheat/wrfnc/scripts/prep_case.sh $1 "$2"
