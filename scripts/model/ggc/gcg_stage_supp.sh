#!/bin/bash
# Login node: stage the ERA5 initial states for the supplement cases (same gc_stage.py; era5 is a symlink to ggc/era5,
# so times already staged for the 107 main cases are reused).
# usage: nohup bash gcg_stage_supp.sh > /scratch/$USER/tcheat/logs/gcgs_stage.log 2>&1 &
export TCH=/scratch/$USER/tcheat/ggc_supp
cd $TCH/scripts
date; ~/venvs/gc/bin/python gc_stage.py --workers 8; date
ls $TCH/era5/ | wc -l; du -sh $TCH/era5/
