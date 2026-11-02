#!/bin/bash
# Login node: stage the ERA5 0.25-deg initial states for the global storm-removal cases (gc_stage.py, WeatherBench2 bucket).
# usage: nohup bash gcg_stage.sh > /scratch/$USER/tcheat/logs/gcg_stage.log 2>&1 &
export TCH=/scratch/$USER/tcheat/ggc
cd $TCH/scripts
date; ~/venvs/gc/bin/python gc_stage.py --workers 8; date
ls $TCH/era5 | wc -l; du -sh $TCH/era5
