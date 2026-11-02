#!/bin/bash
# Login node: stage the ERA5 initial states for the A2 cases (same gc_stage.py; era5 is a symlink to ggc/era5, so times staged
# for analysis A are reused).  gc_stage.py sometimes stops with a spurious KeyError when 8 threads index the store at once;
# it skips files that exist, so it is simply restarted (at most 12 attempts).
# usage: nohup bash gcg_stage_all.sh > /scratch/$USER/tcheat/logs/gcga_stage.log 2>&1 &
export TCH=/scratch/$USER/tcheat/ggc_all
cd $TCH/scripts
date
for i in $(seq 1 12); do
  echo "--- attempt $i"; ~/venvs/gc/bin/python gc_stage.py --workers 8 && { echo "stage exit 0 on attempt $i"; break; }
  sleep 30
done
date; ls $TCH/era5/ | wc -l; du -sh $TCH/era5/
