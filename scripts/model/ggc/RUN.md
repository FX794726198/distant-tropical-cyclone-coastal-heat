# Global storm-removal test (analysis A): how to run on DelftBlue

Root: /scratch/$USER/tcheat/ggc (new). Do not touch other directories under /scratch/$USER except reading
/scratch/$USER/tcheat/{global,scripts,gc}.

0. mkdir -p /scratch/$USER/tcheat/ggc/{scripts,inputs,era5,logs}; ln -s /scratch/$USER/tcheat/gc /scratch/$USER/tcheat/ggc/gc
   Copy to ggc/scripts: gcg_select.py, gcg_common.py, gcg_run.py, gcg_eval.py, gcg_stage.sh, gcg_run.slurm (this folder) and
   from /scratch/$USER/tcheat/scripts: gc_build.py, gc_run.py, gc_stage.py, tcfilter.py, symvortex.py; rev_build.py from
   /scratch/$USER/tcheat/rev/ (must equal the local copy, md5 167a6296626e52ccc7dde2c57c9a9fc7, shipped here as rev_build.py).
   cp /scratch/$USER/tcheat/inputs/gd_stations.csv ggc/inputs/   (placeholder read by gc_build at import)
1. Select the cases (login node, CPU, ~5 min, reads 42 years of 1-deg Tmax at the region cells):
   cd ggc/scripts; TCH=/scratch/$USER/tcheat GLS=/scratch/$USER/tcheat/scripts/global ~/venvs/gc/bin/python -I gcg_select.py /scratch/$USER/tcheat/ggc/inputs
   Check gcg_select_log.txt; freeze: md5sum ggc/inputs/gc_cases.csv.
2. Stage ERA5 (login node, hours; ~2.4 GB per case, ~350 GB for ~145 cases): nohup bash gcg_stage.sh > /scratch/$USER/tcheat/logs/gcg_stage.log 2>&1 &
   Check free space first (df -h /scratch). If it is short, stage and run in two halves: edit gc_stage.py's case range or
   temporarily move the second half of gc_cases.csv aside, run oper+small on the first half, delete ggc/era5, then stage the rest.
   Steps 0-2 in one go: bash ggc_setup.sh SRC_DIR  (SRC_DIR = this folder copied to DelftBlue, e.g. /scratch/$USER/tcheat/ggc_src).
3. GPU: sbatch gcg_run.slurm oper 0-64; sbatch gcg_run.slurm oper 65-200; then sbatch gcg_run.slurm small 0-200
4. Reading: ~/venvs/gc/bin/python -I gcg_eval.py /scratch/$USER/tcheat/ggc  -> ggc/gcg_result.txt
5. Copy back to the local archive, nc_upgrade/ggc_out/: inputs/*.csv, inputs/*.json, inputs/*.txt, out_*/gcg_*.csv,
   out_*/case_*.json, gcg_result.txt, logs. (case_*.nc stay on DelftBlue until the paper figures are made.)
