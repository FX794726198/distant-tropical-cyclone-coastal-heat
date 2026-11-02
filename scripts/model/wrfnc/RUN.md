# WRF storm-removal runs on six coasts: how to run on DelftBlue

Goal: repeat the Guangdong WRF experiment on four more coasts with the same model set-up, and with the same edit as
the GraphCast all-exposure-day test (WIDEL), on the same days, so WRF and GraphCast can be compared case by case.

- Coasts:
  - W17 Mexico Pacific coast, storms SE;
  - W3 northern Mozambique, storms E;
  - C16 US Southeast (a cooling coast), storms SW;
  - W22 US Northeast (mid-latitude), storms S;
  - W5 south-east India is optional, only if the queue allows.
- Cases: 3 per coast (see `inputs/wrfnc_cases.csv` and `select_log.txt`), plus Guangdong Megi (case 6) and Mangkhut
  (case 16), rerun with WIDEL. Their pilot control runs are reused.
- Runs per case: `ctl` and `widel`. A `nullw` run (the same edit at a storm-free sea point) is made for one case per coast.
- Physics and vertical levels are the pilot's (`paper/data/wrf/namelists`); only the domains change.

Root: `/scratch/$USER/tcheat/wrfnc` (new).
- Read-only use: `/scratch/$USER/tcheat/{wrf,scripts,inputs,venv_gc}`, plus `/scratch/$USER/pantanal_wrf/scripts/wps_env.sh` and
  `WPS_GEOG_extract` (sourced and read only).
- Never modify or delete anything under `pantanal_wrf`, `tcheat/global`, `ggc*`, `hydro_h1`, `pw_*` or `h1_*`.

## 0. Inventory (read only, report back)
- `squeue -u $USER`; `sshare -U -u $USER`; `sacctmgr -n show assoc user=$USER format=account,maxjobs,maxsubmit,grptres`
  (job-count caps of innovation and research-ceg-wm).
- `df -h /scratch`; `du -sh /scratch/$USER/tcheat/wrf`.
- Check these exist:
  - `/scratch/$USER/tcheat/wrf/tables/{Vtable.ECMWF,GEOGRID.TBL,METGRID.TBL}`;
  - `/scratch/$USER/tcheat/wrf/geo/geo_em.d0{1,2}.nc`;
  - `/scratch/$USER/tcheat/scripts/{tcfilter.py,symvortex.py}`;
  - `/scratch/$USER/tcheat/inputs/gd_stations.csv` (columns `lon`, `lat`);
  - `/scratch/$USER/tcheat/wrf/case_006/{metem_ctl,ctl}` (with `wrfout_d0*` files) and the same for `case_016`;
  - `~/.cdsapirc`: say only whether it exists, never print it;
  - `~/venvs/cds/bin/python -c "import cdsapi"`.
- `ls /scratch/$USER/WPS_GEOG_extract/WPS_GEOG | head`. The data should be global: the pilot used it for
  Guangdong and South America.
- How was the pilot's `nullw` met_em made? Run `ls /scratch/$USER/tcheat/wrf/case_006/` and grep the scripts for `nullw`.
  This is only for the record.

## 1. Set-up
```
R=/scratch/$USER/tcheat/wrfnc; mkdir -p $R/{scripts,inputs}
# copy this folder (the local archive, nc_upgrade/scripts/wrfnc/*) to $R/scripts/
# copy the local archive, nc_upgrade/wrfnc_inputs/* to $R/inputs/
cp /scratch/$USER/tcheat/inputs/gd_stations.csv $R/inputs/cells_GD.csv
cp /scratch/$USER/tcheat/scripts/{tcfilter.py,symvortex.py} $R/scripts/
md5sum $R/scripts/* $R/inputs/* > $R/MD5_setup.txt
```

## 2. geogrid: one job per coast, about 10–30 min each
```
cd $R/scripts; for c in W17 W3 C16 W22 W5; do sbatch --job-name=geo$c geo.slurm $c; done
```
- Check each `geo_<coast>/geo_em.d02.nc`. Its corner latitudes and longitudes should enclose
  `inputs/cells_<coast>.csv` with about 150 km to spare.
- Report the d01 and d02 corners. Use `ncdump -h` attributes `corner_lats` and `corner_lons`, or the min/max of
  `XLAT_M` and `XLONG_M`.

## 3. ERA5 + WPS + edited met_em: one job per case (CDS request inside the job)
- First try one case and time the CDS requests: `sbatch --job-name=prep393 prep_case.slurm 393 "widel nullw"`.
- If CDS is fast (under about 1 h per request), submit the rest:
```
for k in 398 397;  do sbatch --job-name=prep$k prep_case.slurm $k widel; done            # W17
sbatch --job-name=prep321 prep_case.slurm 321 "widel nullw"; for k in 319 312; do sbatch --job-name=prep$k prep_case.slurm $k widel; done   # W3
sbatch --job-name=prep521 prep_case.slurm 521 "widel nullw"; for k in 522 517; do sbatch --job-name=prep$k prep_case.slurm $k widel; done   # C16
sbatch --job-name=prep419 prep_case.slurm 419 "widel nullw"; for k in 417 426; do sbatch --job-name=prep$k prep_case.slurm $k widel; done   # W22
for k in 6 16; do sbatch --job-name=prep$k prep_case.slurm $k widel; done                # Guangdong: no download, reuses metem_ctl
```
- If `filter_nullw.log` says "no null point" at the first time, run nullw for the next case of that coast instead:
  `bash prep_case.sh <next> "widel nullw"`.
- Check each `case_XXX/filter_widel.log`:
  - every d01 time should be edited;
  - the deficit should fall to about 0–2 hPa;
  - vt just outside the disk should be reduced.
  Report the first line per case.
- If CDS is slow (queued for hours), report it. The fallback is ARCO ERA5 from Google Cloud, written as WPS intermediate
  files; it is not written yet.

## 4. WRF runs: one wrf.exe per job, 48 cores, one node
```
for d in $R/case_*/widel $R/case_*/nullw $R/case_*/ctl; do
  [ -L $d ] && continue                                                  # Guangdong ctl = pilot run, skip
  k=$(basename $(dirname $d)); sbatch --account=innovation --job-name=w${k#case_}$(basename $d) run_one.slurm $d
done
```
- Alternate the accounts (innovation / research-ceg-wm) if innovation is capped at one or a few jobs.
- Order: first the ctl+widel pairs of the first case of each coast, then the rest, nullw last.
- Expected cost: about 5–8 h wall per run (d02 is up to 1.4 times the pilot's), 30 runs, about 9,000 core-h.
- After each run: `tail -1 rsl.error.0000` should say "SUCCESS COMPLETE WRF".
- Keep the `wrfout_d0*` files (hourly). Do not delete anything without asking.

## 5. Copy back (later)
`wrfnc_analyse.py` (to come) reads the d02 cell values and the d01 storm checks, and writes small CSV/npz files.
These go to the local archive `cee_ideas/tc_dry_heat/nc_upgrade/wrfnc_out/`. The `wrfout` files stay on DelftBlue.
