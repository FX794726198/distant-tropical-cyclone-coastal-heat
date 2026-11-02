#!/bin/bash
# Multi-coast WRF: ERA5 -> ungrib/metgrid -> edited met_em for one case.  usage: bash prep_case.sh CASE "ctl widel [nullw]"
# Login node is fine for the ERA5 request; ungrib/metgrid/filter take ~10-20 min (or: sbatch prep_case.slurm CASE ...).
set -e
k=$1; VARS=${2:-"widel"}; C=$(printf %03d $k)
T=/scratch/$USER/tcheat; R=$T/wrfnc; S=$R/scripts; PY=$T/venv_gc/bin/python
export TCH=$T WRFNC=$R PYTHONWARNINGS=ignore
coast=$($PY -c "import pandas as pd; print(pd.read_csv('$R/inputs/wrfnc_cases.csv').set_index('case').loc[$k,'coast'])")
source "${WPS_ENV:-/scratch/$USER/wps_env.sh}" >/dev/null 2>&1
D=$R/case_$C; mkdir -p $D
if [ $coast = GD ]; then
  # Guangdong pilot cases: reuse the pilot's control met_em files (same ERA5, same domains)
  [ -e $D/metem_ctl ] || ln -s $T/wrf/case_$C/metem_ctl $D/metem_ctl
else
  AREA=$($PY $S/wrfnc_setup.py area $k)
  for kind in pl sl; do ~/venvs/cds/bin/python $S/get_era5_wrfnc.py $kind $k "$AREA"; done
  $PY $S/wrfnc_setup.py wpscase $k >/dev/null; cd $D/wps
  ln -sf $T/wrf/tables/Vtable.ECMWF Vtable
  $WPS_DIR/link_grib.csh $R/era5_grib/case_$C/era5_pl.grib $R/era5_grib/case_$C/era5_sl.grib
  [ $(ls $D/metem_ctl | wc -l) -gt 0 ] || { $WPS_DIR/ungrib.exe > ungrib.out 2>&1; tail -1 ungrib.out; $WPS_DIR/metgrid.exe > metgrid.out 2>&1; tail -1 metgrid.out; }
fi
echo "met_em files: $(ls $D/metem_ctl | wc -l)"
CELLS=$R/inputs/cells_$coast.csv
for v in $VARS; do
  [ -d $D/metem_$v ] && [ $(ls $D/metem_$v/met_em.* 2>/dev/null | wc -l) -eq $(ls $D/metem_ctl | wc -l) ] && continue
  $PY $S/metem_filter2.py --indir $D/metem_ctl --outdir $D/metem_$v --case $k --cells $CELLS \
      --tracks $R/inputs/wrfnc_tracks.csv --others $R/inputs/wrfnc_others.csv --mode $v > $D/filter_$v.log 2>&1
  echo "$v: $(grep -c 'deficit' $D/filter_$v.log) d01 times edited; first: $(grep -m1 deficit $D/filter_$v.log)"
done
for v in ctl $VARS; do
  if [ $coast = GD ] && [ $v = ctl ]; then [ -e $D/ctl ] || ln -s $T/wrf/case_$C/ctl $D/ctl; continue; fi   # pilot control run reused
  mkdir -p $D/$v; cd $D/$v
  for f in $WRF_DIR/run/*; do b=$(basename $f); case $b in namelist.input|*.exe|README*) ;; *) ln -sf $f $b;; esac; done
  ln -sf $WRF_DIR/main/real.exe real.exe; ln -sf $WRF_DIR/main/wrf.exe wrf.exe
  for f in $D/metem_$v/met_em.*; do ln -sf $f .; done
  $PY $S/wrfnc_setup.py input $k $v >/dev/null
done
echo "case $k ($coast) ready: $(ls -d $D/*/namelist.input | wc -l) runs"
