#!/bin/bash
#SBATCH --job-name=wps --partition=compute-p1 --account=research-ceg-wm --ntasks=1 --cpus-per-task=2 --mem-per-cpu=3900M --time=02:00:00 --output=/scratch/%u/tcheat/logs/wps_%x_%j.log
# usage: sbatch --job-name=wps044 wps_case.sh 44 2021-09-10_00 2021-09-13_00
set -e
C=$(printf %03d $1); S=$2; E=$3
source "${WPS_ENV:-/scratch/$USER/wps_env.sh}" >/dev/null 2>&1
W=/scratch/$USER/tcheat/wrf; D=$W/case_$C/wps; mkdir -p $D $W/case_$C/metem_ctl; cd $D
sed -e "s/2019-08-07_00:00:00/${S}:00:00/g" -e "s/2019-08-10_00:00:00/${E}:00:00/g" \
    -e "s#opt_output_from_geogrid_path = './'#opt_output_from_geogrid_path = '$W/geo/'#" \
    -e "s#opt_output_from_metgrid_path = './'#opt_output_from_metgrid_path = '$W/case_$C/metem_ctl/'#" $W/geo/namelist.wps > namelist.wps
ln -sf $W/tables/Vtable.ECMWF Vtable
$WPS_DIR/link_grib.csh $W/era5_grib/case_$C/era5_pl.grib $W/era5_grib/case_$C/era5_sl.grib
$WPS_DIR/ungrib.exe > ungrib.out 2>&1; tail -2 ungrib.out
$WPS_DIR/metgrid.exe > metgrid.out 2>&1; tail -2 metgrid.out
ls $W/case_$C/metem_ctl | wc -l
