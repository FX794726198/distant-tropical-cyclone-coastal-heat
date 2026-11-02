#!/bin/bash
# usage: mkrun.sh CASE VARIANT START(YYYY-mm-dd_HH) END [RUN_HOURS]
set -e
C=$(printf %03d $1); V=$2; S=$3; E=$4; RH=${5:-72}
source "${WPS_ENV:-/scratch/$USER/wps_env.sh}" >/dev/null 2>&1
W=/scratch/$USER/tcheat/wrf; R=$W/case_$C/$V; mkdir -p $R; cd $R
for f in $WRF_DIR/run/*; do b=$(basename $f); case $b in namelist.input|*.exe|README*) ;; *) ln -sf $f $b;; esac; done
ln -sf $WRF_DIR/main/real.exe real.exe; ln -sf $WRF_DIR/main/wrf.exe wrf.exe
for f in $W/case_$C/metem_$V/met_em.*; do ln -sf $f .; done
sy=${S:0:4}; sm=${S:5:2}; sd=${S:8:2}; sh=${S:11:2}; ey=${E:0:4}; em=${E:5:2}; ed=${E:8:2}; eh=${E:11:2}
cat > namelist.input <<NML
 &time_control
 run_days = 0, run_hours = $RH, run_minutes = 0, run_seconds = 0,
 start_year = $sy, $sy, start_month = $sm, $sm, start_day = $sd, $sd, start_hour = $sh, $sh,
 end_year = $ey, $ey, end_month = $em, $em, end_day = $ed, $ed, end_hour = $eh, $eh,
 interval_seconds = 10800,
 input_from_file = .true., .true.,
 history_interval = 60, 60,
 frames_per_outfile = 24, 24,
 restart = .false., restart_interval = 100000,
 io_form_history = 2, io_form_restart = 2, io_form_input = 2, io_form_boundary = 2,
 debug_level = 0,
 /
 &domains
 time_step = 60, time_step_fract_num = 0, time_step_fract_den = 1,
 max_dom = 2,
 e_we = 391, 333,
 e_sn = 311, 225,
 e_vert = 45, 45,
 p_top_requested = 5000,
 num_metgrid_levels = 38,
 num_metgrid_soil_levels = 4,
 dx = 12000, 3000,
 dy = 12000, 3000,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, 76,
 j_parent_start = 1, 146,
 parent_grid_ratio = 1, 4,
 parent_time_step_ratio = 1, 4,
 feedback = 1,
 smooth_option = 0,
 /
 &physics
 mp_physics = 8, 8,
 ra_lw_physics = 4, 4,
 ra_sw_physics = 4, 4,
 radt = 12, 12,
 sf_sfclay_physics = 5, 5,
 sf_surface_physics = 4, 4,
 bl_pbl_physics = 5, 5,
 bldt = 0, 0,
 cu_physics = 3, 0,
 cudt = 0, 0,
 num_land_cat = 21,
 sf_urban_physics = 0, 0,
 sst_update = 0,
 /
 &fdda
 /
 &dynamics
 hybrid_opt = 2,
 w_damping = 1,
 diff_opt = 1, 1,
 km_opt = 4, 4,
 diff_6th_opt = 0, 0,
 damp_opt = 3,
 zdamp = 5000., 5000.,
 dampcoef = 0.2, 0.2,
 khdif = 0, 0,
 kvdif = 0, 0,
 non_hydrostatic = .true., .true.,
 moist_adv_opt = 1, 1,
 scalar_adv_opt = 1, 1,
 gwd_opt = 0,
 /
 &bdy_control
 spec_bdy_width = 5,
 specified = .true.
 /
 &grib2
 /
 &namelist_quilt
 nio_tasks_per_group = 0,
 nio_groups = 1,
 /
NML
echo $R
