"""Multi-coast WRF runs: write namelist.wps (per coast) and namelist.input (per run) from domains.json and the case table.

Usage (on DelftBlue, $TCH=/scratch/$USER/tcheat, root $TCH/wrfnc):
  python wrfnc_setup.py wps COAST                  -> $ROOT/geo_COAST/namelist.wps (geogrid; dates filled per case later)
  python wrfnc_setup.py wpscase CASE               -> $ROOT/case_XXX/wps/namelist.wps (ungrib/metgrid for this case)
  python wrfnc_setup.py input CASE VARIANT         -> $ROOT/case_XXX/VARIANT/namelist.input
  python wrfnc_setup.py area CASE                  -> prints N/W/S/E of the case's d01 + 3 degrees (for the ERA5 request)
Physics, dynamics and vertical levels are the Guangdong pilot's (paper/data/wrf/namelists); only the domains change.
Guangdong (coast GD) reuses $TCH/wrf/geo and the pilot's d01/d02.
"""
import json, os, sys
import pandas as pd

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
ROOT = os.environ.get('WRFNC', os.path.join(TCH, 'wrfnc'))
GEOG = os.environ.get('WPS_GEOG', os.path.expandvars('/scratch/$USER/WPS_GEOG/'))
TBL = os.path.join(TCH, 'wrf', 'tables') + '/'
DOM = json.load(open(os.path.join(ROOT, 'inputs', 'domains.json')))
DOM['GD'] = dict(map_proj='mercator', ref_lat=21.0, ref_lon=122.5, truelat1=21.0, stand_lon=122.5,
                 e_we=[391, 333], e_sn=[311, 225], dx=[12000, 3000], i_parent_start=[1, 76], j_parent_start=[1, 146])
CASES = pd.read_csv(os.path.join(ROOT, 'inputs', 'wrfnc_cases.csv')).set_index('case')


def geo_dir(coast):
    return os.path.join(TCH, 'wrf', 'geo') if coast == 'GD' else os.path.join(ROOT, f'geo_{coast}')


def wps_text(d, start, end, geo_path, met_path):
    proj = f" map_proj = '{d['map_proj']}',\n ref_lat   = {d['ref_lat']},\n ref_lon   = {d['ref_lon']},\n truelat1  = {d['truelat1']},\n"
    if d['map_proj'] == 'lambert':
        proj += f" truelat2  = {d['truelat2']},\n"
    proj += f" stand_lon = {d['stand_lon']},\n"
    return f"""&share
 wrf_core = 'ARW',
 max_dom = 2,
 start_date = '{start}:00:00','{start}:00:00',
 end_date   = '{end}:00:00','{end}:00:00',
 interval_seconds = 10800,
 io_form_geogrid = 2,
 opt_output_from_geogrid_path = '{geo_path}/',
/
&geogrid
 parent_id         =   1,   1,
 parent_grid_ratio =   1,   4,
 i_parent_start    =   1, {d['i_parent_start'][1]:3d},
 j_parent_start    =   1, {d['j_parent_start'][1]:3d},
 e_we              = {d['e_we'][0]:3d}, {d['e_we'][1]:3d},
 e_sn              = {d['e_sn'][0]:3d}, {d['e_sn'][1]:3d},
 geog_data_res     = 'default','default',
 dx = 12000,
 dy = 12000,
{proj} geog_data_path = '{GEOG}',
 opt_geogrid_tbl_path = '{TBL}',
/
&ungrib
 out_format = 'WPS',
 prefix = 'FILE',
/
&metgrid
 fg_name = 'FILE',
 io_form_metgrid = 2,
 opt_output_from_metgrid_path = '{met_path}/',
 opt_metgrid_tbl_path = '{TBL}',
/
"""


def input_text(d, start, end, run_hours=72):
    sy, sm, sd, sh = start[:4], start[5:7], start[8:10], start[11:13]
    ey, em, ed, eh = end[:4], end[5:7], end[8:10], end[11:13]
    return f""" &time_control
 run_days = 0, run_hours = {run_hours}, run_minutes = 0, run_seconds = 0,
 start_year = {sy}, {sy}, start_month = {sm}, {sm}, start_day = {sd}, {sd}, start_hour = {sh}, {sh},
 end_year = {ey}, {ey}, end_month = {em}, {em}, end_day = {ed}, {ed}, end_hour = {eh}, {eh},
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
 e_we = {d['e_we'][0]}, {d['e_we'][1]},
 e_sn = {d['e_sn'][0]}, {d['e_sn'][1]},
 e_vert = 45, 45,
 p_top_requested = 5000,
 num_metgrid_levels = 38,
 num_metgrid_soil_levels = 4,
 dx = 12000, 3000,
 dy = 12000, 3000,
 grid_id = 1, 2,
 parent_id = 0, 1,
 i_parent_start = 1, {d['i_parent_start'][1]},
 j_parent_start = 1, {d['j_parent_start'][1]},
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
"""


def main():
    what = sys.argv[1]
    if what == 'wps':
        coast = sys.argv[2]; g = geo_dir(coast); os.makedirs(g, exist_ok=True)
        c = CASES[CASES.coast == coast].iloc[0]
        open(os.path.join(g, 'namelist.wps'), 'w').write(wps_text(DOM[coast], c.start, c.end, g, g))
        print(os.path.join(g, 'namelist.wps'))
    elif what == 'wpscase':
        k = int(sys.argv[2]); c = CASES.loc[k]; d = os.path.join(ROOT, f'case_{k:03d}')
        os.makedirs(os.path.join(d, 'wps'), exist_ok=True); os.makedirs(os.path.join(d, 'metem_ctl'), exist_ok=True)
        open(os.path.join(d, 'wps', 'namelist.wps'), 'w').write(
            wps_text(DOM[c.coast], c.start, c.end, geo_dir(c.coast), os.path.join(d, 'metem_ctl')))
        print(os.path.join(d, 'wps', 'namelist.wps'))
    elif what == 'input':
        k, v = int(sys.argv[2]), sys.argv[3]; c = CASES.loc[k]; d = os.path.join(ROOT, f'case_{k:03d}', v)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'namelist.input'), 'w').write(input_text(DOM[c.coast], c.start, c.end))
        print(os.path.join(d, 'namelist.input'))
    elif what == 'area':
        import netCDF4
        k = int(sys.argv[2]); c = CASES.loc[k]
        nc = netCDF4.Dataset(os.path.join(geo_dir(c.coast), 'geo_em.d01.nc'))
        la, lo = nc['XLAT_M'][0], nc['XLONG_M'][0]; nc.close()
        print(f'{min(90, float(la.max()) + 3):.1f} {float(lo.min()) - 3:.1f} {max(-90, float(la.min()) - 3):.1f} {float(lo.max()) + 3:.1f}')


if __name__ == '__main__':
    main()
