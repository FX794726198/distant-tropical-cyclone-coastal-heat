"""ERA5 3-hourly pressure + single levels (GRIB) for the TC dry-heat WRF cases.
Request lists copied from /scratch/$USER/pantanal_wrf/scripts/get_era5.py (Vtable.ECMWF set).
Usage: python get_era5_wrf.py {pl|sl} CASE"""
import os, sys
import pandas as pd
import cdsapi

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
AREA = [42, 95, 0, 150]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
LEVELS = ["1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100", "125", "150", "175",
          "200", "225", "250", "300", "350", "400", "450", "500", "550", "600", "650", "700",
          "750", "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000"]
PL_VARS = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
           "specific_humidity", "relative_humidity"]
SL_VARS = ["surface_pressure", "mean_sea_level_pressure", "2m_temperature", "2m_dewpoint_temperature",
           "10m_u_component_of_wind", "10m_v_component_of_wind", "skin_temperature",
           "sea_surface_temperature", "sea_ice_cover", "land_sea_mask", "geopotential", "snow_depth",
           "soil_temperature_level_1", "soil_temperature_level_2", "soil_temperature_level_3",
           "soil_temperature_level_4", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
           "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4"]
DATES = {6: ('2016-09-25', '2016-09-28'), 16: ('2018-09-13', '2018-09-16'),
         24: ('2019-08-07', '2019-08-10'), 44: ('2021-09-10', '2021-09-13')}

kind, case = sys.argv[1], int(sys.argv[2])
d = pd.date_range(*DATES[case])
out = os.path.join(TCH, 'wrf', 'era5_grib', f'case_{case:03d}'); os.makedirs(out, exist_ok=True)
target = os.path.join(out, f'era5_{kind}.grib')
if os.path.exists(target) and os.path.getsize(target) > 0:
    print(target, 'exists'); sys.exit()
req = {"product_type": ["reanalysis"], "year": [str(d[0].year)], "month": [f"{d[0].month:02d}"],
       "day": [f"{x.day:02d}" for x in d], "time": TIMES, "area": AREA,
       "data_format": "grib", "download_format": "unarchived"}
if kind == 'pl':
    req.update(variable=PL_VARS, pressure_level=LEVELS); ds = "reanalysis-era5-pressure-levels"
else:
    req.update(variable=SL_VARS); ds = "reanalysis-era5-single-levels"
cdsapi.Client(quiet=True, progress=False).retrieve(ds, req, target + '.part')
os.replace(target + '.part', target)
print(target, os.path.getsize(target) // 2**20, 'MB')
