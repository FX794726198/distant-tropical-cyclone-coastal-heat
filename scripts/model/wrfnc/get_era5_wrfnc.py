"""ERA5 3-hourly pressure + single levels (GRIB) for the multi-coast WRF cases (same request as the Guangdong runs,
paper/scripts/model/wrf/get_era5_wrf.py; only the area and dates change).
Usage: python get_era5_wrfnc.py {pl|sl} CASE "N W S E"     (area from: python wrfnc_setup.py area CASE)"""
import os, sys
import pandas as pd
import cdsapi

TCH = os.environ.get('TCH', os.path.expandvars('/scratch/$USER/tcheat'))
ROOT = os.environ.get('WRFNC', os.path.join(TCH, 'wrfnc'))
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

kind, case, area = sys.argv[1], int(sys.argv[2]), [float(x) for x in sys.argv[3].split()]
c = pd.read_csv(os.path.join(ROOT, 'inputs', 'wrfnc_cases.csv')).set_index('case').loc[case]
t0 = pd.Timestamp(c.start.replace('_', ' ') + ':00'); t1 = pd.Timestamp(c.end.replace('_', ' ') + ':00')
days = pd.date_range(t0.normalize(), t1.normalize(), freq='D')
out = os.path.join(ROOT, 'era5_grib', f'case_{case:03d}'); os.makedirs(out, exist_ok=True)
target = os.path.join(out, f'era5_{kind}.grib')
if os.path.exists(target) and os.path.getsize(target) > 0:
    print(target, 'exists'); sys.exit()
parts = []
for (y, m), g in pd.Series(days, index=days).groupby([days.year, days.month]):   # a run can span two months
    req = {"product_type": ["reanalysis"], "year": [str(y)], "month": [f"{m:02d}"],
           "day": [f"{x.day:02d}" for x in g], "time": TIMES, "area": area,
           "data_format": "grib", "download_format": "unarchived"}
    if kind == 'pl':
        req.update(variable=PL_VARS, pressure_level=LEVELS); ds = "reanalysis-era5-pressure-levels"
    else:
        req.update(variable=SL_VARS); ds = "reanalysis-era5-single-levels"
    p = target + f'.{y}{m:02d}.part'
    cdsapi.Client(quiet=True, progress=False).retrieve(ds, req, p); parts.append(p)
with open(target + '.part', 'wb') as f:
    for p in parts:
        f.write(open(p, 'rb').read())
for p in parts:
    os.remove(p)
os.replace(target + '.part', target)
print(target, os.path.getsize(target) // 2**20, 'MB')
