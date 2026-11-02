# Coastal orientation shapes warming from distant tropical cyclones

Code and minimal data for the manuscript *Coastal orientation shapes warming from distant tropical cyclones*
(Xiao Feng, Juan Du, Bo Chai, Luigi Lombardo; submitted). The repository holds every script used in the analysis and
the data needed to redraw all figures, rebuild the supplementary tables and check the numbers in the text.

## Quick start

```bash
pip install -r requirements.txt
./make_figures.sh            # writes fig/Fig1–Fig4 and fig/FigS1–FigS13 (PNG and PDF)
```

Run everything from the repository root. Rendered with the software listed below, all 17 figures are identical,
pixel for pixel, to those in `fig/`. Other versions of Matplotlib or other fonts move text slightly.

## Contents

| Folder | What it holds |
|---|---|
| `fig_code/` | One script per figure, named after the figure (`figcommon.py`: shared style; `ggc_common.py`: region inputs shared by several figures) |
| `fig/` | The figures as in the manuscript |
| `tables/` | Supplementary Tables S1–S12 as in the manuscript (Markdown) |
| `scripts/global/` | Global map: ERA5 and IBTrACS retrieval, exposure days and anomalies (`gl_map.py`), circulation analogues (`gl_stage2.py`), cool-cell check |
| `scripts/model/ggc/` | Storm removal in GraphCast on the coasts of the global map: case selection, runs on the cluster, readings, distance from the coast; `RUN.md` gives the run order |
| `scripts/model/graphcast/` | GraphCast set-up and the storm-removal code (`gc_build.py`, `tcfilter.py`, `symvortex.py`), first applied to Guangdong |
| `scripts/model/wrfnc/` | WRF runs on six coasts: case selection (`wrfnc_select.py`), namelists (`wrfnc_setup.py`), ERA5 retrieval, storm removal in the met_em files (`metem_filter2.py`), analysis (`wrfnc_analyse.py`), new lows (`wrfnc_newlows.py`), the check of the control runs against ERA5 and stations (`wrf_ctl_check.py`, `wrf_ctl_table.py`) and the coast maps of Fig. 2b (`wrfnc_coast_maps.py`); `RUN.md` gives the run order |
| `scripts/model/wrf/` | The convection-permitting pilot in Guangdong (WPS, storm removal, runs, analysis) |
| `scripts/model/mech/` | Coast orientation (`coast_normals.py`), the outer-wind indicator (`geometry.py`), GraphCast changes and storm-relative composites |
| `scripts/model/isd/` | Hourly station check with ISD-Lite: listing, selection, download, analysis |
| `scripts/impact/` | Population exposure (GHS-POP aggregation, `impact.py`, `impact_ext.py`) and the two drawing tables of Fig. 4 (`fig4_inputs.py`) |
| `scripts/forecast/` | Forecast test with WeatherBench 2 HRES and Pangu-Weather forecasts (`fc_test.py`, `fc_track.py`, `fc_example.py`), Niño-3.4 index |
| `scripts/stats/` | Joint resampling of storms shared by several regions (`boot_joint.py`) |
| `scripts/` (top level), `scripts/inputs/`, `scripts/rev/`, `scripts/export/` | Guangdong station analysis, retrieval of station data, table scripts, checks of the storm removal and the split-sample check, exports of model results |
| `data/` | Minimal data (below) |

### Data

| Folder | What it holds |
|---|---|
| `data/global/`, `data/global_s2/`, `data/global_split/` | Global map: grid-point results, warm and cool regions, placebo test, circulation-analogue results, split-sample check |
| `data/ggc/` | GraphCast storm removal on the coasts of the map: cases, regions (`gcg_regions.json`), region and grid-point results for the 0.25° (`oper`) and 1° (`small`) models |
| `data/wrfnc/` | WRF runs on six coasts: per-run values (`wrfnc_runs.csv`, `wrfnc_hourly.csv`), effects (`wrfnc_effects.csv`), new lows, the thinned d02 maps at the case time (`wrfnc_maps.npz`), the coast maps drawn in Fig. 2b (`wrfnc_maps_coast.npz`) and the control-run check (`ctlcheck/`) |
| `data/wrfnc_inputs/` | Case list, storm tracks, region cells, coast normals and domains of the six-coast runs |
| `data/mech/` | Outer-wind indicator for regions and case days, coast normals, storm-relative composites |
| `data/isd/` | Hourly station check: stations, region results, binned and per-station results |
| `data/impact/` | Exposure results (`impact_*.csv`, `impact_result.txt`, `impact_ext_result.txt`), the 1° population grid (`pop_1deg.npz`, from GHS-POP R2023A) and the drawing tables of Fig. 4 |
| `data/forecast/` | Forecast test results for HRES and Pangu-Weather, by region, lead and track error; the example of Fig. 4i |
| `data/stats/` | Joint-resampling intervals (`boot_joint.csv`) |
| `data/obs/`, `data/db/`, `data/graphcast/`, `data/figpack/`, `data/revision/`, `data/wrf/` | Guangdong: station results, station and case lists, GraphCast and WRF pilot results, storm-removal diagnostics, WRF namelists |
| `data/inputs/` | IBTrACS subset (`ibx_06utc.npz`) and the ISD-Lite listing and station history used for the station selection (dated 1 November 2026, so that the selection can be repeated exactly) |

Not included, to keep the repository small: ERA5, IBTrACS, GHS-POP and the WeatherBench 2 forecasts (public, retrieved by
the scripts), the GraphCast and WRF model output (several TB), the station-day tables of GSOD, ISD and GHCN-Daily for
Guangdong and the coastal station-day table of the station check (rebuilt from the public archives by
`scripts/inputs/` and `scripts/model/isd/`).

## Figure and table sources

| Item | Script | Data |
|---|---|---|
| Fig. 1 | `fig_code/fig1_global_map.py` | `data/global/`, `data/global_s2/`, `data/isd/` |
| Fig. 2 | `fig_code/fig2_storm_removal.py` | `data/ggc/`, `data/wrfnc/`, `data/wrfnc_inputs/`, `data/graphcast/`, `data/obs/` |
| Fig. 3 | `fig_code/fig3_coastal_orientation.py` | `data/mech/`, `data/isd/`, `data/wrfnc/` |
| Fig. 4 | `fig_code/fig4_impact_forecast.py` | `data/impact/`, `data/forecast/` |
| Fig. S1 | `fig_code/figS01_storm_direction.py` | `data/global/`, `data/global_s2/` |
| Fig. S2 | `fig_code/figS02_split_sample.py` | `data/global_split/`, `data/global/` |
| Fig. S3 | `fig_code/figS03_guangdong_stations.py` | `data/obs/` |
| Figs S4, S5 | `fig_code/figS04_graphcast_regions.py`, `figS05_graphcast_cells.py` | `data/ggc/` |
| Figs S6–S9 | `fig_code/figS06_guangdong_graphcast.py`, `figS07_S09_graphcast_checks.py`, `figS08_graphcast_circulation.py` | `data/figpack/`, `data/graphcast/`, `data/revision/`, `data/db/` |
| Fig. S10 | `fig_code/figS10_storm_edit.py` | `data/revision/` |
| Fig. S11 | `fig_code/figS11_guangdong_wrf.py` | `data/wrf/`, `data/figpack/`, `data/obs/`, `data/db/` |
| Fig. S12 | `fig_code/figS12_stations.py` | `data/isd/` |
| Fig. S13 | `fig_code/figS13_outer_winds.py` | `data/mech/` |
| Tables S1, S2 | `python -I scripts/92_global_tables.py data/global data/global_s2 OUT 6 7` (writes `tabS6.md`, `tabS7.md`) | `data/global/`, `data/global_s2/` |
| Table S3 | `python -I scripts/rev/tabS8_split.py data/global_split OUT` (`tabS8.md`) | `data/global_split/` |
| Tables S4, S8, S9 | `python -I scripts/91_tables.py data/obs data/figpack data/db OUT` (`tabS1.md`, `tabS3.md`, `tabS4.md`); for S8, copy `tabS3.md` to `tabS3_v01cols.md` and run `python -I scripts/rev/tabS3_addsf.py OUT data/revision/figure_inputs` | `data/obs/`, `data/figpack/`, `data/db/`, `data/revision/` |
| Table S5 | `python -I scripts/rev/tabS2_hourly.py data/obs OUT` (`tabS2.md`) | `data/obs/` |
| Tables S6, S7 | `TABOUT=OUT python -I scripts/global/93_ggc_tables.py` (`tabS9.md`, `tabS10.md`) | `data/ggc/` |
| Table S10 | copy `data/wrf/tabS5_base.md` to OUT, then `python -I scripts/rev/tabS5_wrf.py OUT data/wrf` (`tabS5.md`) | `data/wrf/` |
| Table S11 | assembled from `data/wrfnc_inputs/wrfnc_cases.csv` (map anomaly), `data/ggc/all_oper_cases.csv` (GraphCast, `d_free`; null rows `d_null`), `data/graphcast/gcx_case.csv` (Guangdong), `data/wrfnc/wrfnc_effects.csv` (WRF `Tmax`), `data/wrfnc/wrfnc_runs.csv` (storm depth, `deficit_Vm24`, `deficit_V`) and `data/wrfnc/wrfnc_newlows.csv` (deepest new low within 1,500 km of the region between V − 12 h and V + 6 h); notes written by hand | `data/wrfnc/`, `data/wrfnc_inputs/`, `data/ggc/` |
| Table S12 | `python -I scripts/model/wrfnc/wrf_ctl_table.py data/wrfnc/ctlcheck OUT/tabS12.md` | `data/wrfnc/ctlcheck/` |

The table scripts keep the numbers of an earlier draft in their output names (`tabS1.md` … `tabS10.md`). Regenerated
from this repository, the numbers of Tables S1–S10 and S12 match the manuscript exactly; some column headings were
reworded in the manuscript.

## Run order of the analysis

The heavy steps ran on the DelftBlue cluster (TU Delft). Set `TCH` to a working directory (default
`/scratch/$USER/tcheat`); the Slurm files show the resources used, and account and partition names must be changed for
other clusters. `WPS_ENV` and `WPS_GEOG` point to the WRF/WPS environment script and the WPS static data.

1. **Global map (Fig. 1, Tables S1–S3).** `scripts/global/gl_get.py`, `gl_get2.py` (ERA5 from WeatherBench 2, IBTrACS),
   `gl_map.py`, `gl_stage2.py` (circulation analogues), `gl_cool_other.py`; split check with `scripts/rev/gl_split_patch.py`
   (`YEARSET=odd|even`) and `gl_split_compare.py`.
2. **Storm removal in GraphCast on the coasts (Fig. 2a,d, Figs S4, S5, Tables S6, S7).** `scripts/model/ggc/RUN.md`:
   selection (`gcg_select_all.py`), ERA5 staging, runs (`gcg_run_all.slurm`), readings (`gcg_eval_all.py`), grid points by
   distance from the coast (`ggc_cells.py`). The storm removal is in `scripts/model/graphcast/` and `scripts/rev/rev_build.py`.
3. **WRF on six coasts (Fig. 2b–d, Fig. 3f, Tables S11, S12).** `scripts/model/wrfnc/RUN.md`: `wrfnc_select.py`
   (cases chosen from the observed map anomaly only), `geo.slurm`, `prep_case.slurm` (ERA5, WPS and the edited met_em
   files), `run_one.slurm`, `wrfnc_analyse.slurm`, `wrfnc_newlows.slurm`; then `wrf_ctl_check.py` (ARCO-ERA5 and ISD-Lite,
   any machine with internet access) and `wrf_ctl_table.py`.
4. **Outer winds and coast orientation (Fig. 3, Fig. S13).** `scripts/model/mech/RUN.md`: `coast_normals.py` (Natural
   Earth 1:50 million land), `mech_gc.py`, `mech_win.py`, `geometry.py`, `mech_explore.py`.
5. **Hourly stations (Fig. 1e, Fig. 3d,e, Fig. S12).** `scripts/model/isd/`: `isd_select.py`, `isd_fetch.py`,
   `isd_analyze.py` (about 15 minutes and 0.5 GB of downloads).
6. **People exposed (Fig. 4a–e).** `scripts/impact/RUN.md`: `pop_get.sh`, `pop_agg.py`, `impact.py`, `impact_ext.py`,
   then `python -I scripts/impact/fig4_inputs.py data/impact`.
7. **Forecasts (Fig. 4f–i).** `scripts/forecast/fc_test.py hres|pangu …`, `fc_track.py`, `fc_example.py`.
8. **Joint resampling (Supplementary S3).** `python -I scripts/stats/boot_joint.py`.
9. **Guangdong (Figs S3, S6–S11, Tables S4, S5, S8–S10).** `scripts/inputs/` (GSOD, ISD, GHCN-Daily, ERA5),
   `scripts/12_isd_hourly.py`, `scripts/10_obs_gd_v2.py`, `scripts/model/graphcast/` (`gc_stage.py`, `gc_build.py`,
   `gc_run.py`, `gc_analyse.py`), `scripts/model/wrf/`, the checks in `scripts/rev/` and the exports in `scripts/export/`.

The selection scripts record in their docstrings the rules by which the model cases were chosen; the WRF cases use only the
observed map anomaly. The link between the outer winds and the sign of the effect was found while examining the outputs
of step 4. Internal labels in the scripts: analysis A is the first set of GraphCast days (the hottest days of each region),
A2 the second set (all usable days), A3 the impact analysis and A4 the outer-wind analysis. Comments and printed
summaries of some scripts were edited for release; the computations are those of the runs, so an MD5 sum printed in a
stored output can differ from that of the released script.

## Notes on precision

Numbers in the text and tables come from the full-precision tables (`.csv`). Map files used only for drawing store
fields as 8-bit integers with a scale (`gcx_maps.npz`, `fp_maps.npz`, `wrx_maps.npz`, `wrx_xsec.npz`,
`edit_maps_q.npz`, `steps_*_maps_q.npz`; −128 marks missing values) or at 0.1 K (`wrfnc_maps_coast.npz`). The WRF grid
is nearly regular, so the linear interpolation in `wrfnc_coast_maps.py` depends on the triangulation: rerun with another
SciPy version, it can change single grid values of Fig. 2b, while the coast means stay the same.

## Software

Figures and tables: Python 3.13 with numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, xarray 2026.9.0, Matplotlib 3.11.2,
cartopy 0.26.0 and shapely 2.2.0, with Arial or, where Arial is not installed, Liberation Sans (the figures in `fig/` use
Liberation Sans). Cluster steps: Python 3.11 with numpy, pandas, xarray, netCDF4 and JAX; GraphCast
(https://github.com/google-deepmind/graphcast) on one A100 GPU; WRF and WPS 4.6.1.

Public data: ERA5 (Copernicus Climate Data Store; WeatherBench 2 and ARCO-ERA5 on Google Cloud), IBTrACS v04r01,
ISD, ISD-Lite, GSOD and GHCN-Daily (NOAA NCEI), GHS-POP R2023A (European Commission, Joint Research Centre), HRES and
Pangu-Weather forecasts (WeatherBench 2), Natural Earth.

## Licence

Code: MIT licence (`LICENSE`). Data and figures in `data/`, `fig/` and `tables/`: Creative Commons Attribution 4.0
International (`LICENSE-DATA`). Data derived from third-party sources keep the terms of those sources.

## Citation

Please cite the article once published. The repository will be archived on Zenodo with a DOI on acceptance.
