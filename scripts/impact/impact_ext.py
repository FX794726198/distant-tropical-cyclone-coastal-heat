"""A3 extension for Fig. 4 of the NC manuscript v10 (exploratory, 2026-11-01): when, how severe, what follows, where.

Same definitions as impact.py (warm regions, f = 0.50, 2020 population, P90 of storm-season days): a storm-caused hot
day is an exposure day in the region's direction with A >= P90 and A - delta < P90, delta = 0.5 * max(ex_dir, 0).
Adds, without changing any definition:
  season    month of storm-caused hot days and of all storm-season hot days (P90), by group (southern China W7, other
            Northern Hemisphere, Southern Hemisphere)                                   -> impact_season.csv
  severity  share of storm-caused days among hot days at P90, P95, P99 and Tmax >= 35 C  -> impact_severity.csv
  sequence  heat then storm: after a storm-caused hot day, does the storm that was 300-1,300 km away (the nearest storm
            of that day, as in the exposure codes) come within 300 km of the grid point within 1-5 days, and after how
            many days?  Compared with all exposure days and with hot exposure days not caused by the storm.
                                                                                         -> impact_sequence.csv
  cities    storm-caused hot days per year at the 1-degree grid point of major coastal cities in the warm regions
                                                                                         -> impact_cities.csv
  events    the days of 2016-2022 with the most storm-caused hot person-days (for the forecast example panel)
                                                                                         -> impact_events.csv
  v2 (11-01 10:00Z): where beyond southern China and how it changes
  regions   per region: people, storm-caused hot person-days, days per person, share of local hot days, exposure days
                                                                                         -> impact_ext_regions.csv
  years     per region and year: person-days (2020 population), days per person, exposure days, hot days
                                                                                         -> impact_region_years.csv
  change    per region cell: storm-caused hot days per year in 1980-2000 and 2001-2021, linear trend, exposure change
                                                                                         -> impact_cells_change.csv
  timing    monthly distribution of storm-caused hot days by group in 1980-2000 and 2001-2021
                                                                                         -> impact_season_periods.csv
  v3 (11-01 10:30Z): how strong is the storm effect when it is there, and in the largest heat days
  risk      per region and pooled: chance of a hot day on exposure days with the storm, without it (A - delta),
            and on other storm-season days; risk ratio and storm share of hot exposure days, f = 0.34, 0.50, 0.61
                                                                                         -> impact_risk.csv
  bigdays   per region: days ranked by hot person-days in the region (storm season); storm-caused share of hot
            person-days, share of days with a storm in position, and share of days that would fall below the tier
            threshold without the storm, for the top 1%, 5%, 10% of days and all hot days  -> impact_bigdays.csv
            largest 30 days of southern China (W7), 1980-2021                             -> impact_bigdays_W7_top.csv
Inputs as impact.py, plus $GL/ibtracs.since1980.list.v04r01.csv.  Writes $IMP/out/impact_ext_result.txt.
Usage: python -I impact_ext.py      (env GL, IMP as for impact.py)
"""
import json, os, time
import numpy as np, pandas as pd

GL = os.environ.get('GL', '/scratch/$USER/tcheat/global')
IMP = os.environ.get('IMP', '/scratch/$USER/tcheat/impact')
OUT = os.path.join(IMP, 'out')
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
F_MAIN, NEAR, LAGS = 0.50, 300.0, 5
CITIES = [  # name, lat, lon (major coastal or near-coast cities in or near tropical-cyclone regions)
    ('Guangzhou', 23.13, 113.26), ('Shenzhen', 22.54, 114.06), ('Hong Kong', 22.32, 114.17), ('Dongguan', 23.02, 113.75),
    ('Foshan', 23.02, 113.12), ('Xiamen', 24.48, 118.09), ('Fuzhou', 26.07, 119.30), ('Shantou', 23.35, 116.68),
    ('Zhanjiang', 21.27, 110.36), ('Haikou', 20.04, 110.20), ('Nanning', 22.82, 108.37), ('Wenzhou', 28.00, 120.67),
    ('Shanghai', 31.23, 121.47), ('Hangzhou', 30.27, 120.15), ('Taipei', 25.03, 121.56), ('Kaohsiung', 22.63, 120.30),
    ('Manila', 14.60, 120.98), ('Hanoi', 21.03, 105.85), ('Haiphong', 20.86, 106.68), ('Da Nang', 16.05, 108.20),
    ('Ho Chi Minh City', 10.82, 106.63), ('Naha', 26.21, 127.68), ('Busan', 35.18, 129.08), ('Seoul', 37.57, 126.98),
    ('Tokyo', 35.68, 139.69), ('Osaka', 34.69, 135.50), ('Chennai', 13.08, 80.27), ('Visakhapatnam', 17.69, 83.22),
    ('Bhubaneswar', 20.30, 85.82), ('Kolkata', 22.57, 88.36), ('Dhaka', 23.81, 90.41), ('Chittagong', 22.36, 91.78),
    ('Yangon', 16.84, 96.17), ('Mumbai', 19.08, 72.88), ('Karachi', 24.86, 67.00), ('Colombo', 6.93, 79.86),
    ('Maputo', -25.97, 32.57), ('Beira', -19.83, 34.84), ('Nampula', -15.12, 39.27), ('Antananarivo', -18.88, 47.51),
    ('Toamasina', -18.15, 49.40), ('Dar es Salaam', -6.79, 39.21), ('Mombasa', -4.04, 39.67), ('Durban', -29.86, 31.02),
    ('Darwin', -12.46, 130.84), ('Brisbane', -27.47, 153.03), ('Townsville', -19.26, 146.82), ('Cairns', -16.92, 145.77),
    ('Perth', -31.95, 115.86), ('Port Moresby', -9.44, 147.18), ('Noumea', -22.27, 166.45), ('Suva', -18.14, 178.44),
    ('Mazatlan', 23.25, -106.41), ('Culiacan', 24.81, -107.39), ('Hermosillo', 29.07, -110.96), ('Guadalajara', 20.66, -103.35),
    ('Acapulco', 16.85, -99.88), ('Los Angeles', 34.05, -118.24), ('San Diego', 32.72, -117.16), ('Monterrey', 25.69, -100.32),
    ('Veracruz', 19.17, -96.13), ('Merida', 20.97, -89.62), ('Houston', 29.76, -95.37), ('New Orleans', 29.95, -90.07),
    ('Tampa', 27.95, -82.46), ('Miami', 25.76, -80.19), ('Havana', 23.11, -82.37), ('Santo Domingo', 18.49, -69.93),
    ('San Juan', 18.47, -66.11), ('Charleston', 32.78, -79.93), ('Washington', 38.91, -77.04), ('New York', 40.71, -74.01),
    ('Boston', 42.36, -71.06), ('Halifax', 44.65, -63.57), ('Guatemala City', 14.63, -90.51), ('San Salvador', 13.69, -89.22),
    ('Managua', 12.11, -86.24), ('Panama City', 8.98, -79.52), ('Caracas', 10.48, -66.90), ('Recife', -8.05, -34.88)]


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def harmonic_anomalies(X, days):                             # identical to gl_map.py / impact.py
    t = (days.year.values - 2000.5) / 10.0
    w = 2 * np.pi * days.dayofyear.values / 365.25
    D = np.stack([np.ones_like(t), t] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)], 1)
    B = np.linalg.lstsq(D, X, rcond=None)[0]
    return (X - D @ B).astype(np.float32)


def load_tracks(path):                                       # identical to gl_map.py
    df = pd.read_csv(path, skiprows=[1], low_memory=False,
                     usecols=['SID', 'ISO_TIME', 'LAT', 'LON', 'USA_WIND', 'WMO_WIND', 'NATURE', 'BASIN'])
    for c in ('LAT', 'LON', 'USA_WIND', 'WMO_WIND'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['time'] = pd.to_datetime(df.ISO_TIME)
    df['wind'] = df.USA_WIND.fillna(df.WMO_WIND)
    df = df[(df.time.dt.hour.isin([0, 6, 12, 18])) & (df.time.dt.minute == 0) & (df.NATURE.str.strip() == 'TS')
            & (df.wind >= 34)].dropna(subset=['LAT', 'LON'])
    df['day'] = (df.time + pd.to_timedelta(df.LON / 15.0, unit='h')).dt.normalize()
    return df[['SID', 'time', 'day', 'LAT', 'LON', 'wind', 'BASIN']].reset_index(drop=True)


def main():
    os.makedirs(OUT, exist_ok=True); T0 = time.time(); log = []

    def say(s=''):
        print(s, flush=True); log.append(str(s))

    Z = np.load(os.path.join(GL, 'out', 'gl_codes.npz'))
    code, ci, cj, seas_m = Z['code'], Z['ci'], Z['cj'], Z['seas_m']
    days = pd.DatetimeIndex(Z['days'])
    st = np.load(os.path.join(GL, 'static_1deg.npz')); lon, lat = st['lon'].astype(float), st['lat'].astype(float)
    clon, clat = ((lon[ci] + 180) % 360) - 180, lat[cj]
    CEL = pd.read_csv(os.path.join(GL, 'out', 'gl_cells.csv'))
    assert (CEL.i.values == ci).all() and (CEL.j.values == cj).all()
    key = {(int(round(clon[k])), int(round(clat[k]))): k for k in range(len(ci))}
    REG = json.load(open(os.path.join(IMP, 'gcg_regions.json')))
    rows = []
    for r, d in REG.items():
        if d['sign'] != 'warm':
            continue
        s = int(d['s']); cells = d['cells'] if isinstance(d['cells'], list) else json.loads(d['cells'])
        for lo_, la_ in cells:
            k = key[(int(round(((float(lo_) + 180) % 360) - 180)), int(round(float(la_))))]
            rows.append(dict(region=r, s=s, cell=k, lon=clon[k], lat=clat[k], ex_dir=CEL[f'ex_{SECT[s]}'].values[k]))
    RC = pd.DataFrame(rows)
    U = np.unique(RC.cell.values); pos = {c: n for n, c in enumerate(U)}
    say(f'impact_ext.py: {RC.region.nunique()} warm regions, {len(RC)} region cells, {len(U)} grid points, {len(days)} days')

    want = set(days.values.astype('datetime64[D]').tolist()); X = []; got = []
    for y in range(days[0].year, days[-1].year + 1):
        d = np.load(os.path.join(GL, f'era5_t2_{y}.npz'))
        dd = d['days'].astype('datetime64[D]'); m = np.array([v in want for v in dd.tolist()])
        X.append(d['tmax'][m][:, ci[U], cj[U]]); got.append(dd[m])
    X = np.concatenate(X); got = np.concatenate(got)
    assert len(got) == len(days) and (got == days.values.astype('datetime64[D]')).all()
    A = harmonic_anomalies(X, days); TC = X - 273.15
    mon = days.month.values; y0 = days[0].year; yidx = days.year.values - y0; ny = yidx.max() + 1
    SE = seas_m[mon - 1][:, U]
    P = {q: np.array([np.percentile(A[SE[:, n], n], q) for n in range(len(U))]) for q in (90, 95, 99)}
    PZ = np.load(os.path.join(IMP, 'pop', 'pop_1deg.npz')); ep = list(PZ['epochs'])
    p20 = PZ['pop'][ep.index(2020)][ci[U], cj[U]].astype(float)
    say(f'loaded in {time.time() - T0:.0f} s')

    # storm-caused flags per region cell [day]
    flag, expo_, hotexp = {}, {}, {}
    sev = {k: [0.0, 0.0] for k in ('P90', 'P95', 'P99', '35C')}
    seen = set()
    for _, row in RC.iterrows():
        n = pos[row.cell]; c = row.cell; sd = SE[:, n]; a_ = A[:, n]; t_ = TC[:, n]
        expo = sd & (code[:, c] == row.s)
        dl = F_MAIN * max(row.ex_dir if np.isfinite(row.ex_dir) else 0.0, 0.0)
        k = (row.region, c)
        flag[k] = expo & (a_ >= P[90][n]) & (a_ - dl < P[90][n]); expo_[k] = expo; hotexp[k] = expo & (a_ >= P[90][n])
        for nm, hot, cause in (('P90', a_ >= P[90][n], a_ - dl < P[90][n]), ('P95', a_ >= P[95][n], a_ - dl < P[95][n]),
                               ('P99', a_ >= P[99][n], a_ - dl < P[99][n]), ('35C', t_ >= 35.0, t_ - dl < 35.0)):
            sev[nm][0] += p20[n] * (expo & sd & hot & cause).sum()
            if c not in seen:                                   # denominator over distinct grid points, as impact.py
                sev[nm][1] += p20[n] * (sd & hot).sum()
        seen.add(c)
    Q = sum(p20[pos[c]] * f.sum() for (r, c), f in flag.items()) / ny
    say(f'check against impact.py: storm-caused hot person-days per year {Q / 1e9:.3f} bn (impact.py: 0.267)')

    # ---- severity
    S = pd.DataFrame([dict(threshold=k, storm_caused_bn_per_yr=v[0] / ny / 1e9, hot_bn_per_yr=v[1] / ny / 1e9,
                           share=v[0] / max(v[1], 1)) for k, v in sev.items()])
    S.to_csv(os.path.join(OUT, 'impact_severity.csv'), index=False, float_format='%.5g')
    say('\n== severity: share of hot person-days caused by distant storms'); say(S.round(4).to_string(index=False))

    # ---- season
    grp = lambda r, la: 'southern China (W7)' if r == 'W7' else ('Northern Hemisphere, other' if la > 0 else 'Southern Hemisphere')
    rows = []; seen = set()
    for (r, c), f in flag.items():
        n = pos[c]; g = grp(r, clat[c])
        hot = SE[:, n] & (A[:, n] >= P[90][n])
        for m in range(1, 13):
            mm = mon == m
            rows.append(dict(group=g, month=m, storm_caused=p20[n] * (f & mm).sum() / ny,
                             exposure=p20[n] * (expo_[(r, c)] & mm).sum() / ny,
                             hot_all=(p20[n] * (hot & mm).sum() / ny) if (g, c) not in seen else 0.0))
        seen.add((g, c))
    SS = pd.DataFrame(rows).groupby(['group', 'month'], as_index=False).sum()
    for g, x in SS.groupby('group'):
        SS.loc[x.index, 'storm_frac'] = x.storm_caused / max(x.storm_caused.sum(), 1)
        SS.loc[x.index, 'hot_frac'] = x.hot_all / max(x.hot_all.sum(), 1)
    SS['share_of_hot'] = SS.storm_caused / SS.hot_all.where(SS.hot_all > 0)
    SS.to_csv(os.path.join(OUT, 'impact_season.csv'), index=False, float_format='%.5g')
    say('\n== season (person-days per year; fractions within group)'); say(SS.round(3).to_string(index=False))

    # ---- sequence: heat then storm
    tr = load_tracks(os.path.join(GL, 'ibtracs.since1980.list.v04r01.csv'))
    TD = {d: (g.SID.values, g.LON.values, g.LAT.values) for d, g in tr.groupby('day')}
    di = {d: k for k, d in enumerate(days)}

    def follow(k, c):
        """(sid, lag of first approach within NEAR km by the same storm or 0, lag for any storm or 0)"""
        d0 = days[k]; t = TD.get(d0)
        if t is None:
            return None, 0, 0
        dist = hav(clon[c], clat[c], t[1], t[2]); sid = t[0][np.argmin(dist)]
        same = anyl = 0
        for L in range(1, LAGS + 1):
            u = TD.get(d0 + pd.Timedelta(days=L))
            if u is None:
                continue
            dd = hav(clon[c], clat[c], u[1], u[2])
            if anyl == 0 and (dd <= NEAR).any():
                anyl = L
            if same == 0 and (dd[u[0] == sid] <= NEAR).any():
                same = L
            if same and anyl:
                break
        return sid, same, anyl

    rec = []
    for (r, c), f in flag.items():
        n = pos[c]
        for kind, m in (('storm-caused hot', f), ('hot, not storm-caused', hotexp[(r, c)] & ~f), ('all exposure', expo_[(r, c)])):
            for k in np.flatnonzero(m):
                sid, same, anyl = follow(k, c)
                rec.append(dict(kind=kind, region=r, cell=c, day=days[k], sid=sid, w=p20[n], same=same, anyl=anyl))
    R = pd.DataFrame(rec)
    out = []
    for (kind, g), x in [((k, 'all'), v) for k, v in R.groupby('kind')] + [((k[0], k[1]), v) for k, v in R.groupby(['kind', 'region'])]:
        w = x.w.values; o = dict(kind=kind, region=g, n_celldays=len(x), persondays_per_yr=w.sum() / ny)
        for L in range(1, LAGS + 1):
            o[f'same_by_{L}d'] = (w * ((x.same > 0) & (x.same <= L))).sum() / max(w.sum(), 1)
            o[f'any_by_{L}d'] = (w * ((x.anyl > 0) & (x.anyl <= L))).sum() / max(w.sum(), 1)
        for L in range(1, LAGS + 1):
            o[f'same_lag_{L}d'] = (w * (x.same == L)).sum() / max(w.sum(), 1)
        out.append(o)
    SQ = pd.DataFrame(out); SQ.to_csv(os.path.join(OUT, 'impact_sequence.csv'), index=False, float_format='%.5g')
    say('\n== heat then storm: person-weighted share of days followed by the same storm (any storm) within 300 km')
    say(SQ[SQ.region == 'all'][['kind', 'n_celldays', 'persondays_per_yr'] + [f'same_by_{L}d' for L in (1, 2, 3, 5)]
                                + ['any_by_5d']].round(3).to_string(index=False))

    # ---- cities
    rows = []
    cl = pd.DataFrame(dict(cell=RC.cell, region=RC.region, lon=RC.lon, lat=RC.lat))
    for name, la, lo in CITIES:
        dd = hav(cl.lon.values, cl.lat.values, lo, la)
        if dd.min() > 80:                                        # the city's own 1-degree grid point must be a region cell
            continue
        c = int(cl.cell.values[np.argmin(dd)]); n = pos[c]
        regs = sorted(cl[cl.cell == c].region.unique())
        days_yr = sum(flag[(r, c)].sum() for r in regs) / ny
        hot_yr = (SE[:, n] & (A[:, n] >= P[90][n])).sum() / ny
        rows.append(dict(city=name, lat=la, lon=lo, regions='+'.join(regs), grid_lon=clon[c], grid_lat=clat[c],
                         storm_hot_days_per_yr=days_yr, hot_days_per_yr=hot_yr, share=days_yr / max(hot_yr, 1e-9),
                         grid_pop2020_M=p20[n] / 1e6))
    CI = pd.DataFrame(rows).sort_values('storm_hot_days_per_yr', ascending=False)
    CI.to_csv(os.path.join(OUT, 'impact_cities.csv'), index=False, float_format='%.4g')
    say('\n== cities (grid point of the city; storm-caused hot days per year)'); say(CI.round(2).to_string(index=False))

    # ---- events 2016-2022
    E = R[(R.kind == 'storm-caused hot') & (R.day.dt.year >= 2016) & (R.day.dt.year <= 2022)]
    EV = (E.groupby(['day', 'region', 'sid'], as_index=False).agg(persondays=('w', 'sum'), cells=('cell', 'nunique'))
          .sort_values('persondays', ascending=False).head(30))
    EV.to_csv(os.path.join(OUT, 'impact_events.csv'), index=False, float_format='%.5g')
    say('\n== largest storm-caused hot days 2016-2022'); say(EV.head(15).to_string(index=False))

    # ---- v2: regions, years, change, timing
    years = np.arange(y0, y0 + ny); early = years <= 2000
    yc = lambda m: np.bincount(yidx[m], minlength=ny).astype(float)
    RG, RY, CC = [], [], []
    for r_, g in RC.groupby('region', sort=False):
        cells_ = g.cell.values; w = np.array([p20[pos[c]] for c in cells_])
        st_ = np.stack([yc(flag[(r_, c)]) for c in cells_])            # [cell, year] storm-caused hot days
        ex_ = np.stack([yc(expo_[(r_, c)]) for c in cells_])
        ho_ = np.stack([yc(SE[:, pos[c]] & (A[:, pos[c]] >= P[90][pos[c]])) for c in cells_])
        pdy = (st_ * w[:, None]).sum(0)                               # person-days per year
        dpp = pdy / max(w.sum(), 1)                                   # days per person per year
        RG.append(dict(region=r_, lat=float(g.lat.mean()), lon=float(g.lon.mean()), cells=len(cells_), pop2020_M=w.sum() / 1e6,
                       Q_M_per_yr=pdy.mean() / 1e6, days_per_person_per_yr=dpp.mean(), area_days_per_yr=st_.mean(),
                       share_of_local_hot=(st_ * w[:, None]).sum() / max((ho_ * w[:, None]).sum(), 1),
                       exposure_days_per_yr=ex_.mean(), delta_K=F_MAIN * float(np.nanmax([np.nanmean(np.clip(g.ex_dir, 0, None)), 0])),
                       trend_dpp_per_decade=np.polyfit(years, dpp, 1)[0] * 10, dpp_1980_2000=dpp[early].mean(), dpp_2001_2021=dpp[~early].mean()))
        for k, yv in enumerate(years):
            RY.append(dict(region=r_, year=yv, persondays_M=pdy[k] / 1e6, days_per_person=dpp[k], exposure_days=ex_[:, k].mean(),
                           hot_days=ho_[:, k].mean(), area_days=st_[:, k].mean()))
        for n_, c in enumerate(cells_):
            CC.append(dict(region=r_, cell=int(c), lon=clon[c], lat=clat[c], pop2020=w[n_], days_1980_2000=st_[n_, early].mean(),
                           days_2001_2021=st_[n_, ~early].mean(), trend_per_decade=np.polyfit(years, st_[n_], 1)[0] * 10,
                           exp_1980_2000=ex_[n_, early].mean(), exp_2001_2021=ex_[n_, ~early].mean()))
    RG = pd.DataFrame(RG).sort_values('Q_M_per_yr', ascending=False)
    RG.to_csv(os.path.join(OUT, 'impact_ext_regions.csv'), index=False, float_format='%.5g')
    pd.DataFrame(RY).to_csv(os.path.join(OUT, 'impact_region_years.csv'), index=False, float_format='%.5g')
    pd.DataFrame(CC).to_csv(os.path.join(OUT, 'impact_cells_change.csv'), index=False, float_format='%.5g')
    say('\n== regions: people, person-days, days per person, share of local hot days, change'); say(RG.round(3).to_string(index=False))
    rows = []
    for (r, c), f in flag.items():
        g = grp(r, clat[c]); w = p20[pos[c]]
        for per, pm in (('1980-2000', days.year <= 2000), ('2001-2021', days.year > 2000)):
            for m in range(1, 13):
                rows.append(dict(group=g, period=per, month=m, storm_caused=w * (f & pm & (mon == m)).sum()))
    SP = pd.DataFrame(rows).groupby(['group', 'period', 'month'], as_index=False).sum()
    SP['frac'] = SP.storm_caused / SP.groupby(['group', 'period']).storm_caused.transform('sum').where(lambda v: v > 0)
    SP.to_csv(os.path.join(OUT, 'impact_season_periods.csv'), index=False, float_format='%.5g')
    # ---- v3: conditional risk on exposure days
    FV = (0.34, F_MAIN, 0.61); acc = {}
    for _, row in RC.iterrows():
        n = pos[row.cell]; c = row.cell; sd = SE[:, n]; a_ = A[:, n]; w = p20[n]
        ex = expo_[(row.region, c)]; hot = sd & (a_ >= P[90][n])
        exd = max(row.ex_dir if np.isfinite(row.ex_dir) else 0.0, 0.0)
        for kk in (row.region, 'all'):
            d = acc.setdefault(kk, dict(expo=0.0, hot_expo=0.0, nonexpo=0.0, hot_nonexpo=0.0, **{f'c{f:.2f}': 0.0 for f in FV}))
            d['expo'] += w * ex.sum(); d['hot_expo'] += w * (hot & ex).sum()
            d['nonexpo'] += w * (sd & ~ex).sum(); d['hot_nonexpo'] += w * (hot & ~ex).sum()
            for f in FV:
                d[f'c{f:.2f}'] += w * (hot & ex & (a_ - f * exd < P[90][n])).sum()
    rk = []
    for kk, d in acc.items():
        o = dict(region=kk, exposure_pd_M_per_yr=d['expo'] / ny / 1e6, p_hot_exposure=d['hot_expo'] / max(d['expo'], 1),
                 p_hot_other_days=d['hot_nonexpo'] / max(d['nonexpo'], 1), heated_hot_pd_M_per_yr=d['hot_expo'] / ny / 1e6)
        for f in FV:
            cz = d[f'c{f:.2f}']; p0 = (d['hot_expo'] - cz) / max(d['expo'], 1)
            o[f'p_hot_nostorm_f{f:.2f}'] = p0; o[f'RR_f{f:.2f}'] = o['p_hot_exposure'] / p0 if p0 > 0 else np.nan
            o[f'caused_share_f{f:.2f}'] = cz / max(d['hot_expo'], 1); o[f'caused_pd_M_per_yr_f{f:.2f}'] = cz / ny / 1e6
        rk.append(o)
    RK = pd.DataFrame(rk); RK = pd.concat([RK[RK.region == 'all'], RK[RK.region != 'all'].sort_values('exposure_pd_M_per_yr', ascending=False)])
    RK.to_csv(os.path.join(OUT, 'impact_risk.csv'), index=False, float_format='%.5g')
    say('\n== risk on exposure days (person-weighted): with the storm, without it (f), other storm-season days')
    say(RK[['region', 'exposure_pd_M_per_yr', 'p_hot_exposure', f'p_hot_nostorm_f{F_MAIN:.2f}', 'p_hot_other_days',
            f'RR_f{F_MAIN:.2f}', 'RR_f0.34', 'RR_f0.61', f'caused_share_f{F_MAIN:.2f}']].round(3).to_string(index=False))

    # ---- v3: storm share of the largest regional heat days
    bd = []; TIERS = (('top 1%', 0.01), ('top 5%', 0.05), ('top 10%', 0.10), ('all hot days', None))
    pool = {t: [0.0, 0.0, 0.0, 0.0, 0.0] for t, _ in TIERS}             # S, H, HE, days with exposure, days
    for r_, g in RC.groupby('region', sort=False):
        cs = g.cell.values; ns = [pos[c] for c in cs]; w = p20[ns]
        hot = np.stack([SE[:, n] & (A[:, n] >= P[90][n]) for n in ns], 1)  # [day, cell]
        st = np.stack([flag[(r_, c)] for c in cs], 1); ex = np.stack([expo_[(r_, c)] for c in cs], 1)
        H = hot @ w; S = st @ w; HE = (hot & ex) @ w; Ed = ex.any(1)
        idx = np.where(np.stack([SE[:, n] for n in ns], 1).any(1))[0]
        if H[idx].sum() <= 0:
            continue
        Hs = np.sort(H[idx])[::-1]
        for t, q in TIERS:
            if q is None:
                sel_ = idx[H[idx] > 0]; left = np.nan; thr = np.nan
            else:
                thr = Hs[max(int(round(q * len(idx))), 1) - 1]; thr = thr if thr > 0 else Hs[Hs > 0][-1]
                sel_ = idx[H[idx] >= thr]; left = float(((H - S)[sel_] < thr).mean())
            bd.append(dict(region=r_, pop2020_M=w.sum() / 1e6, cells=len(cs), tier=t, n_days=len(sel_), threshold_pd_M=thr / 1e6,
                           hot_pd_mean_M=H[sel_].mean() / 1e6, storm_caused_share=S[sel_].sum() / max(H[sel_].sum(), 1),
                           heated_share=HE[sel_].sum() / max(H[sel_].sum(), 1), frac_days_with_storm=Ed[sel_].mean(),
                           frac_leave_tier_without_storm=left))
            v = pool[t]; v[0] += S[sel_].sum(); v[1] += H[sel_].sum(); v[2] += HE[sel_].sum(); v[3] += Ed[sel_].sum(); v[4] += len(sel_)
        if r_ == 'W7':
            o = idx[np.argsort(-H[idx], kind='stable')][:30]
            pd.DataFrame(dict(day=days[o].strftime('%Y-%m-%d'), hot_pd_M=H[o] / 1e6, storm_caused_pd_M=S[o] / 1e6,
                              heated_pd_M=HE[o] / 1e6, storm_in_position=Ed[o], hot_cells=hot[o].sum(1))).to_csv(
                os.path.join(OUT, 'impact_bigdays_W7_top.csv'), index=False, float_format='%.5g')
    for t, _ in TIERS:
        v = pool[t]
        bd.append(dict(region='all (pooled)', pop2020_M=np.nan, cells=np.nan, tier=t, n_days=v[4], threshold_pd_M=np.nan,
                       hot_pd_mean_M=np.nan, storm_caused_share=v[0] / max(v[1], 1), heated_share=v[2] / max(v[1], 1),
                       frac_days_with_storm=v[3] / max(v[4], 1), frac_leave_tier_without_storm=np.nan))
    BD = pd.DataFrame(bd); BD.to_csv(os.path.join(OUT, 'impact_bigdays.csv'), index=False, float_format='%.5g')
    say('\n== largest regional heat days: storm-caused share of hot person-days, days with storm in position, leave tier without storm')
    say(BD[BD.region.isin(['all (pooled)', 'W7', 'W5', 'W10', 'W13', 'W6', 'W12', 'W21'])].round(3).to_string(index=False))
    say(f'\ntotal {time.time() - T0:.0f} s')
    open(os.path.join(OUT, 'impact_ext_result.txt'), 'w').write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()
