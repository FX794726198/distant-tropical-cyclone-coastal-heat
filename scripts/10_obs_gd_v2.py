"""Formal observational analysis for Guangdong (Fig. 1 and SI S1), v2 (revision after the independent review, 2026-10-29).

Changes from v1: bootstrap resamples are seeded per call, so identical inputs give identical intervals in every table;
wet-bulb temperature from simultaneous hourly temperature and dew point (ISD-lite, 12_isd_hourly.py) at 14 LT, over
12-17 LT, at the hour of the highest temperature and as a daily maximum; cross-shore wind at the coastal stations along
each station's observed sea-breeze axis; how often the storm later approached; analogue-quality diagnostics; the
GHCN-GSOD station match; an interval for the number of hot days added by storms.

Heat on days when a tropical cyclone (TC) sits 300-1300 km from Guangdong, compared with circulation-analogue days.
Inputs (paths in DATA below): GSOD daily Tmax/dew point for 19 Guangdong stations; GHCN-Daily Tmax (same network, check);
ISD hourly winds at 5 coastal stations; ERA5 daily-mean Z500 and 850-hPa T/q/u/v on a 1.5-deg grid (90-160E, 5-45N,
June-October); TC positions at 06 UTC (IBTrACS v04r01 subset from ib_export.py, or the ERA5 tracker for testing).

Usage: python 10_obs_gd.py [--tracks ibx_06utc.npz | --tracks-era5 tc_tracks.csv] --out DIR [--nboot 1000]
"""
import argparse, json, os
import numpy as np, pandas as pd

E5 = os.environ.get('ERA5DIR', 'era5')                  # z500_uv.npz and wb2_850.npz from scripts/inputs/z500_get.py, wb2_850.py
DATA = dict(gsod='data/inputs/gsod_sc.parquet', isd='data/inputs/isd_gd.parquet',
            z500=os.path.join(E5, 'z500_uv.npz'), f850=os.path.join(E5, 'wb2_850.npz'),
            ghcn='data/inputs/ea_pack.npz', stations='data/db/gd_stations.csv',
            isdday='data/inputs/isd_day.parquet')
GD = (113.5, 23.0)
Y0, Y1 = 1980, 2022
SECT = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
BANDS = [(300, 700), (700, 1300)]
COAST = ['59493099999', '59501099999', '59663099999', '59673099999', '59658099999']


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def harmonic_anom(x, days, k=3, by_year=True):
    """Remove a k-harmonic seasonal cycle (fit on all data) and, if by_year, each June-October season's mean."""
    doy = days.dayofyear.values; yr = days.year.values
    X = np.stack([np.ones_like(doy)] + [f(2 * np.pi * h * doy / 365.25) for h in range(1, k + 1) for f in (np.sin, np.cos)], 1).astype(float)
    x = np.asarray(x, float); A = np.full(x.shape, np.nan)
    m = np.isfinite(x)
    if m.sum() < 1500:
        return A
    b = np.linalg.lstsq(X[m], x[m], rcond=None)[0]; A = x - X @ b
    if by_year:
        for Y in np.unique(yr):
            mm = yr == Y
            A[mm] = A[mm] - np.nanmean(A[mm]) if np.isfinite(A[mm]).sum() > 60 else np.nan
    return A


def field_anom(f, days):
    doy = days.dayofyear.values; yr = days.year.values
    X = np.stack([np.ones_like(doy)] + [fn(2 * np.pi * h * doy / 365.25) for h in (1, 2) for fn in (np.sin, np.cos)], 1).astype(float)
    b = np.linalg.lstsq(X, f, rcond=None)[0]; fa = f - X @ b
    for Y in np.unique(yr):
        m = yr == Y; fa[m] -= fa[m].mean(0)
    return fa


def stull_tw(T, RH):
    return (T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) + np.arctan(T + RH) - np.arctan(RH - 1.676331)
            + 0.00391838 * RH ** 1.5 * np.arctan(0.023101 * RH) - 4.686035)


def load_tracks(a, days):
    """Per day (06 UTC): list of (lon, lat, storm id, wind kt) for TCs (>= 25 kt)."""
    if a.tracks:
        z = np.load(a.tracks)
        d = pd.Timestamp('1980-01-01') + pd.to_timedelta(z['day'].astype(int), 'D')
        uw = z['usa_wind'].astype(float); ww = z['wmo_wind'].astype(float)
        uw[uw == 255] = np.nan; ww[ww == 255] = np.nan
        wind = np.where(np.isfinite(uw), uw, ww)
        df = pd.DataFrame(dict(day=d, lon=z['lon10'] / 10, lat=z['lat10'] / 10, sid=z['sids'][z['sid']], wind=wind,
                               nature=z['nature']))
        df = df[df.nature != 1]                        # drop extratropical fixes
        names = dict(zip(z['sids'], z['names']))
    else:
        tr = pd.read_csv(a.tracks_era5, parse_dates=['time']); tr = tr[tr.time.dt.hour == 6]
        df = pd.DataFrame(dict(day=tr.time.dt.normalize(), lon=tr.lon, lat=tr.lat, sid=tr.tid.astype(str), wind=tr.w / 0.5144))
        names = {}
    by = {k: g for k, g in df.groupby('day')}
    out = []
    for d in days:
        g = by.get(d)
        out.append(np.zeros((0, 4)) if g is None else np.c_[g.lon.values, g.lat.values, np.zeros(len(g)), g.wind.values])
    sid = [None if by.get(d) is None else by[d].sid.values for d in days]
    return out, sid, names, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks'); ap.add_argument('--tracks-era5'); ap.add_argument('--out', required=True)
    ap.add_argument('--nboot', type=int, default=1000)
    ap.add_argument('--coverage', action='store_true', help='write coverage.csv for the hourly measures and stop')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(42)

    # ---- grids and days (June-October, 1980-2022)
    Z = np.load(DATA['z500'], allow_pickle=True)
    zd_all = pd.DatetimeIndex(Z['days']); keep = (zd_all.year >= Y0) & (zd_all.year <= Y1)
    zd = zd_all[keep]; yr = zd.year.values; doy = zd.dayofyear.values
    zlat, zlon = Z['lat'], Z['lon']
    za = field_anom((Z['z'] / 9.80665)[keep].astype(float), zd)
    W = np.load(DATA['f850'], allow_pickle=True)
    WLO, WLA = np.meshgrid(W['lon'], W['lat'], indexing='ij'); key = {(round(x, 2), round(y, 2)): i for i, (x, y) in enumerate(zip(WLO.ravel(), WLA.ravel()))}
    ix = np.array([key[(round(x, 2), round(y, 2))] for x, y in zip(zlon, zlat)])
    f850 = {k: field_anom(W[v][keep][:, ix].astype(float), zd) for k, v in
            (('t', 'temperature'), ('u', 'u_component_of_wind'), ('v', 'v_component_of_wind'), ('q', 'specific_humidity'))}
    F = {'z': za / za.std(), 't': f850['t'] / f850['t'].std(), 'u': f850['u'] / f850['u'].std(), 'v': f850['v'] / f850['v'].std()}
    dGD = hav(zlon, zlat, *GD); iGD = np.argsort(dGD)[:4]
    T850 = f850['t'][:, iGD].mean(1); Q850 = 1000 * f850['q'][:, iGD].mean(1)

    # ---- stations
    stn = pd.read_csv(DATA['stations'], index_col=0); ids = [int(s) for s in stn.index]
    g = pd.read_parquet(DATA['gsod']); g['STATION'] = g.STATION.astype(np.int64)
    g = g[g.STATION.isin(ids)]
    TX = g.pivot_table(index='DATE', columns='STATION', values='MAX').reindex(zd)
    TD = g.pivot_table(index='DATE', columns='STATION', values='DEWP').reindex(zd)
    AX = pd.DataFrame({s: harmonic_anom(TX[s].values, zd) for s in TX.columns}, index=zd)
    y_tmax = AX.mean(axis=1).values
    es = lambda T: 6.112 * np.exp(17.67 * T / (T + 243.5))
    RH = np.clip(100 * es(TD) / es(TX), 1, 100); TW = stull_tw(TX, RH)
    y_td = np.nanmean(np.column_stack([harmonic_anom(TD[s].values, zd) for s in TD.columns]), 1)
    y_tw = np.nanmean(np.column_stack([harmonic_anom(TW[s].values, zd) for s in TW.columns]), 1)
    y_rh = np.nanmean(np.column_stack([harmonic_anom(RH[s].values, zd) for s in RH.columns]), 1)
    # GHCN-Daily, the same network processed independently
    G = np.load(DATA['ghcn'], allow_pickle=True); gdays = pd.DatetimeIndex(G['days'])
    gsel = np.char.startswith(G['ids'].astype(str), 'CHM') & (G['lat'] >= 21) & (G['lat'] <= 25.5) & (G['lon'] >= 109.5) & (G['lon'] <= 117.5)
    pos = pd.Series(np.arange(len(gdays)), index=gdays).reindex(zd).values
    y_ghcn = np.nanmean(np.column_stack([harmonic_anom(G['TMAX'][i][pos].astype(float), zd) for i in np.where(gsel)[0]]), 1)
    # ISD: afternoon (12-17 LT) onshore share at 5 coastal stations, wind from 100-260 deg
    D = pd.read_parquet(DATA['isd']); D['loct'] = D.DATE + pd.Timedelta('8h'); D = D[D.loct.dt.hour.between(12, 17)]
    D['day'] = D.loct.dt.normalize()
    D['onshore'] = np.where(D.wd.notna(), ((D.wd >= 100) & (D.wd <= 260)).astype(float), np.nan)
    D.loc[D.ws == 0, 'onshore'] = 0.0
    gg = D.groupby(['day', 'STATION']).agg(on=('onshore', 'mean'), n=('onshore', 'count')); gg = gg[gg.n >= 3]
    y_on = gg.on.unstack().reindex(zd).mean(axis=1).values
    # ISD-lite hourly: simultaneous temperature and dew point -> wet-bulb temperature (network means of station anomalies)
    Hd = pd.read_parquet(DATA['isdday']); Hd['STATION'] = Hd.STATION.astype(np.int64); Hd = Hd[Hd.STATION.isin(ids)]
    def netmean(col, stns=None):
        P = Hd.pivot_table(index='day', columns='STATION', values=col).reindex(zd)
        if stns is not None:
            P = P[[c for c in P.columns if c in stns]]
        return np.nanmean(np.column_stack([harmonic_anom(P[c].values, zd) for c in P.columns]), 1)
    HV = {k: netmean(k) for k in ('T14', 'Td14', 'RH14', 'Tw14', 'T_pm', 'Td_pm', 'Tw_pm', 'Tw_atTmax', 'Tw_max', 'Tw_max_8', 'Tmax_h')}
    # cross-shore wind at the five coastal stations: onshore axis = direction of the mean 14 LT minus 02 LT wind vector
    # (the sea-breeze/land-breeze axis seen in the observations); u_cs > 0 blows onshore
    cs = {}; axis = {}
    for c in COAST:
        q = Hd[Hd.STATION == int(c)].set_index('day')
        def vec(h):
            wd, ws = q[f'wd{h:02d}'], q[f'ws{h:02d}']
            return -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))      # vector the air moves towards
        u14, v14 = vec(14); u02, v02 = vec(2)
        du, dv = np.nanmean(u14) - np.nanmean(u02), np.nanmean(v14) - np.nanmean(v02)
        ex_, ey_ = du / np.hypot(du, dv), dv / np.hypot(du, dv)
        axis[c] = float((np.degrees(np.arctan2(ex_, ey_)) + 180) % 360)              # direction the onshore wind comes from
        cs[c] = pd.DataFrame({'cs14': u14 * ex_ + v14 * ey_, 'cs02': u02 * ex_ + v02 * ey_})
    CS14 = pd.concat({c: cs[c]['cs14'] for c in COAST}, axis=1).reindex(zd)
    CS02 = pd.concat({c: cs[c]['cs02'] for c in COAST}, axis=1).reindex(zd)
    y_cs14 = np.nanmean(np.column_stack([harmonic_anom(CS14[c].values, zd) for c in COAST]), 1)
    y_cs02 = np.nanmean(np.column_stack([harmonic_anom(CS02[c].values, zd) for c in COAST]), 1)
    y_csd = np.nanmean(np.column_stack([harmonic_anom((CS14[c] - CS02[c]).values, zd) for c in COAST]), 1)
    raw_cs14 = CS14.mean(axis=1).values; raw_csd = (CS14 - CS02).mean(axis=1).values

    # ---- TCs
    tcpos, sids, names, TR = load_tracks(a, zd)
    nd = len(zd); dist = np.full(nd, 9999.0); brg = np.full(nd, np.nan); wnd = np.full(nd, np.nan); sid = np.array([''] * nd, object)
    dmin_any = np.full(nd, 9999.0)
    for j in range(nd):
        p = tcpos[j]
        if not len(p):
            continue
        dd = hav(GD[0], GD[1], p[:, 0], p[:, 1]); dmin_any[j] = dd.min()
        ts = p[:, 3] >= 34
        if ts.any():
            q = np.where(ts)[0][np.argmin(dd[ts])]
            dist[j] = dd[q]; brg[j] = bearing(GD[0], GD[1], p[q, 0], p[q, 1]); wnd[j] = p[q, 3]; sid[j] = sids[j][q]
    sec = np.full(nd, -1); ok = np.isfinite(brg); sec[ok] = (((brg[ok] + 22.5) % 360) // 45).astype(int)
    near = dmin_any < 300
    expo = (dist >= 300) & (dist < 1300) & ~near
    east = expo & (brg < 180)
    sw = expo & (brg >= 180) & (brg < 270)
    pool = dmin_any > 1500
    if a.coverage:
        # sample coverage of the hourly measures: east-side days and storm-free candidate analogue days pass the same
        # station-level rules (12_isd_hourly.py) and need at least one valid station for the network mean
        rows = []
        for col, lab in (('Tw14', '14 LT'), ('Tw_pm', '12-17 LT mean'), ('Tw_atTmax', 'hour of highest temperature'),
                         ('Tw_max', 'highest of the reports, >= 4 reports'), ('Tw_max_8', 'highest of the reports, >= 8 reports')):
            P = Hd.pivot_table(index='day', columns='STATION', values=col).reindex(zd).reindex(columns=ids)
            nst = P.notna().sum(1).values; y = HV[col]
            for glab, gsel in (('east', east), ('storm-free pool', pool)):
                used = gsel & np.isfinite(y)
                r = dict(measure=lab, group=glab, days=int(gsel.sum()), days_used=int(used.sum()),
                         station_days=int(nst[used].sum()), share_station_days=float(nst[gsel].sum() / (len(ids) * gsel.sum())),
                         stations_per_day=float(nst[used].mean()), year0=int(yr[used].min()), year1=int(yr[used].max()),
                         years=int(len(np.unique(yr[used]))))
                if glab == 'east':
                    r['storms'] = int(len(np.unique(sid[used]))); r['storms_all'] = int(len(np.unique(sid[gsel])))
                rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(a.out, 'coverage.csv'), index=False, float_format='%.4g')
        print(pd.DataFrame(rows).to_string()); return

    DIAG = []
    def analog(sel, y, K=5, rad=1000.0, fields=('z',), pool_=None, per=None, diag=False):
        """Per-day excess of y over the mean of K circulation analogues (other years, +-15 days, no TC within 1500 km)."""
        pool_ = pool if pool_ is None else pool_
        idx = np.where(sel & np.isfinite(y))[0]
        if per is not None:
            idx = idx[(yr[idx] >= per[0]) & (yr[idx] <= per[1])]
        ex = np.full(len(idx), np.nan)
        for n, j in enumerate(idx):
            mask = dGD > 600
            for lo_, la_ in tcpos[j][:, :2]:
                mask &= hav(zlon, zlat, lo_, la_) > rad
            cand = np.where(pool_ & (yr != yr[j]) & (np.abs(doy - doy[j]) <= 15) & np.isfinite(y))[0]
            r = sum(((F[f][cand][:, mask] - F[f][j, mask]) ** 2).mean(1) for f in fields)
            best = cand[np.argsort(r)[:K]]
            ex[n] = y[j] - y[best].mean()
            if diag:
                zd_, za_ = za[j, mask], za[best][:, mask].mean(0)
                DIAG.append(dict(day=j, mse=float(((zd_ - za_) ** 2).mean()), var=float((zd_ ** 2).mean()),
                                 corr=float(np.corrcoef(zd_, za_)[0, 1]), bias=float((zd_ - za_).mean()),
                                 rand_mse=float(((zd_ - za[rng.choice(cand, K)][:, mask].mean(0)) ** 2).mean())))
        return idx, ex

    def boot(ex, grp, nb=a.nboot):
        rng = np.random.default_rng(20261029)          # seeded per call: the same days give the same interval everywhere
        ug = np.unique(grp); gi = {u: np.where(grp == u)[0] for u in ug}
        bs = np.array([np.nanmean(ex[np.concatenate([gi[u] for u in rng.choice(ug, len(ug))])]) for _ in range(nb)])
        m = np.nanmean(ex); se = np.nanstd(bs)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(m) / se / sqrt(2)))) if se > 0 else np.nan
        return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p

    def summary(sel, y, label, **kw):
        if (sel & np.isfinite(y)).sum() < 10:
            return dict(label=label, n_days=int((sel & np.isfinite(y)).sum()))
        idx, ex = analog(sel, y, **kw)
        grp = np.array([s if s else f'd{j}' for s, j in zip(sid[idx], idx)])
        m, lo, hi, p = boot(ex, grp)
        return dict(label=label, n_days=int(len(idx)), n_storms=int(len(np.unique(grp))), raw=float(np.nanmean(y[idx])),
                    excess=float(m), lo=float(lo), hi=float(hi), p=float(p))

    res = {}
    # 1. rose: 2 distance bands x 8 sectors
    rose = []
    for b0, b1 in BANDS:
        for k, s in enumerate(SECT):
            sel = (dist >= b0) & (dist < b1) & ~near & (sec == k)
            if sel.sum() < 15:
                rose.append(dict(band=f'{b0}-{b1}', sector=s, n_days=int(sel.sum()))); continue
            r = summary(sel, y_tmax, f'{b0}-{b1} {s}'); r.update(band=f'{b0}-{b1}', sector=s); rose.append(r)
    pd.DataFrame(rose).to_csv(os.path.join(a.out, 'rose.csv'), index=False, float_format='%.4g')
    # 2. main contrasts and fingerprints (analogue matched on Z500 + 850-hPa T, u, v as in round 5 robustness)
    main = []
    for lab, sel in (('east', east), ('southwest', sw)):
        for vn, y in (('Tmax', y_tmax), ('Tmax_GHCN', y_ghcn), ('T850', T850), ('Tmax_minus_T850', y_tmax - T850), ('q850', Q850),
                      ('dewpoint', y_td), ('wetbulb', y_tw), ('RH', y_rh), ('onshore_share', y_on),
                      ('T_14LT', HV['T14']), ('Td_14LT', HV['Td14']), ('RH_14LT', HV['RH14']), ('Tw_14LT', HV['Tw14']),
                      ('T_12_17LT', HV['T_pm']), ('Td_12_17LT', HV['Td_pm']), ('Tw_12_17LT', HV['Tw_pm']), ('Tw_at_Tmax', HV['Tw_atTmax']),
                      ('Tw_daily_max', HV['Tw_max']), ('Tw_daily_max_8obs', HV['Tw_max_8']), ('Tmax_hourly', HV['Tmax_h']),
                      ('crossshore_14LT', y_cs14), ('crossshore_02LT', y_cs02), ('crossshore_14_minus_02', y_csd)):
            r = summary(sel, y, f'{lab} {vn}'); r.update(group=lab, var=vn, match='Z500'); main.append(r)
    pd.DataFrame(main).to_csv(os.path.join(a.out, 'fingerprints.csv'), index=False, float_format='%.4g')
    # 3. robustness of the east-side Tmax excess
    rob = []
    POOL2 = 1700 if a.tracks else 2500                # the IBTrACS subset holds fixes within 1,700 km of Guangdong only
    for lab, kw in (('base: Z500, K=5, TC mask 1000 km', {}), ('K=3', dict(K=3)), ('K=10', dict(K=10)),
                    ('TC mask 800 km', dict(rad=800.0)), ('TC mask 1500 km', dict(rad=1500.0)),
                    ('Z500 + 850-hPa T, u, v', dict(fields=('z', 't', 'u', 'v'))), (f'pool: no TC within {POOL2} km', dict(pool_=dmin_any > POOL2)),
                    ('1980-2000', dict(per=(1980, 2000))), ('2001-2022', dict(per=(2001, 2022)))):
        r = summary(east, y_tmax, lab, **kw); rob.append(r)
    r = summary(east, y_ghcn, 'GHCN-Daily stations'); rob.append(r)
    for b0, b1 in BANDS:
        rob.append(summary(east & (dist >= b0) & (dist < b1), y_tmax, f'east, {b0}-{b1} km'))
    for w0, w1, lab in ((34, 64, 'tropical storm'), (64, 96, 'typhoon cat 1-2'), (96, 999, 'typhoon cat 3-5')):
        rob.append(summary(east & (wnd >= w0) & (wnd < w1), y_tmax, f'east, {lab}'))
    rnd = np.zeros(nd, bool); rnd[rng.choice(np.where(pool)[0], 600, replace=False)] = True
    rob.append(summary(rnd, y_tmax, 'random TC-free days (null)'))
    pd.DataFrame(rob).to_csv(os.path.join(a.out, 'robustness.csv'), index=False, float_format='%.4g')
    # 4. per-station excess (east side)
    st_rows = []
    for s in AX.columns:
        r = summary(east, AX[s].values, str(s)); r.update(station=str(s), lat=float(stn.loc[s, 'lat']), lon=float(stn.loc[s, 'lon'])); st_rows.append(r)
    pd.DataFrame(st_rows).to_csv(os.path.join(a.out, 'stations.csv'), index=False, float_format='%.4g')
    # 5. onset composite: first day in the sector after 2 days outside it (no TC within 300 km in the 3 days before)
    ons = []
    for lab, sel in (('east', east), ('southwest', sw)):
        ent = [j for j in range(5, nd - 3) if sel[j] and not sel[j - 1] and not sel[j - 2] and (dmin_any[j - 3:j] >= 300).all()
               and yr[j - 5] == yr[j] and yr[j + 3] == yr[j]]
        for lag in range(-5, 4):
            v = np.array([y_tmax[j + lag] for j in ent])
            bs = [np.nanmean(rng.choice(v, len(v))) for _ in range(a.nboot)]
            ons.append(dict(group=lab, lag=lag, n=len(ent), mean=float(np.nanmean(v)), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(ons).to_csv(os.path.join(a.out, 'onset.csv'), index=False, float_format='%.4g')
    # 6. hot days: share with an east-side TC vs calendar-matched expectation, two periods; dry vs humid heat
    hot = y_tmax >= np.nanpercentile(y_tmax, 95)
    share = []
    for per in ((1980, 2000), (2001, 2022)):
        ys = np.arange(per[0], per[1] + 1)
        def stat(sample):
            k = np.isin(yr, sample)
            hk = np.where(hot & k)[0]
            obs = east[hk].mean()
            exp = np.mean([east[k & (np.abs(doy - doy[j]) <= 7)].mean() for j in hk])
            return obs, exp
        o, e = stat(ys); bs = np.array([stat(rng.choice(ys, len(ys))) for _ in range(300)])
        share.append(dict(period=f'{per[0]}-{per[1]}', n_hot=int((hot & np.isin(yr, ys)).sum()), observed=o, expected=e, ratio=o / e,
                          obs_lo=np.percentile(bs[:, 0], 2.5), obs_hi=np.percentile(bs[:, 0], 97.5),
                          ratio_lo=np.percentile(bs[:, 0] / bs[:, 1], 2.5), ratio_hi=np.percentile(bs[:, 0] / bs[:, 1], 97.5)))
    pd.DataFrame(share).to_csv(os.path.join(a.out, 'hot_share.csv'), index=False, float_format='%.4g')
    idx, exh = analog(east, hot.astype(float))
    grp = np.array([s if s else f'd{j}' for s, j in zip(sid[idx], idx)])
    ug = np.unique(grp); gi = {u: np.where(grp == u)[0] for u in ug}; rb = np.random.default_rng(20261029)
    tot_b = np.array([np.nansum(exh[np.concatenate([gi[u] for u in rb.choice(ug, len(ug))])]) for _ in range(a.nboot)])
    res['tc_made_hot_days'] = dict(total=float(np.nansum(exh)), lo=float(np.percentile(tot_b, 2.5)), hi=float(np.percentile(tot_b, 97.5)),
                                   n_hot=int(hot.sum()), share=float(np.nansum(exh) / hot.sum()))
    # how often the storm of an east-side day later came within 300 km of the centre of the network (06 UTC fixes)
    later = {1: 0, 2: 0, 3: 0, 5: 0}; ne = 0
    for j in np.where(east)[0]:
        s0 = sid[j]; ne += 1
        for L in later:
            hit = False
            for k in range(1, L + 1):
                if j + k < nd and yr[j + k] == yr[j] and sids[j + k] is not None:
                    m_ = np.array([x == s0 for x in sids[j + k]])
                    if m_.any() and (hav(GD[0], GD[1], tcpos[j + k][m_, 0], tcpos[j + k][m_, 1]) < 300).any():
                        hit = True
            later[L] += hit
    res['east_storm_within_300km_later'] = {f'{L}d': later[L] / ne for L in later}
    # analogue quality: east-side days and, for comparison, the same number of random storm-free days
    DIAG.clear(); analog(east, y_tmax, diag=True); De = pd.DataFrame(DIAG)
    DIAG.clear(); analog(rnd, y_tmax, diag=True); Dr = pd.DataFrame(DIAG)
    res['analogue_quality'] = dict(east_explained=float(1 - De.mse.mean() / De['var'].mean()), east_corr=float(De['corr'].median()),
                                   east_bias_m=float(De.bias.mean()), east_random_explained=float(1 - De.rand_mse.mean() / De['var'].mean()),
                                   null_explained=float(1 - Dr.mse.mean() / Dr['var'].mean()), null_corr=float(Dr['corr'].median()))
    # GHCN-Daily stations and their nearest GSOD station
    gi_ = np.where(gsel)[0]
    rows_g = []
    for i in gi_:
        dd = hav(G['lon'][i], G['lat'][i], stn.lon.values, stn.lat.values)
        rows_g.append(dict(ghcn=str(G['ids'][i]), lat=float(G['lat'][i]), lon=float(G['lon'][i]), nearest_gsod=str(stn.index[np.argmin(dd)]),
                           dist_km=float(dd.min())))
    pd.DataFrame(rows_g).to_csv(os.path.join(a.out, 'ghcn_gsod_match.csv'), index=False, float_format='%.4g')
    res['coast_onshore_axis_from_deg'] = axis
    res['crossshore_raw'] = dict(east_14=float(np.nanmean(raw_cs14[east])), pool_14=float(np.nanmean(raw_cs14[pool])),
                                 east_diurnal=float(np.nanmean(raw_csd[east])), pool_diurnal=float(np.nanmean(raw_csd[pool])))
    np.savez(os.path.join(a.out, 'kde.npz'), tc_hot=np.c_[y_tmax[east & hot], y_td[east & hot]],
             free_hot=np.c_[y_tmax[pool & hot], y_td[pool & hot]], tc_all=np.c_[y_tmax[east], y_td[east]])
    # 7. TC position density on east-side and SW exposure days (1-deg bins) and case table helpers
    H = {}
    for lab, sel in (('east', east), ('southwest', sw), ('all', expo)):
        pts = np.concatenate([tcpos[j][:, :2] for j in np.where(sel)[0]])
        H[lab], xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=[np.arange(95, 141), np.arange(5, 41)])
    np.savez(os.path.join(a.out, 'tc_density.npz'), xe=xe, ye=ye, **H)
    pd.DataFrame(dict(day=zd, tmax=y_tmax, dist=dist, brg=brg, sector=sec, wind=wnd, sid=sid, east=east, sw=sw, pool=pool, hot=hot)).to_csv(
        os.path.join(a.out, 'daily.csv'), index=False, float_format='%.4g')
    res.update(n_days=int(nd), n_east=int(east.sum()), n_sw=int(sw.sum()), n_pool=int(pool.sum()), n_hot=int(hot.sum()),
               tracks=a.tracks or a.tracks_era5, years=[Y0, Y1], n_ghcn=int(gsel.sum()), n_gsod=int(AX.shape[1]))
    json.dump(res, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    print(pd.DataFrame(rob)[['label', 'n_days', 'n_storms', 'excess', 'lo', 'hi']].round(2).to_string())
    print(pd.DataFrame(main)[['label', 'n_days', 'excess', 'lo', 'hi', 'p']].round(3).to_string())


if __name__ == '__main__':
    main()
