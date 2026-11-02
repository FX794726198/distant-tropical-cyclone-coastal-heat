"""Revision variants of the storm removal (independent review, 2026-10-29). Imported after gc_build; patches B.modify.

WIDEL: as WIDE (core + axisymmetric outer circulation), but the outer removal tapers to zero 50 km before the station
       nearest to the storm, so that no station is edited at the initial time.
W1000: as WIDE with the outer removal ending at 1,000 km instead of 1,500 km.
"""
import numpy as np
import gc_build as B
from symvortex import SymRemover
from tcfilter import gc_dist

_modify = B.modify


def sym_remove_r(out, lat, lon, lon0, lat0, R_in, frac, info, R_out):
    half = (R_out + 150.0) / 111.0
    j = np.where((lat >= lat0 - half) & (lat <= lat0 + half))[0]
    hx = half / max(np.cos(np.radians(min(abs(lat0) + half, 80.0))), 0.2)
    i = np.where((lon >= lon0 - hx) & (lon <= lon0 + hx))[0]
    sl = (slice(j[0], j[-1] + 1), slice(i[0], i[-1] + 1))
    LON, LAT = np.meshgrid(lon[i], lat[j])
    S = SymRemover(LAT, LON, lon0, lat0, R_in=R_in, R_out=R_out)
    pairs = [('10m_u_component_of_wind', '10m_v_component_of_wind', None)] + \
            [('u_component_of_wind', 'v_component_of_wind', k) for k in range(out['u_component_of_wind'].shape[0])]
    for un, vn, k in pairs:
        u = out[un][sl] if k is None else out[un][k][sl]
        v = out[vn][sl] if k is None else out[vn][k][sl]
        du, dv = S.wind_part(u, v)
        if k is None:
            out[un][sl] = u - frac * du; out[vn][sl] = v - frac * dv
        else:
            out[un][k][sl] = u - frac * du; out[vn][k][sl] = v - frac * dv
    out['mean_sea_level_pressure'][sl] -= frac * S.scalar_part(out['mean_sea_level_pressure'][sl])
    for k in range(out['geopotential'].shape[0]):
        out['geopotential'][k][sl] -= frac * S.scalar_part(out['geopotential'][k][sl])
    info['R_out'] = float(R_out)


def modify(fields, lat, lon, centre, kind, null_centre=None, shift_centre=None):
    if kind in ('WIDEL', 'W1000'):
        out, info = _modify(fields, lat, lon, centre, 'NOTC')
        R, dmin = B.radius_for(*centre)
        R_out = (dmin - 50.0) if kind == 'WIDEL' else 1000.0
        sym_remove_r(out, lat, lon, centre[0], centre[1], info['R'], 1.0, info, max(R_out, info['R'] + 100.0))
        info['dmin'] = float(dmin)
        return out, info
    return _modify(fields, lat, lon, centre, kind, null_centre=null_centre, shift_centre=shift_centre)


B.modify = modify
