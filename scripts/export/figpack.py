"""Make a small transfer pack from the part 8a exports (only what the paper figures need).

Usage: python figpack.py EXPORT_DIR OUT_DIR
Reads gcx_maps.npz, gcx_case.csv, gcx_lead.csv, ibx_06utc.npz from EXPORT_DIR.  Writes OUT_DIR/figpack.tar.xz and
OUT_DIR/figpack.b64 (numbered base64 lines, each ending with a 4-hex CRC so single bad lines can be found and resent).
Written without angle brackets or ampersands so it survives being pasted through a chat message.
"""
import base64, io, os, sys, tarfile, zlib
import numpy as np, pandas as pd

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
GD = (113.5, 23.0)
CROP = dict(lon=(106.0, 128.0), lat=(16.0, 29.0))
KEEP = ['oper|heat|WIDE|t2m|mean', 'oper|heat|WIDE|t2m|agree', 'oper|heat|WIDE|t850|mean', 'oper|heat|WIDE|t850|agree',
        'oper|heat|WIDE|q850|mean', 'oper|heat|WIDE|w500|mean', 'oper|heat|WIDE|w500|agree', 'oper|heat|WIDE|u10|mean',
        'oper|heat|WIDE|v10|mean', 'oper|heat|WIDE|tp6|mean', 'oper|heat|WIDE|msl|mean',
        'oper|placebo|WIDE|t2m|mean', 'oper|placebo|WIDE|t2m|agree', 'oper|heat|NOTC|t2m|mean', 'oper|heat|NOTC|t2m|agree',
        'oper|heat|NULLW|t2m|mean', 'oper|heat|NULL|t2m|mean', 'oper|heat|WIDEH|t2m|mean', 'oper|heat|SHIFTN|t2m|mean',
        'oper|placebo|NOTC|t2m|mean', 'small|heat|WIDE|t2m|mean', 'small|heat|WIDE|t2m|agree', 'small|heat|NOTC|t2m|mean',
        'oper|heat|CTL|msl|mean', 'oper|placebo|CTL|msl|mean', 'oper|heat|CTLminusERA5|t2m|mean', 'small|heat|CTLminusERA5|t2m|mean']


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# 1. maps: crop, keep int8 codes and scales
z = np.load(os.path.join(src, 'gcx_maps.npz'))
keys = list(z['keys']); sc = z['scales']
out = {}
for m in ('oper', 'small'):
    la, lo = z[f'lat_{m}'], z[f'lon_{m}']
    iy = np.where(np.isclose(np.clip(la, *CROP['lat']), la))[0]
    ix = np.where(np.isclose(np.clip(lo, *CROP['lon']), lo))[0]
    out[f'lat_{m}'] = la[iy]; out[f'lon_{m}'] = lo[ix]
    for i, k in enumerate(keys):
        if k in KEEP and k.startswith(m + '|'):
            out['m%03d' % len([x for x in out if x.startswith('m')])] = z['m%03d' % i][iy][:, ix]
            out.setdefault('_k', []).append(k); out.setdefault('_s', []).append(float(sc[i]))
missing = [k for k in KEEP if k not in out.get('_k', [])]
print('maps kept', len(out['_k']), 'missing', missing)
kk, ss = out.pop('_k'), out.pop('_s')
np.savez_compressed(os.path.join(dst, 'fp_maps.npz'), keys=np.array(kk), scales=np.array(ss, np.float32), **out)

# 2. per-case station table (lead mean), rounded
C = pd.read_csv(os.path.join(src, 'gcx_case.csv'))
cols = ['model', 'case', 'kind', 'tid', 'V', 'variant', 'nlead', 't2m', 't850', 'q850', 'tp6', 'onshore', 'v10c', 'u10c', 'T_truth']
C = C[[c for c in cols if c in C.columns]]
C.to_csv(os.path.join(dst, 'fp_case.csv'), index=False, float_format='%.3f')

# 3. per-lead table, trimmed
L = pd.read_csv(os.path.join(src, 'gcx_lead.csv'))
L = L[['model', 'case', 'variant', 'lead', 'T', 'T850', 'msl600', 'msl600_lon', 'msl600_lat', 'T_truth']]
L.round(dict(T=2, T850=2, msl600=1, msl600_lon=1, msl600_lat=1, T_truth=2)).to_csv(os.path.join(dst, 'fp_lead.csv'), index=False)

# 4. IBTrACS fixes within 1700 km of Guangdong, 1980-2022, not extratropical
b = np.load(os.path.join(src, 'ibx_06utc.npz'))
day = pd.Timestamp('1980-01-01') + pd.to_timedelta(b['day'].astype(int), 'D')
dist = hav(b['lon10'] / 10, b['lat10'] / 10, *GD)
uw = b['usa_wind'].astype(int); ww = b['wmo_wind'].astype(int)
keep = np.logical_and.reduce([np.less_equal(dist, 1700), b['nature'] != 1, np.less_equal(day.year.values, 2022)])
I = pd.DataFrame(dict(s=b['sid'][keep], day=b['day'][keep], lat10=b['lat10'][keep], lon10=b['lon10'][keep], uw=uw[keep], ww=ww[keep]))
I.to_csv(os.path.join(dst, 'fp_ib.csv'), index=False)
used = np.unique(I.s)
pd.DataFrame(dict(s=used, sid=b['sids'][used], name=b['names'][used], season=b['seasons'][used])).to_csv(os.path.join(dst, 'fp_ibstorms.csv'), index=False)
print('ib fixes', len(I), 'storms', len(used))

# 5. pack and encode with per-line CRC
names = ['fp_maps.npz', 'fp_case.csv', 'fp_lead.csv', 'fp_ib.csv', 'fp_ibstorms.csv']
tp = os.path.join(dst, 'figpack.tar.xz')
with tarfile.open(tp, 'w:xz', preset=9) as tf:
    for n in names:
        tf.add(os.path.join(dst, n), arcname=n)
raw = open(tp, 'rb').read()
import hashlib
b64 = base64.b64encode(raw).decode()
lines = [b64[i:i + 76] for i in range(0, len(b64), 76)]
with open(os.path.join(dst, 'figpack.b64'), 'w') as f:
    for i, l in enumerate(lines):
        f.write('%04d %s %04x\n' % (i + 1, l, zlib.crc32(l.encode()) % 65536))
print('figpack.tar.xz bytes', len(raw), 'md5', hashlib.md5(raw).hexdigest(), 'b64 lines', len(lines))
