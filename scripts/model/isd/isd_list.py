"""List ISD-Lite files (key, size) on the NOAA AWS mirror for 1980-2021 -> isd_lite_listing.csv.
Usage: python isd_list.py OUT.csv"""
import sys, re, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

B = 'https://noaa-isd-pds.s3.amazonaws.com/'


def year(y):
    rows, tok = [], None
    while True:
        q = {'list-type': '2', 'prefix': f'isd-lite/data/{y}/', 'max-keys': '1000'}
        if tok:
            q['continuation-token'] = tok
        s = urllib.request.urlopen(B + '?' + urllib.parse.urlencode(q), timeout=60).read().decode()
        rows += [(y, k.split('/')[-1], int(z)) for k, z in re.findall(r'<Key>(.*?)</Key>.*?<Size>(\d+)</Size>', s)]
        m = re.search(r'<NextContinuationToken>(.*?)</NextContinuationToken>', s)
        if not m:
            return rows
        tok = m.group(1)


with ThreadPoolExecutor(14) as ex:
    out = [r for rr in ex.map(year, range(1980, 2022)) for r in rr]
d = pd.DataFrame(out, columns=['year', 'file', 'bytes'])
d['sid'] = d.file.str.slice(0, 12)
d.to_csv(sys.argv[1], index=False)
print(d.groupby('year').size().describe(), len(d), d.bytes.sum() / 1e9, 'GB')
