import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsod, pandas as pd
st=['579570','579720','579930','588470','589110','589210','589440','590460','590580','590720','590820','590870','590960','591020','591170','591340','592540','592650','592780','592870','592930','594310','594560','594930','595010','596320','596440','596580','596630','596730','597580','588340','590230']
st=[s+'99999' for s in st]
d=gsod.load(st, range(1979,2023), 'raw/gsod', nthreads=32)
d.to_parquet('data/inputs/gsod_sc.parquet')
print(d.shape, d.STATION.nunique())
