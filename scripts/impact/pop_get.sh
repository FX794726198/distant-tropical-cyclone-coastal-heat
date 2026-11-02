#!/bin/bash
# A3: download GHS-POP R2023A (30 arcsec, WGS84) for five epochs to $IMP/pop (login node; public JRC data, no login).
set -euo pipefail
IMP=${IMP:-/scratch/$USER/tcheat/impact}
mkdir -p "$IMP/pop"; cd "$IMP/pop"
for Y in 1980 1990 2000 2010 2020; do
  N=GHS_POP_E${Y}_GLOBE_R2023A_4326_30ss
  U=https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/${N}/V1-0/${N}_V1_0.zip
  [ -s "${N}_V1_0.tif" ] && { echo "have $Y"; continue; }
  curl -fL --retry 4 --retry-delay 5 -o "${N}_V1_0.zip.part" "$U" && mv "${N}_V1_0.zip.part" "${N}_V1_0.zip"
  unzip -o -j "${N}_V1_0.zip" "*.tif" && rm -f "${N}_V1_0.zip"
  ls -l "${N}_V1_0.tif"
done
md5sum *.tif > pop_md5.txt; cat pop_md5.txt
