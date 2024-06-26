#!/usr/bin/env bash

set -e

cslc ./layout.csl --fabric-dims=8,3 \
--fabric-offsets=4,1 --params=N:3,d_k:4 -o out --memcpy --channels 1
cs_python run.py --name out