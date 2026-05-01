#!/bin/bash

module load conda
conda activate walrus

python3 scripts/convert_lbm_to_well.py --input datasets/kinet/doubly_periodic/weakly_compressible_isoT_fluids/sys_Re-5e4_Ma-1en1/D2Q9_shape-256-256_T-10000_H-b6e704.h5 --bc-x periodic --bc-y periodic --split --split-ratios 0.5,0.1,0.4