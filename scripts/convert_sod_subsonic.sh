#!/bin/bash

module load conda
conda activate walrus

python3 scripts/convert_lbm_to_well.py \
    --input datasets/kinet/sod/compressible_fluids/sys_Pr-71en2_vuy-2_visc-25en3/D2Q9_shape-3001-5_T-1000_H-8908d7.h5 \
    --bc-x wall --bc-y periodic \
    --crop-to-even \
    --pad-to-multiple 32 \
    --split --split-ratios 0.5,0.1,0.4
