# #!/bin/bash

default_idx=-1
h_scale=1.0
c_scale=1.0

chord_indices=(9 128 259 295 374 532 797 830 1732 2278 4764 5873 7508 10907 17937 19068)          # 替换成你自己的
hyperscore_indices=(9 128 259 295 374 532 797 830 1732 2278 4764 5873 7508 10907 17937 19068)       # 替换成你自己的

for chord_idx in "${chord_indices[@]}"; do
    for hyperscore_idx in "${hyperscore_indices[@]}"; do
        
        output_folder="demo/c${chord_idx}v${hyperscore_idx}"
        mkdir -p "$output_folder"

        python pipeline.py \
            --folder="$output_folder" \
            --mode=cond \
            --num_gen=3 --bar_gen=8 \
            --chord_idx="${chord_idx}" \
            --hyperscore_idx="${hyperscore_idx}" \
            --continuation_idx="${default_idx}" \
            --hyperscore_scale="${h_scale}" \
            --chord_scale="${c_scale}"
            done
        done

