#!/bin/bash
device_id=$1
result_dir="results/$2"
dataset_path="./videos"
RANDOM_SEEDS=(42)

if [ -z "$device_id" ]; then
    echo "Usage: $0 <device_id>"
    exit 1
fi

for RANDOM_SEED in ${RANDOM_SEEDS[@]}; do
    echo "Random Seed" ${RANDOM_SEED}
    curr_result_dir=${result_dir}_seed${RANDOM_SEED}

    for i in "sph" "tray" "cup" "planes" "ash"; do  
        echo "baseline training on ${i}"
        CUDA_VISIBLE_DEVICES=${device_id} python train_pgsnet.py\
        -dataset_path ${dataset_path}\
        -mirror_type ${i}\
        -result_dir ${curr_result_dir}/PGS/${i}\
        -batch_size 2\
        -random_seed ${RANDOM_SEED}\
        -model pgsnet

        echo "baseline inference on ${i}"
        CUDA_VISIBLE_DEVICES=${device_id} python infer_pgsnet.py\
        -dataset_path ${dataset_path}\
        -mirror_type ${i}\
        -result_dir ${curr_result_dir}/PGS/${i}\
        -random_seed ${RANDOM_SEED}\
        -model pgsnet

        echo "proposed phase 1"
        CUDA_VISIBLE_DEVICES=${device_id} python train_proposed.py\
            -dataset_path ${dataset_path}\
            -mirror_type ${i}\
            -result_dir ${result_dir}/proposed/${i}\
            -batch_size 2\
            -random_seed ${RANDOM_SEED}\
            -patient 10\
            -phase 1

        echo "proposed phase 2"
        CUDA_VISIBLE_DEVICES=${device_id} python train_proposed.py\
            -dataset_path ${dataset_path}\
            -mirror_type ${i}\
            -result_dir ${curr_result_dir}/proposed/${i}\
            -batch_size 2\
            -random_seed ${RANDOM_SEED}\
            -patient 15\
            -phase 2

        echo "proposed inference on ${i}"
        CUDA_VISIBLE_DEVICES=${device_id} python infer_proposed.py\
            -dataset_path ${dataset_path}\
            -mirror_type ${i}\
            -result_dir ${curr_result_dir}/proposed/${i}\
            -random_seed ${RANDOM_SEED}\
            --test_only

    done
done