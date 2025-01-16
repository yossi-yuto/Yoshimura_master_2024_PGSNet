#!/bin/bash
device_id=$1
result_dir="results/$2"
dataset_path="/data2/yoshimura/mirror_detection/DATA/videos"
RANDOM_SEED=2024
result_dir=${result_dir}_seed${RANDOM_SEED}

# 引数が渡されていない場合のエラーチェック
if [ -z "$device_id" ]; then
    echo "Usage: $0 <device_id>"
    exit 1
fi

# ループ処理
for i in "sph" "tray" "cup" "planes" "ash"; do  

    echo "proposed training on ${i}"
    CUDA_VISIBLE_DEVICES=${device_id} python train.py\
        -dataset_path ${dataset_path}\
        -mirror_type ${i}\
        -result_dir ${result_dir}/proposed/${i}\
        -batch_size 2\
        -random_seed ${RANDOM_SEED}\
        -patient 15

    echo "proposed inference on ${i}"
    CUDA_VISIBLE_DEVICES=${device_id} python infer.py\
        -dataset_path ${dataset_path}\
        -mirror_type ${i}\
        -result_dir ${result_dir}/proposed/${i}

    echo "baseline training on ${i}"
    CUDA_VISIBLE_DEVICES=${device_id} python train_pgsnet.py\
    -dataset_path ${dataset_path}\
    -mirror_type ${i}\
    -result_dir ${result_dir}/VMD/${i}\
    -batch_size 2\
    -random_seed ${RANDOM_SEED}\
    -model pgsnet

    echo "baseline inference on ${i}"
    CUDA_VISIBLE_DEVICES=${device_id} python infer_pgsnet.py\
    -dataset_path ${dataset_path}\
    -mirror_type ${i}\
    -result_dir ${result_dir}/VMD/${i}\
    -model pgsnet
done