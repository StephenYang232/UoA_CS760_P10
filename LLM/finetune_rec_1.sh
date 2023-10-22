#!/bin/bash

echo "$1", "$2", "$3"
data_name=$2
seed=$3
base_model='huggyllama/llama-7b'
train_data="./${data_name}/train.jsonl"
val_data="./${data_name}/valid.jsonl"
output_dir="./${data_name}/model"
instruction_model='./alpaca-lora-1'
for sample in 64 256; do
    mkdir -p "${output_dir}_A1_${seed}_${sample}"
    echo "seed: $seed, sample: $sample"
    CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
        --base_model $base_model \
        --train_data_path "$train_data" \
        --val_data_path "$val_data" \
        --output_dir "${output_dir}_A1_${seed}_${sample}" \
        --sample $sample \
        --seed "$seed" \
        --batch_size 256 \
        --micro_batch_size 64 \
        --cutoff_len 512 \
        --num_epochs 200 \
        --learning_rate '1e-4' \
        --group_by_length \
        --resume_from_checkpoint $instruction_model
done
