#!/bin/bash

echo "$1", "$2", "$3"
data_name=$2
seed=$3
base_model='meta-llama/Llama-2-7b-hf'
train_data="./${data_name}/train.jsonl"
val_data="./${data_name}/valid.jsonl"
output_dir="./${data_name}/model"
instruction_model='./alpaca-lora-2'
for sample in 16 64 256; do
    mkdir -p "${output_dir}_A2_${seed}_${sample}"
    echo "seed: $seed, sample: $sample"
    CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
        --base_model $base_model \
        --train_data_path "$train_data" \
        --val_data_path "$val_data" \
        --output_dir "${output_dir}_A2_${seed}_${sample}" \
        --sample $sample \
        --seed "$seed" \
        --batch_size 256 \
        --micro_batch_size 64 \
        --cutoff_len 512 \
        --num_epochs 200 \
        --learning_rate '1e-4' \
        --group_by_length \
        --resume_from_checkpoint $instruction_model \
        --token "$token"
done
