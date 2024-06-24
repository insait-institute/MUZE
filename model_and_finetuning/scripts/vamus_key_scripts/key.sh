#!/bin/bash -x

source ~/.bashrc
micromamba activate museum

export MASTER_PORT=12803
export MASTER_ADDR=localhost
# {amp,amp_bf16,amp_bfloat16,bf16,fp16,pure_bf16,pure_fp16,fp32}] 

export PYTHONPATH="$PYTHONPATH:$PWD/src"
# srun -n 2 --cpu_bind=v -p batch.36h -w hala -G a6000:2 -c 12 --mem 70G python -u /home/username/open_clip_original/src/training/main.py \

#srun -n 1 --cpu_bind=v -w gcpl4-eu-0 -G l4-24g:1 -c 12 --mem 180G --pty --time 0-4 bash
#bash /home/username/open_clip/scripts/installs.sh
#bash /home/username/scripts/installs.sh
#  -w gcpl4-eu-1
# --include-context
# --train-vision \
#--include-context \
python -u /home/username/open_clip_original/src/training/main_lora_zeroshotclassif_transformer.py \
    --supreme-key "$1" \
    --include-context \
    --exp-name "${3}_${1}_T" \
    --group "${3}_T${2}_28feb"\
    --head-architecture "$2" \
    --lora-rank 32 \
    --lora-alpha 1.0 \
    --lora-dropout 0.0 \
    --save-frequency 2 \
    --report-to wandb \
    --wandb_project_name "museum-tabdata-test"\
    --dataset-type csv \
    --train-data "/data/work-gcp-europe-west4-a/username/data/dataset_splits/${3}_train.csv" \
    --val-data "/data/work-gcp-europe-west4-a/username/data/dataset_splits/${3}_val.csv" \
    --test-data "/data/work-gcp-europe-west4-a/username/data/dataset_splits/${3}_test.csv" \
    --csv-img-key image_path \
    --csv-caption-key caption \
    --accum-freq 1 \
    --csv-separator , \
    --model ViT-B-32 \
    --pretrained 'laion2b_s34b_b79k' \
    --seed 0 \
    --local-loss \
    --gather-with-grad \
    --warmup 0 \
    --batch-size 1024 \
    --lr 1e-4 \
    --epochs 101 \
    --workers 8 \
    --coca-contrastive-loss-weight 0 \
    --coca-caption-loss-weight 1 \
    --dataset-resampled \
    --lock-text \
    --lock-text-unlocked-layers 13 \
    --lock-image \
    --lock-image-unlocked-groups 5 \
    --precision fp16 \
    --classes "/data/work-gcp-europe-west4-a/username/data/dataset_splits/${3}_classes.json" \



    # --local-model "/home/username/open_clip/saved_models/clip1/distributed/2024_02_06-12_42_19-model_ViT-B-32-lr_0.0001-b_1024-j_8-p_amp/checkpoints/epoch_94.pt" \
    # --local-model-is-lora 1 \
    # --train-data "/home/username/open_clip/finetuning/train_va_dataset_new.csv" \
    # --local-model "/home/username/open_clip/saved_models/clip1/distributed/2024_02_06-12_42_19-model_ViT-B-32-lr_0.0001-b_1024-j_8-p_amp/checkpoints/epoch_94.pt" \
    # --local-model-is-lora 1 \


# --report-to wandb \

    # --cross_entropy \
    # --pretrained 'laion2b_s34b_b79k' \
    # --val-data "/home/username/open_clip/finetuning/val12.csv" \
    # --local-model-is-lora \
    #/home/username/open_clip_original/logs/2024_02_13-09_45_32-model_ViT-B-32-lr_1e-05-b_1024-j_8-p_fp16/checkpoints/epoch_86.pt
        # --local-model "/home/username/open_clip/saved_models/clip1/distributed/2024_02_06-12_42_19-model_ViT-B-32-lr_0.0001-b_1024-j_8-p_amp/checkpoints/epoch_94.pt" \