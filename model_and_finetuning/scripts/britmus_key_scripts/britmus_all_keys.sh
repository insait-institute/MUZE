#!/bin/bash -x

source ~/.bashrc
micromamba activate museum

export MASTER_PORT=12803
export MASTER_ADDR=localhost
# {amp,amp_bf16,amp_bfloat16,bf16,fp16,pure_bf16,pure_fp16,fp32}] 

export PYTHONPATH="$PYTHONPATH:$PWD/src"
# srun -n 2 --cpu_bind=v -p batch.36h -w hala -G a6000:2 -c 12 --mem 70G python -u /home/username/open_clip_original/src/training/main.py \

#srun -n 1 --cpu_bind=v -w gcpl4-eu-3 -G l4-24g:1 -c 12 --mem 180G --pty --time 0-4 bash
#bash /home/username/open_clip/scripts/installs.sh
#bash /home/username/scripts/installs.sh
#  -w gcpl4-eu-1

#"Technique" "Subjects" "School/style" "Production place" "Production date" "Culture" "Object type"
# "School/style"
architecture=2
dataset="britmus"
for supreme_key in "Technique" "Materials" "Subjects" "Culture" "Inscription" "Title" "Production place" "Production date" "Object type" "Assoc name" "Producer name" "Curators Comments" 
do
    srun -n 8 -w gcpl4-eu-7 --cpu_bind=v -G l4-24g:8 -c 12 --mem 300G bash /home/username/open_clip/scripts/britmus_key_scripts/key.sh "$supreme_key" "$architecture" "$dataset"
done