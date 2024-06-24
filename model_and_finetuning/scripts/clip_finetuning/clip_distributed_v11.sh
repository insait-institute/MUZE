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
#"Materials" "Technique" "Subjects" "Culture" "Inscription" "Title" "Production place" "Production date" "Object type" "Assoc name" "Producer name" "Curators Comments" "School/style"
#for supreme_key in  "Production place" "Production date" "Object type" "Assoc name" "Producer name" "Curators Comments" "School/style"
# dataset=vamus
dataset=britmus
# for supreme_key in "categories materials objectType" #  styles artistMakerPerson briefDescription historicalContext marksAndInscriptions materials materialsAndTechniques objectHistory objectType partTypes physicalDescription placesOfOrigin production productionDates styles summaryDescription techniques titles
#
for supreme_key in "Materials" #  styles artistMakerPerson briefDescription historicalContext marksAndInscriptions materials materialsAndTechniques objectHistory objectType partTypes physicalDescription placesOfOrigin production productionDates styles summaryDescription techniques titles
do
    srun -n 8 -w gcpl4-eu-2 --cpu_bind=v -G l4-24g:8 -c 12 --mem 256G bash /home/username/open_clip/scripts/clip_finetuning/run_args.sh "$supreme_key" "$dataset"
done