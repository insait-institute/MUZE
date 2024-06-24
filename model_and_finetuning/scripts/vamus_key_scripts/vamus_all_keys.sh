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

#artistMakerPerson briefDescription categories historicalContext marksAndInscriptions materials materialsAndTechniques objectHistory objectType partTypes physicalDescription placesOfOrigin production productionDates styles summaryDescription techniques titles
#styles categories materials
# artistMakerPerson briefDescription categories historicalContext marksAndInscriptions materials materialsAndTechniques objectHistory objectType partTypes physicalDescription placesOfOrigin production productionDates styles summaryDescription techniques titles
architecture=2
dataset="vamus"
for supreme_key in categories objectType placesOfOrigin production productionDates styles techniques materials artistMakerPerson historicalContext marksAndInscriptions materialsAndTechniques briefDescription objectHistory partTypes physicalDescription summaryDescription titles
do
    srun -n 8 -w gcpl4-eu-3 --cpu_bind=v -G l4-24g:8 -c 12 --mem 300G bash /home/username/open_clip/scripts/vamus_key_scripts/key.sh "$supreme_key" "$architecture" "$dataset"
done