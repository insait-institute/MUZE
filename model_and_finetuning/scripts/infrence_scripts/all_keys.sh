#!/bin/bash -x

source ~/.bashrc
micromamba activate museum

export MASTER_PORT=12803
export MASTER_ADDR=localhost
export PYTHONPATH="$PYTHONPATH:$PWD/src"

#artistMakerPerson briefDescription categories historicalContext marksAndInscriptions materials materialsAndTechniques objectHistory objectType partTypes physicalDescription placesOfOrigin production productionDates styles summaryDescription techniques titles
#styles categories materials
#"Technique" "Subjects" "Culture" # "Materials" "Inscription" "Title" "Production place" "Production date" "Object type" "Assoc name" "Producer name" "Curators Comments" 
# artistMakerPerson briefDescription categories historicalContext marksAndInscriptions materials materialsAndTechniques objectHistory objectType partTypes physicalDescription placesOfOrigin production productionDates styles summaryDescription techniques titles
architecture=2
dataset="vamus"
# 
for supreme_key in materials
do
    srun -n 8 -w gcpl4-eu-2 --cpu_bind=v -G l4-24g:8 -c 12 --mem 300G bash /home/username/open_clip/scripts/infrence_scripts/key.sh "$supreme_key" "$architecture" "$dataset"
done