if [ -d /scratch/username/micromamba/envs/museum ]
then
    echo "micromamba already configured"
else
    mkdir /scratch/username/micromamba/
    cd /scratch/username/micromamba/
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    export MAMBA_ROOT_PREFIX=/scratch/username/micromamba/
    ./bin/micromamba shell init -s bash -p /scratch/username/micromamba/
    source ~/.bashrc
    cd ~/museum_project
    micromamba activate
    micromamba create -n museum -c conda-forge
    micromamba activate museum
    micromamba install -y python=3.9.17 -c conda-forge
    micromamba install -y pytorch==2.1 torchvision torchaudio cudatoolkit=11.8 pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge
    pip install -r /home/username/open_clip/scripts/requirements.txt
fi


mkdir -p /scratch/username/data/
if [ -d /scratch/username/data/britmus_images ]
then
    echo "museum already configured"
else
    cp /home/username/museum_project/britmus.zip /scratch/username/data/
    unzip /scratch/username/data/britmus.zip -d /scratch/username/data/
fi

if [ -d /scratch/username/data/va_images ]
then
    echo "museum already configured"
else
    cp /home/username/museum_project/va_images.zip /scratch/username/data/
    unzip /scratch/username/data/va_images.zip -d /scratch/username/data/
fi