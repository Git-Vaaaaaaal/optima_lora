#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd /beegfs/data/work/c-2iia/vb710264/optimus
module load python/3.11/anaconda/2024.02
bash
source ~/.bashrc
conda activate optimus_venv

cd /beegfs/data/work/c-2iia/vb710264/optimus/optima_lora
python /beegfs/data/work/c-2iia/vb710264/optimus/optima_lora/tiles_dataset_extract.py