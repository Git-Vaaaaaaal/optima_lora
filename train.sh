#!/bin/ksh
#$ -q gpu
#$ -j y
#$ -o result.out
#$ -N mil_train
cd /beegfs/data/work/c-2iia/vb710264/optimus
module load python
source /beegfs/data/work/c-2iia/vb710264/optimus/optimus_lora/bin/activate
export PYTHONPATH=/work/c-2iia/vb710264/optimus/optimus_lora/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/c-2iia/vb710264/.cache/matplotlib

cd /beegfs/data/work/c-2iia/vb710264/optimus/optima_lora
python /beegfs/data/work/c-2iia/vb710264/optimus/optima_lora/tiles_dataset_extract.py