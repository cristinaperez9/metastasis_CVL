#!/bin/bash
#SBATCH --constraint='geforce_gtx_titan_x|geforce_gtx_1080_ti'
source /itet-stor/calmagro/net_scratch/conda/etc/profile.d/conda.sh
conda activate medicaltoolkit
python -u exec.py --mode analysis --exp_source experiments/my_dataset --exp_dir //usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset_modify_architectures --folds 4 "$@"



