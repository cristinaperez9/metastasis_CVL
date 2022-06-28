#!/bin/bash
#SBATCH --constraint='geforce_gtx_titan_x'
source /itet-stor/calmagro/net_scratch/conda/etc/profile.d/conda.sh
conda activate medicaltoolkit
python -u exec.py --mode train --exp_source experiments/my_dataset --exp_dir /usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset --folds 4 --resume_to_checkpoint /usr/bmicnas01/data-biwi-01/bmicdatasets/Processed/Cristina_Almagro/my_3D_dataset/fold_4/last_checkpoint "$@"



