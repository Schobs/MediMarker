#!/bin/bash
#SBATCH --mem=32G
#SBATCH --mail-user=ogavin1@sheffield.ac.uk
#SBATCH --mail-type=FAIL
#SBATCH --gpus-per-node=1
# Rename the job
#SBATCH --comment=unetr_model_test``

module load Anaconda3/5.3.0

# module load libs/cudnn/7.5.0.56/binary-cuda-10.0.130
module load cuDNN/7.6.4.38-gcccuda-2019b

source activate my_env

# python ../../main.py --cfg ../../configs/configs_BSC_projects/ceph_oscar.yaml
python main.py --cfg configs/configs_BSC_projects/ceph_oscar.yaml