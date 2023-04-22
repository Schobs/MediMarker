#!/bin/bash
#SBATCH --account=kale
#SBATCH --partition=kale
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=34G
#SBATCH --cpus-per-task=12
#SBATCH --time=30:00:00
#SBATCH --mail-user=ogavin1@sheffield.ac.uk

export SLURM_EXPORT_ENV=ALL
module load Anaconda3/5.3.0
source activate my_env
srun python main.py --cfg "configs/configs_BSC_projects/cephalometric_cv.yaml"