#!/bin/bash


module load Anaconda3/5.3.0

# module load libs/cudnn/7.5.0.56/binary-cuda-10.0.130
module load cuDNN/7.6.4.38-gcccuda-2019b

source activate landmark_unet

python ../../run_inference.py  "$@" 



# srun --partition=dcs-gpu-test --account=dcs-res --qos=gpu --nodes=1 --gpus-per-node=1 --mem=60G --cpus-per-task=6 --pty bash

# source run_inference_config.sh --cfg ../configs_bess/4CH_512F_256Res_8GS_AugV1_DS.yaml
# qsub -l gpu=1 -l rmem=12G run_train_config.sh --cfg ../configs_exp/4CH_256F_256Res_4GS.yaml
# qsub -l gpu=1 -l rmem=12G run_train_config.sh --cfg ../configs_exp/4CH_256F_256Res_8GS.yaml
# qsub -l gpu=1 -l rmem=12G run_train_config.sh --cfg ../configs_exp/4CH_256F_256Res_8GS.yaml
# qsub -l gpu=1 -l rmem=12G run_train_config.sh --cfg ../configs_exp/4CH_256F_256Res_12GS.yaml
# qsub -l gpu=1 -l rmem=12G run_train_config.sh --cfg ../configs_exp/4CH_512F_256Res_8GS.yaml
# qsub -l gpu=1 -l rmem=12G run_train_config.sh --cfg ../configs_exp/4CH_512F_512Res_8GS.yaml


# done