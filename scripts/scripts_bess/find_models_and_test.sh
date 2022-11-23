#!/bin/bash

module load Anaconda3/5.3.0

# module load libs/cudnn/7.5.0.56/binary-cuda-10.0.130
module load cuDNN/7.6.4.38-gcccuda-2019b

source activate landmark_unet


#this  script finds trained models in a directory and calls run_inference to test them all.

# ensemble_string_pre="ensemble_" 
max_fold=3

search_dir=../../configs/configs_bess/april
# regex='(^../configs_bess/ISBI_).*5GS'
regex='(/configs_local/april/ISBI_)'

for eachconfig in "$search_dir"/*

do
    # echo "test Config: " $eachconfig


    echo "Config: " $eachconfig

    fold_count=0

    while [ $fold_count -le $max_fold ]
    do  

    

        echo "Fold Count: " $fold_count

        python ../../run_inference.py --cfg $eachconfig --fold $fold_count
        # /mnt/bess/home/acq19las/landmark_unet/LaNNU-Net/run_inference.py --cfg $eachconfig --fold $fold_count
        # exit 1
        # nnUNet_train 2d nnUNetTrainerV2 $annotator $fold_count --npz

        # echo " "


        ((fold_count++))
    
    done


done