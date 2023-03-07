# ensemble_string_pre="ensemble_" 
folds=(0)
landmarks=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)

config=../../configs/configs_bess/gp_project/conv_baseline/BFUSA_8GS.yaml
# search_dir=../../configs/configs_bess/nnunet
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)'
# regex='Thaler'

    # echo "test Config: " $eachconfig


echo "Config: " $config
z
for lm in ${landmarks[@]}
do  
    for fold in ${folds[@]}
    do  


        outstring="/L${lm}/F${fold}"
        echo "outstring : " $outstring

        sbatch --partition=dcs-gpu --account=dcs-res --nodes=1 --gpus-per-node=1 --mem=32 --cpus-per-task=9 --time=8:00:00 run_train_config.sh --cfg $config --fold $fold --landmark $lm --out_path_append $outstring


        # nnUNet_train 2d nnUNetTrainerV2 $annotator $fold_count --npz

        # echo " "


    done
done

