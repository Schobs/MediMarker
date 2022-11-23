# ensemble_string_pre="ensemble_" 
folds=(0 1 2 3 4 5 6 7 8 9 99)
config=../../configs/configs_bess/nnunet/aspire_medium_SA/BFUSA_8GS.yaml
# search_dir=../../configs/configs_bess/nnunet
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)'
# regex='Thaler'
regex=''

    # echo "test Config: " $eachconfig


echo "Config: " $config

for fold in ${folds[@]}
do  



    echo "Fold : " $fold

    sbatch --partition=dcs-gpu --account=dcs-res --nodes=1 --gpus-per-node=1 --mem=40G --cpus-per-task=9 --time=8:00:00 run_train_config.sh --cfg $config --fold $fold


    # nnUNet_train 2d nnUNetTrainerV2 $annotator $fold_count --npz

    # echo " "



done

