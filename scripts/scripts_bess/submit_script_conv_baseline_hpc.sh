# ensemble_string_pre="ensemble_" 
folds=(0 1 2 3)
landmarks=(8 9 11)

# config=../../configs/configs_bess/gp_project/conv_baselines/
# search_dir=../../configs/configs_bess/nnunet
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)'
regex='^.*5gs*'
# not_include_regex="^(?!\s*local\s*$).*"

    # echo "test Config: " $eachconfig

search_dir=../../configs/configs_bess/gp_project/cnn_baselines/
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)'
# regex='Thaler'
# regex='challenge'
echo "search_dir: " $search_dir

for eachconfig in "$search_dir"*
do

    if [[ $eachconfig =~ $regex ]] 
    then
        echo "test Config: " $eachconfig

        for lm in ${landmarks[@]}
        do  
            # echo "lm: " $lm
            for fold in ${folds[@]}
            do  
                fold_regx="^.*_f${fold}.yaml"
                # echo "fold_regx: " $fold_regx
                if [[ $eachconfig =~ $fold_regx ]]
                then



                    outstring="/F${fold}L${lm}_V2/"
                    echo "params: ": $lm, $fold
                    echo "matched config : " $eachconfig

                    sbatch --partition=dcs-gpu --account=dcs-res --nodes=1 --gpus-per-node=1 --mem=40G --cpus-per-task=2 --time=4:00:00 run_train_gpflow.sh --cfg $eachconfig --fold $fold --landmark $lm --out_path_append=$outstring


                    # nnUNet_train 2d nnUNetTrainerV2 $annotator $fold_count --npz

                    # echo " "
                fi


            done
        done
    fi
done
