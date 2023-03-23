# ensemble_string_pre="ensemble_" 
folds=(3)
landmarks=(9)

# config=../../configs/configs_bess/gp_project/conv_baselines/
# search_dir=../../configs/configs_bess/nnunet
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)'
# regex='^.*2gs*'
# not_include_regex="^(?!\s*local\s*$).*"

    # echo "test Config: " $eachconfig

search_dir=../../configs/configs_bess/gp_project/final_gp_exps_local/
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)
# regex='Thaler'
# regex='challenge'
echo "search_dir: " $search_dir

for eachconfig in "$search_dir"*
do
    echo "test Config: " $eachconfig

    for lm in ${landmarks[@]}
    do  
        # echo "lm: " $lm
        for fold in ${folds[@]}
        do  
            fold_regx="^.*_f${fold}_l${lm}.yaml"
            # echo "fold_regx: " $fold_regx
            if [[ $eachconfig =~ $fold_regx ]]
            then



                outstring="/F${fold}L${lm}/"
                echo "params: ": $lm, $fold
                echo "matched config : " $eachconfig

                python ../../main.py --cfg $eachconfig 


                # nnUNet_train 2d nnUNetTrainerV2 $annotator $fold_count --npz

                # echo " "
            fi


        done
    done
done
