# ensemble_string_pre="ensemble_" 
max_fold=0
search_dir=../../configs/configs_bess/nnunet/mimn/
# regex='(^../configs_bess/ISBI_).*5GS'
# regex='(^../configs_bess/april/ISBI_)'
# regex='Thaler'
regex='challenge'
for eachconfig in "$search_dir"/*

do
    # echo "test Config: " $eachconfig

    if [[ $eachconfig =~ $regex ]]
    then
        if [[ $eachconfig =~ "gneg" ]]
        then


            echo "Config: " $eachconfig

            fold_count=0

            while [ $fold_count -le $max_fold ]
            do  

            

                echo "Fold Count: " $fold_count
                echo $eachconfig
                # sbatch --partition=dcs-gpu --account=dcs-res --nodes=1 --gpus-per-node=1 --mem=40G --cpus-per-task=9 --time=30:00:00 run_train_config.sh --cfg $eachconfig 
                source run_train_config.sh --cfg $eachconfig 

                # nnUNet_train 2d nnUNetTrainerV2 $annotator $fold_count --npz

                # echo " "


                ((fold_count++))
            
            done
        fi
    fi
    

done