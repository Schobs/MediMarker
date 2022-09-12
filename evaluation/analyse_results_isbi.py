import numpy as np
import os 
import pandas as pd
from pandas import ExcelWriter
from localization_evaluation import success_detection_rate, generate_summary_df
import copy


def get_parameters(name):
    """Gets important parameters from a file name

    

    Args:
        name (str): file name
    """
    split_name = name.split("_")

    param_dict = {
        "Name": name,
        "Dataset": split_name[0],
        "Max Features": split_name[1],
        "Resolution": split_name[2],
        "Gauss Sigma": split_name[3],
        "Min Feature Resolution": split_name[4],
        "Aug": split_name[5],
        "Deep Supervision": split_name[6],     
        }

    return param_dict


def analyse_all_folds(root_path, name_of_exp, models_to_test, early_stop_strategy, folds):
    """Function that takes experiment name and selection of folds and summaries results.


    Args:
        root_path (string): Path to saved experiments
        name_of_exp (string): Name of experiment to analyse
        models_to_test [string]: List of model checkpoints to analyse 
            (If using test data only pick the one chosen by your stopping strategy, otherwise you are overfitting the test set.)
        early_stop_strategy (string): String describing your early stop strategy, only used for saving info.
        folds [int]: List of folds to analyse
    """
    '''
    It will return: 
    1) failed experiments (i.e. experiments which did not generate results files)
    2) For every saved model in models_to_test it will generate:
        A) Individual pandas DF of completed experiments, results for each sample with a note of their fold
        B) Summary pandas DF of mean error, SDR results of completed experiments 
    3) Experiment summary of parameters extracted from the experiment name
    4) It will also do seperate analysis of all models of the experiments and ascertain the best model, 
        collating results for all folds. It will note the epoch. This will be saved in DF "overfitted_analysis".
        This is to check how well the early stopping strategy is over/underfitting the data.
        It will note if a model in models_to_test was the best model afterall.
    4)
        

    '''

    exp_path = os.path.join(root_path,name_of_exp)
    individual_dicts = {}
    skipped_folds = []
    summary_dicts = {}

    #Use the experiment name to extract exp parameters
    info_dict = get_parameters(name_of_exp)

        
    #For each fold load the results files
    for fold in folds:
        summ_file_name = os.path.join(exp_path, "T_summary_results_fold"+str(fold) +".xlsx")
        ind_file_name = os.path.join(exp_path, "T_individual_results_fold"+str(fold) +".xlsx")

        try:                                    
            summary_file = pd.ExcelFile(summ_file_name)
            ind_file = pd.ExcelFile(ind_file_name,)
        except:
            print("Results for fold %s not found, the exp might have failed. Checked paths %s and %s. Skipping this fold in analysis." % (fold, summ_file_name, ind_file_name))
            skipped_folds.append(fold)
            continue

        #get all the sheets in the results files (i.e. results for all model checkpoints.)
        all_sheet_names = (summary_file.sheet_names)
        for ckpt in all_sheet_names:
            ckpt_general = ckpt.split("_fold")[0] #split here to get rid of "_foldX" suffix
            if ckpt_general in models_to_test:  
                
                #Read individual results and add fold column
                ind_results = ind_file.parse(ckpt,  index_col=[0])
                ind_results["Fold"] = fold


                #Move fold column to the front
                ind_results = ind_results[ ['Fold'] + [ col for col in ind_results.columns if col != 'Fold' ] ]

                #If entry for this checkpoint does not exist, create one, else add onto it
                if ckpt_general in individual_dicts:

                    individual_dicts[ckpt_general] = pd.concat([individual_dicts[ckpt_general], ind_results])
                else:
                    individual_dicts[ckpt_general] = (ind_results)


    #Summary
    if skipped_folds == folds:
        print("No folds found for this exp. It didn't train \n")
        return info_dict, True #return only the info dict and a bool indicating failure

    else:
        #get all landmark keys
        all_lm_keys = list(individual_dicts[list(individual_dicts.keys())[0]].filter(regex="[L]{1}\d").columns.values)


        for chkpt_key, results in individual_dicts.items():
            filter_df = results[all_lm_keys +["uid"]]
            list_res = [{"uid": int(x["uid"]), "ind_errors": x[all_lm_keys]} for idx, x in filter_df.iterrows()]

            #Get SDR results
            radius_list = [5,10,15,20]
            outlier_results = {}
            for rad in radius_list:
                out_res_rad = success_detection_rate(list_res, rad)
                outlier_results[rad] = (out_res_rad)    

                # print("Outlier results ",rad, out_res_rad["all"])


            #Get all landmark errors into a 2D list, 1 list for each landmark.
            all_errors = results[all_lm_keys].values.T.tolist()
        
            summary_results = generate_summary_df(all_errors, outlier_results  )
            if skipped_folds == []:
                skipped_folds = "None"
            summary_results["Skipped Folds"] = str(skipped_folds)

            #Get Info there
            for info_key, info in info_dict.items():
                summary_results[info_key] = info
            summary_dicts[chkpt_key] = summary_results

 

    with ExcelWriter(os.path.join(exp_path, "individual_results_allfolds.xlsx")) as writer:
        for n, df in (individual_dicts).items():
            df.to_excel(writer, n, index=False)

    with ExcelWriter(os.path.join(exp_path, "summary_results_allfolds.xlsx")) as writer:
        for n, df in (summary_dicts).items():
            df.to_excel(writer, n, index=False)
    

    #Also save elsewhere
    collation_location = os.path.join(root_path, "all_summaries")
    with ExcelWriter(os.path.join(collation_location, name_of_exp+"_individual.xlsx")) as writer:
        for n, df in (individual_dicts).items():
            df.to_excel(writer, n, index=False)

    with ExcelWriter(os.path.join(collation_location, name_of_exp+"_summary_results_allfolds.xlsx")) as writer:
        for n, df in (summary_dicts).items():
            df.to_excel(writer, n, index=False)


    return summary_dicts, False


# root_path = "/mnt/bess/home/acq19las/landmark_unet/LaNNU-Net/outputsISBI/v2"
root_path = "/shared/tale2/Shared/schobs/landmark_unet/lannUnet_exps/ISBI/param_search"
# name_of_exp = "ISBI_256F_512Res_8GS_32MFR_AugACEL_DS3"
models_to_test = ["model_best_valid_coord_error", "model_best_valid_loss", "model_latest"]
# models_to_test = ["model_best_valid_coord_error"]

early_stop_strategy = "150 epochs val best coord error"
folds= [0,1,2,3]




#Summarise all summaries into one big file #################


#Walk through dir to get all exps
all_exps =  [x for x in os.listdir(root_path) if "ISBI" in x]
info_keys = ["Name","Dataset", "Max Features", "Resolution","Gauss Sigma",  "Min Feature Resolution","Aug", "Deep Supervision"]

collation_location = os.path.join(root_path, "all_summaries")
os.makedirs(collation_location, exist_ok=True)

summary_of_summaries = {}
failed_experiments = []
for exp in all_exps:
    print("Analysing Experiment: ", exp)
    summary_dicts, failed_exp = analyse_all_folds(root_path, exp, models_to_test, early_stop_strategy, folds)

    if not failed_exp:

        for chkpt_key, results in summary_dicts.items():
            
            #Extract a summary from the summary!
            relevant_results = {}

            #Get Info there
            for info_key in info_keys:
                relevant_results[info_key] = results[info_key]


            relevant_results["Skipped Folds"] = results["Skipped Folds"]

            for err_key, err in results["All"].items():
                relevant_results[err_key] = err

            
            relevant_results = pd.DataFrame.from_dict(relevant_results).drop_duplicates()
    

            if chkpt_key in summary_of_summaries:

                summary_of_summaries[chkpt_key] = pd.concat([summary_of_summaries[chkpt_key], relevant_results])
            else:
                summary_of_summaries[chkpt_key] = (relevant_results)
    else:
        #Deal with failed experiments later, because we might not know the checkpoint keys yet.
        failed_experiments.append([exp, summary_dicts])



# Deal with failed experiments. For each failed exp, write an entry into each chkpt sheet.
for failed_exp, info_dict in failed_experiments:
    for chkpt_key, results in summary_of_summaries.items():

        #Create a row wth the name, note that it was skipped and the info we returned and saved
        new_row = {"Name": failed_exp, "Skipped Folds": "All"}
        for info_key, info in info_dict.items():
            new_row[info_key] = info

        summary_of_summaries[chkpt_key] = pd.concat([results, pd.DataFrame.from_dict([new_row])])




with ExcelWriter(os.path.join(collation_location, "summary_of_summaries.xlsx")) as writer:
    for n, df in (summary_of_summaries).items():
        df.to_excel(writer, n, index=False)