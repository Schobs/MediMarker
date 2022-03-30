import numpy as np
import pandas as pd

def success_detection_rate(sample_dicts, threshold):
    """ Calculates the percent of samples under a given error threshold. 

    Args:
        sample_dicts ([Dict]): A list of dictionaries where each dict is a sample. 
            Each sample (dict in list) has "ind_errors" which is a list of all landmark errors
            and a "uid", a string representing a unique id.
        Threshold (int): Error threshold.
    Returns:
        Dict: results of "all" Landmarks  
        Dict: Results of the "individual" (landmark).
        Each tensor consists of boolean variables to show if this prediction ranks top k with each value of k.
    """

    total_samples = len(sample_dicts)
    total_landmarks = len(sample_dicts[0]["ind_errors"])

    images_within_thresh_all = []
    images_over_thresh_all = []

    images_over_thresh_per_lm = [ [] for x in sample_dicts[0]["ind_errors"]]
    images_within_thresh_per_lm = [[] for x in sample_dicts[0]["ind_errors"]]

    for i, sample_dict in enumerate(sample_dicts):

        uid = sample_dict["uid"]
        ind_errors = sample_dict["ind_errors"]

      



        for j, lm_err in enumerate(ind_errors):
            if lm_err > threshold:
                images_over_thresh_per_lm[j].append(uid)
                images_over_thresh_all.append(uid)
            else:
                images_within_thresh_per_lm[j].append(uid)
                images_within_thresh_all.append(uid)


    #Calc % of samples over threshold (using average error) 
    perc_over_all = len(images_within_thresh_all)/((total_samples*total_landmarks))

    #Calc % of samples over threshold, calculate per landmark
    perc_over_ind = [len(x)/total_samples for x in images_within_thresh_per_lm ]

    details_dictionary = {"all": perc_over_all, "individual": perc_over_ind, "uids all": images_within_thresh_all,  "uids individual": images_within_thresh_per_lm,  }

    return details_dictionary


def generate_summary_df(ind_lms_results, sdr_dicts):

    """     
    Generates pandas dataframe with summary statistics.


    Args:
        ind_lms_results [[]]: A 2D list of all lm errors. A list for each landmark
        sdr_dicts ([Dict]): A list of dictionaries from the function 
            localization_evaluation.success_detection_rate().
    Returns:pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
                           for i in user_dict.keys() 
                           for j in user_dict[i].keys()},
                       orient='index')
        PandasDataframe: A dataframe with statsitics including mean error, std error of All and individual
            landmarks. Also includes the SDR detection rates.
            It should look like:
            df = {"Mean Er": {"All": 1, "L0": 1,...}, "std Er": {"All":1, "l0": 1, ...}, ...}

    """

    results_dict = {"Error Mean": {}, "Error Std": {}}
    for k in  sdr_dicts.keys():
        results_dict["SDR"+str(k)] ={}


    # print("empty keys", results_dict)


    #First do Results over entire data
    results_dict["Error Mean"]["All"] = np.mean(ind_lms_results)
    results_dict["Error Std"]["All"] = np.std(ind_lms_results)

    for key, sdr in (sdr_dicts).items():
        results_dict["SDR"+str(key)]["All"] = sdr["all"]


    #Now do individual landmarks
    for i, lm_er in enumerate(ind_lms_results):
        results_dict["Error Mean"]["L"+str(i)] = np.mean(lm_er)
        results_dict["Error Std"]["L"+str(i)] = np.std(lm_er)

        for key, sdr in (sdr_dicts).items():
            results_dict["SDR"+str(key)]["L"+str(i)] = sdr["individual"][i]

    # print("results dictionary: ", results_dict)

    pd_df = pd.DataFrame.from_dict(results_dict, orient='index')

    # print("pd df: ", pd_df)

    return pd_df
    