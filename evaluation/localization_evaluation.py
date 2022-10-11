import numpy as np
import pandas as pd

def success_detection_rate(sample_dicts, threshold):
    """ Calculates the percent of samples under a given error threshold. 

    Args:
        sample_dicts ([Dict]): A list of dictionaries where each dict is a sample. 
            Each sample (dict in list) has "ind_errors" which is a list of all landmark errors
            and a "uid", a string representing a unique id.
        Threshold (int): Error threshold.
        key_prepend (str): A string to prepend to the keys when looking at the ind errors e.g. smha emha or ecpv. 
            Currently, only use when using ensembles. Will update single use to smha in future.
    Returns:
        Dict: results of "all" Landmarks  
        Dict: Results of the "individual" (landmark).
        Each tensor consists of boolean variables to show if this prediction ranks top k with each value of k.
    """

    #First filter out samples with no annotations (therefore no errors)
    if "annotation_available" in sample_dicts[0].keys():
        sample_dicts = [s for s in sample_dicts if (s["annotation_available"])]

        #If not annotations avaliable, we cannot measure SDR so return None
        if len(sample_dicts) == 0:
            # print("No samples with annotations available")
            return None 

    error_key = "ind_errors"
   
    total_samples = len(sample_dicts)
    total_landmarks = len(sample_dicts[0][error_key])


    images_within_thresh_all = []
    images_over_thresh_all = []

    images_over_thresh_per_lm = [ [] for x in sample_dicts[0][error_key]]
    images_within_thresh_per_lm = [[] for x in sample_dicts[0][error_key]]

    for i, sample_dict in enumerate(sample_dicts):

        uid = sample_dict["uid"]
        ind_errors = sample_dict[error_key]

      



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
        key_append: string to specify which errors to use e.g. smha emha or ecpv. Currently, only use when using ensembles. Will update single use to smha in future.
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

    # print(sdr_dicts)

    #First do Results over entire data
    only_w_anns = []
    for sublist in ind_lms_results:
        only_w_anns.append([elem for elem in sublist if elem is not None])

    # print("only_w_anns", only_w_anns, len(only_w_anns))
    #If no annotations, return a pd dataframe with all None
    if not any(only_w_anns):
        return pd.DataFrame.from_dict(results_dict, orient='index')


    # only_w_anns = [elem for elem in x for x in ind_lms_results if elem is not None]#filter out ones without GT annotations (i.e. None values)
    
    # for xidx, x in enumerate(only_w_anns):
    #     print("full len %s, only_w_anns len %s" %( len(ind_lms_results[xidx]), len(x)))
    #  [ind_lms_results[x][ind_lms_results[x] != None] for x in range(len(ind_lms_results))] 

    results_dict["Error Mean"]["All"] = np.mean(only_w_anns)
    results_dict["Error Std"]["All"] = np.std(only_w_anns)

    for key, sdr in (sdr_dicts).items():
        results_dict["SDR"+str(key)]["All"] = sdr["all"]

    #Now do individual landmarks
    for i, lm_er in enumerate(only_w_anns):
        # print(i)
        results_dict["Error Mean"]["L"+str(i)] = np.mean(lm_er)
        results_dict["Error Std"]["L"+str(i)] = np.std(lm_er)

        for key, sdr in (sdr_dicts).items():
            results_dict["SDR"+str(key)]["L"+str(i)] = sdr["individual"][i]

    # print("results dictionary: ", results_dict)

    pd_df = pd.DataFrame.from_dict(results_dict, orient='index')

    # print("pd df: ", pd_df)

    return pd_df
    