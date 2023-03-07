import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import re


def success_detection_rate(sample_dicts, threshold):
    """Calculates the percent of samples under a given error threshold.

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

    # First filter out samples with no annotations (therefore no errors)
    if "annotation_available" in sample_dicts[0].keys():
        sample_dicts = [s for s in sample_dicts if (s["annotation_available"])]

        # If not annotations avaliable, we cannot measure SDR so return None
        if len(sample_dicts) == 0:
            # print("No samples with annotations available")
            return None

    error_key = "ind_errors"

    total_samples = len(sample_dicts)
    total_landmarks = len(sample_dicts[0][error_key])

    images_within_thresh_all = []
    images_over_thresh_all = []

    images_over_thresh_per_lm = [[] for x in sample_dicts[0][error_key]]
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

    # Calc % of samples over threshold (using average error)
    perc_over_all = len(images_within_thresh_all) / ((total_samples * total_landmarks))

    # Calc % of samples over threshold, calculate per landmark
    perc_over_ind = [len(x) / total_samples for x in images_within_thresh_per_lm]

    details_dictionary = {
        "all": perc_over_all,
        "individual": perc_over_ind,
        "uids all": images_within_thresh_all,
        "uids individual": images_within_thresh_per_lm,
    }

    return details_dictionary


def nlpd_evaluation(list_of_dicts):
    nlpd_results = {}
    mean_errors = {}
    for sample in list_of_dicts:
        uid = sample["uid"]

        s = sample["mean"].replace(' ', ', ')
        s = re.sub(r'(?<!\d)(\s?)(,)(\s?)(?!\d)', r'\1\3', s)
        pred_mean = np.array(eval(s))

        c = sample["cov"][2:-2].replace('\n', '')
        # Use regular expressions to replace spaces with commas
        c = re.sub("[ ]{1,}", ",", c)
        c = c.replace(',]', ']')
        c = c.replace('[,', '[')

        pred_cov = np.array(eval(c))

        pred_cov = np.array(eval(c))
        # sample["cov"].astype(np.float64)
        target = np.array(eval(sample["target"].replace('.', ', ')))

        assert pred_cov.shape == (2, 2), "Should be only single prediction. Covariance matrix is not 2x2"
        assert target.shape == (1, 2), "Should be only single target. "
        assert pred_mean.shape == (1, 2), "Should be only single prediction."

        target = target[0]
        pred_mean = pred_mean[0]
        nlpd, mean_error = calculate_nlpd(pred_mean, pred_cov, target)
        nlpd_results[uid] = nlpd
        mean_errors[uid] = mean_error
    return nlpd_results,mean_errors


def calculate_nlpd(pred_mean: np.ndarray, pred_cov: np.ndarray, target_mean: np.ndarray, target_cov: np.ndarray = None) -> float:
    """
    Calculates the negative log predictive density (NLPD) between a predicted distribution and a target distribution.

    Parameters:
        pred_mean (ndarray): The mean vector of the predicted distribution.
        pred_cov (ndarray): The covariance matrix of the predicted distribution.
        target_mean (ndarray): The mean vector of the target distribution.
        target_cov (ndarray, optional): The covariance matrix of the target distribution. If not provided, the target distribution is assumed to be a degenerate Gaussian with zero covariance.

    Returns:
        float: The NLPD value.

    """
    pred_dist = multivariate_normal(mean=pred_mean, cov=pred_cov)
    nlpd = -np.log(pred_dist.pdf(target_mean))

    mean_error = np.linalg.norm(pred_mean - target_mean)

    if target_cov is None:
        target_cov = np.zeros_like(pred_cov)
    if np.allclose(target_cov, 0):
        target_cov = 1e-6 * np.eye(target_cov.shape[0])

    nlpd = -np.log(pred_dist.pdf(target_mean))
    # nlpd = -np.mean(np.log(pred_dist.pdf(target_mean)) + 0.5 * np.log(np.linalg.det(target_cov))
    #                 + 0.5 * np.trace(np.linalg.solve(target_cov, pred_cov)) + 0.5 * len(pred_mean) * np.log(2*np.pi))

    # logger = logging.getLogger()

    # logger.info("Prediction Mean %s, Covariance %s and Target %s. NLPD: ", pred_mean, pred_cov, target_mean, nlpd)
    return nlpd, mean_error


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
    for k in sdr_dicts.keys():
        results_dict["SDR" + str(k)] = {}

    # print("empty keys", results_dict)

    # print(sdr_dicts)

    # First do Results over entire data
    only_w_anns = []
    for sublist in ind_lms_results:
        only_w_anns.append([elem for elem in sublist if elem is not None])

    # print("only_w_anns", only_w_anns, len(only_w_anns))
    # If no annotations, return a pd dataframe with all None
    if not any(only_w_anns):
        return pd.DataFrame.from_dict(results_dict, orient="index")

    # only_w_anns = [elem for elem in x for x in ind_lms_results if elem is not None]#filter out ones without GT annotations (i.e. None values)

    # for xidx, x in enumerate(only_w_anns):
    #     print("full len %s, only_w_anns len %s" %( len(ind_lms_results[xidx]), len(x)))
    #  [ind_lms_results[x][ind_lms_results[x] != None] for x in range(len(ind_lms_results))]

    results_dict["Error Mean"]["All"] = np.mean(only_w_anns)
    results_dict["Error Std"]["All"] = np.std(only_w_anns)

    for key, sdr in (sdr_dicts).items():
        results_dict["SDR" + str(key)]["All"] = sdr["all"]

    # Now do individual landmarks
    for i, lm_er in enumerate(only_w_anns):
        # print(i)
        results_dict["Error Mean"]["L" + str(i)] = np.mean(lm_er)
        results_dict["Error Std"]["L" + str(i)] = np.std(lm_er)

        for key, sdr in (sdr_dicts).items():
            results_dict["SDR" + str(key)]["L" + str(i)] = sdr["individual"][i]

    # print("results dictionary: ", results_dict)

    pd_df = pd.DataFrame.from_dict(results_dict, orient="index")

    # print("pd df: ", pd_df)

    return pd_df
