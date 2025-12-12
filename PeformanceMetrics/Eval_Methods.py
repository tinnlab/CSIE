import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines import CoxPHFitter
from typing import List, Tuple, Union
from pycox.evaluation import EvalSurv
from sksurv.metrics import concordance_index_censored

Numeric = Union[float, int]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.array]

warnings.filterwarnings('ignore')

ResPath = './SP_Results'
savePath = './SP_Results'
all_methods = ["Baseline", "CSIE", "CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF", "NEMO", "MRGCN", "hMKL", "MDICC", "DLSF", "DSIR"]


#### DEFINE FUNCTIONS
## function to find indices for specific times
def find_idx(times, all_times):
    idx_vec = []
    for time in times:
        temp_times = np.abs(all_times - time)
        idx = np.where(temp_times == np.min(temp_times))[0]
        idx_vec.append(idx[0])

    return idx_vec


## calculate the survival probability using hazard ratio
def cal_survprob(pred_train, pred_val):
    pred_train['log_HR'] = pred_train['predTrain'].apply(np.log)

    cph = CoxPHFitter()
    cph.fit(pred_train.loc[:, ['log_HR', 'time', 'status']], duration_col='time', event_col='status')
    baseline_cum_hazard = cph.baseline_cumulative_hazard_

    all_times = baseline_cum_hazard.index.values
    max_time = np.max(all_times)
    target_times = np.linspace(0, max_time, 20).tolist()
    baseline_haz_indices = find_idx(target_times, all_times)
    baseline_haz = baseline_cum_hazard.iloc[baseline_haz_indices, 0].values

    HR_val = pred_val['predVal'].to_numpy()
    CHF_val = HR_val[:, np.newaxis] * baseline_haz
    survprob_val = np.exp(-CHF_val)
    survprob_val = survprob_val.T
    survprob_val = pd.DataFrame(survprob_val)
    survprob_val.index = target_times

    return survprob_val


## calculate risk score using survival probability
def cal_ci(predRes):
    risk_score = -np.log(predRes)
    risk_score = np.sum(risk_score, axis=1)
    risk_score = risk_score / np.max(risk_score)
    return risk_score


def to_array(array_like: NumericArrayLike, to_boolean: bool = False) -> np.array:
    array = np.asarray(array_like)
    shape = np.shape(array)
    if len(shape) > 1:
        raise ValueError(
            f"Input should be a 1-d array. Got a shape of {shape} instead."
        )
    if np.any(array < 0):
        raise ValueError("All event times must be greater than or equal to zero.")
    if to_boolean:
        check_indicators(array)
        return array.astype(bool)
    return array


# get the survival probability for each patient at the time they died or were censored
def get_1surv_prob(pred_df, survival_times):
    time_points = pred_df.index.values
    probabilities = np.zeros(len(survival_times))

    for i, time in enumerate(survival_times):
        distances = np.abs(time_points - time)
        min_distance = np.min(distances)
        candidate_indices = np.where(distances == min_distance)[0]

        if len(candidate_indices) == 1:
            selected_idx = candidate_indices[0]
        else:
            candidate_probs = pred_df.iloc[candidate_indices, i]
            min_prob = candidate_probs.min()
            min_prob_indices = candidate_indices[candidate_probs == min_prob]
            selected_idx = min_prob_indices[0]
        probabilities[i] = pred_df.iloc[selected_idx, i]
    return probabilities


## function to create table for each metric
def create_metric_table(data, metric, datasets):
    metric_data = {method: values[metric] for method, values in data.items()}
    df = pd.DataFrame(metric_data, index=[f'Run {i + 1}' for i in range(len(datasets))])
    df.index = datasets
    return df


## create summary table for C-Index
def create_summary_table(cindex_table):
    summary_data = {
        'Cindex': cindex_table.mean(),
    }

    summary_df = pd.DataFrame(summary_data)
    return summary_df


#### CALCULATE METRIC FOR ALL METHODS
allRes = {}
for method in all_methods:
    all_datasets = os.listdir(os.path.join(ResPath, method))
    # all_datasets = [x for x in all_datasets if "TCGA-" in x]
    all_datasets = sorted(all_datasets)  # Returns new sorted list
    allCIndex = []
    alltdCIndex = []
    allCal = []
    allIBS = []

    for dataset in all_datasets:
        NewPath = os.path.join(ResPath, method, dataset)
        CIndex = []
        tdCIndex = []
        IBS = []
        Cal = []

        for time in range(1, 6):
            for fold in range(1, 6):
                # TrainRes = pd.read_csv(os.path.join(ResPath, method, dataset, "Time" + str(time)) + "/Train_Res_" + str(fold) + ".csv",
                #                        header=0, index_col=0)
                ValRes = pd.read_csv(
                    os.path.join(ResPath, method, dataset, "Time" + str(time)) + "/Val_Res_" + str(fold) + ".csv",
                    header=0,
                    index_col=0)

                if len(ValRes.columns) <= 3:
                    tmp_CI = np.NaN
                else:
                    trueDat = np.array(ValRes[['time', 'status']])
                    predVal = ValRes.drop(['time', 'status'], axis=1)
                    try:
                        riskScore = cal_ci(predVal.to_numpy())
                        tmp_CI = concordance_index_censored(event_indicator=ValRes['status'].values.astype(bool),
                                                            event_time=ValRes['time'].values,
                                                            estimate=riskScore)[0]
                    except:
                        tmp_CI = np.NaN

                CIndex.append(tmp_CI)

        CIndex = np.nanmean(CIndex)

        allCIndex.append(CIndex)

    # allCIndex = np.nanmean(allCIndex)
    allRes[method] = {'CIndex': allCIndex}

CIndex_Table = create_metric_table(allRes, 'CIndex', all_datasets)

CIndex_Table.to_csv(savePath + "/CIndex_Table.csv", sep=",", header=True)

print(CIndex_Table)
