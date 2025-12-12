import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cluster
from scipy.spatial import distance
from sklearn.utils.validation import check_array
import time
import tempfile
import random

def keep_high_var_features(omics_list, num_features=2000):
    retained_omics_list = []
    for i in range(len(omics_list)):
        temp_omics = omics_list[i]
        if temp_omics.shape[1] > num_features:
            features_vars = temp_omics.var(axis=0)
            threshold = sorted(features_vars, reverse=True)[num_features]
            new_omics = temp_omics.loc[:, features_vars > threshold]
            retained_omics_list.append(new_omics)
        else:
            retained_omics_list.append(temp_omics)

    return retained_omics_list


def normalize_matrix(omics_list, type='min-max'):
    retained_omics_list = []
    for i in range(len(omics_list)):
        temp_omics = omics_list[i]
        if type == 'z-score':
            new_omics = preprocessing.scale(temp_omics, axis=0)
            retained_omics_list.append(new_omics)
        elif type == 'min-max':
            new_omics = preprocessing.minmax_scale(temp_omics, axis=0)
            retained_omics_list.append(new_omics)
        else:
            print("Error! required z-score or min-max")

    return retained_omics_list

def proc_data(data_arrays,
                               num_features=2000,
                               normalization_type='min-max'):
    omics_list = []
    for i, data in enumerate(data_arrays):
        # Ensure proper data type conversion
        np_data = np.array(data, dtype=np.float32)
        # Handle any NaN or infinite values
        np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
        df = pd.DataFrame(np_data)
        omics_list.append(df)
        # print(f"Data modality {i + 1} shape: {omics_list[-1].shape}, dtype: {omics_list[-1].dtype}")

    num_omics = len(omics_list)
    print(num_omics)

    # Preprocessing
    omics_list = keep_high_var_features(omics_list, num_features=num_features)
    omics_list = normalize_matrix(omics_list, type=normalization_type)

    # Get sample information
    num_samples = omics_list[0].shape[0]
    # samples_id = omics_list[0].index

    return(num_samples)

