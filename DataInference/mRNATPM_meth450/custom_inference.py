from inference import Predictor
import pandas as pd
from pathlib import Path
import os
import numpy as np

predictor = Predictor('./checkpoints/best_model.pt')

# datPath = '/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TestDataGene/DG2_2'
datPath = '/data/daotran/Cancer_Subtyping/BiB_Submission/Data/NotTCGA_csv'
savePath = '/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TestDataGene/DG4_1'

if os.path.exists(savePath) == False:
    os.makedirs(savePath)

def list_subfolders_pathlib(parent_folder):
    """Lists only the immediate subfolders in a directory using pathlib."""
    parent_path = Path(parent_folder)
    # Use a generator expression to iterate and filter
    subfolders = [
        item.name for item in parent_path.iterdir() if item.is_dir()
    ]
    return subfolders

datasets = list_subfolders_pathlib(datPath)
# datasets = ["GSE61335"]

for dataset in datasets:
    print(dataset)
    if os.path.exists(savePath + '/' + dataset) == False:
        os.makedirs(savePath + '/' + dataset)

    
    allfiles = list(Path(datPath + '/' + dataset).glob('*.csv'))
    allfiles = [f.stem for f in allfiles]

    if dataset == "GSE13041":
        allfiles = ["mRNAGPL570"]

    if dataset == "GSE61335":
        allfiles = ["mRNAGPL19184"]

    if dataset in ["GSE1456", "GSE4271", "GSE4412"]:
        allfiles = ["mRNAGPL96"]

    for file in allfiles:
        # Load your data
        mrna_data = pd.read_csv(datPath + '/' + dataset + '/' + file + '.csv', index_col=0)

        # mrna_data.to_csv(savePath + '/' + dataset + '/' + file + '.csv')

        # Predict meth450 from mRNA
        meth_pred, mrna_new = predictor.predict_new_data(
            mrna_data,
            source_modality='mrna',
            target_modality='meth',
            log_transform=True,
            normalize_setting='independent',
            return_dataframe=True
        )

        q5 = np.quantile(meth_pred.values, 0.05)
        meth_pred[meth_pred < q5] = 0
        meth_pred[meth_pred < 0] = 0
        meth_pred[meth_pred > 1] = 1

        meth_pred.to_csv(savePath + '/' + dataset + '/' + file + '_meth450.csv')
        # mrna_new.to_csv(savePath + '/' + dataset + '/' + file + '.csv')




