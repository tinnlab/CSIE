from inference import Predictor
import pandas as pd
from pathlib import Path
import os

predictor = Predictor('./checkpoints/best_model.pt')

# datPath = '/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TestDataGene/DG2_2'
datPath = '/data/daotran/Cancer_Subtyping/BiB_Submission/Data/NotTCGA_csv'
savePath = '/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TestDataGene/DG4_1'

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

    # if dataset == "GSE13041":
    #     allfiles = ["mRNAGPL570"]

    # if dataset == "GSE61335":
    #     alfiles = ["mRNAGPL19184"]
        
    # if dataset in ["GSE1456", "GSE4271", "GSE4412"]:
    #     allfiles = ["mRNAGPL96"]

    for file in allfiles:
        # Load your data
        mrna_data = pd.read_csv(datPath + '/' + dataset + '/' + file + '.csv', index_col=0)

        # Predict meth450 from mRNA
        # mirna_pred = predictor.predict_new_data(
        #     mrna_data,
        #     source_modality='mrna',
        #     target_modality='mirna',
        #     log_transform=True,
        #     return_dataframe=True
        # )

        mirna_pred, mrna_new = predictor.predict_new_data(
            mrna_data,
            source_modality='mrna',
            target_modality='mirna',
            log_transform=True,
            normalize_setting='independent',
            return_dataframe=True
        )

        mirna_pred[mirna_pred < 0] = 0
        mrna_new[mrna_new < 0] = 0

        mrna_new.to_csv(savePath + '/' + dataset + '/' + file + '.csv')

        if (dataset not in ["GSE13041", "GSE1456", "GSE4271", "GSE4412", "GSE61335"] or
            (dataset == "GSE13041" and file == "mRNAGPL570") or
            (dataset == "GSE61335" and file == "mRNAGPL19184") or
            (dataset in ["GSE1456", "GSE4271", "GSE4412"] and file == "mRNAGPL96")):
            mirna_pred.to_csv(savePath + '/' + dataset + '/' + file + '_miRNA.csv')








