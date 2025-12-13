# CSIE

This repo contains the source code and scripts for reproducing the results reported in our paper "CSIE: Cancer Subtyping via Inference and Ensemble".
Please follow the below instructions:

## 1. Requirements

### Core Requirements (CSIE only)

CSIE includes two different modules: Data Inference and Ensemble Clustering. 
To run the Data Inference module, you need a Python-installed environment. You can create one using the provided yaml file (InfEnv.yml).

To run the Ensemble Clustering module you need an R-installed environment the following packages. **Note:** The specified versions are required if you want to reproduce results of CSIE reported in the paper.

| Package | Version |
|---------|---------|
| r-base | 4.3.2 |
| tidyverse | 2.0.0 |
| survival | 3.5.7 |
| matrixStats | 1.1.0 |
| SNFtool | 2.3.1 |
| igraph | 2.0.2 |
| cluster | 2.1.8 |
| RhpcBLASctl | latest |



### Additional Requirements (Comparison methods)

To run 12 comparison methods you need to install additional packages to the R environment. Some of the comparison methods are designed in Python, so you need to use the previously created Python environment to run them. 

#### R packages

**From CRAN:**
- devtools
- data.table
- mltools
- survminer
- metap
- PMA
- IntNMF
- wordspace
- kernlab
- polycor
- psych
- FactoMineR
- pbmcapply
- dplyr
- reticulate
- callr
- MASS
- quadprog
- Rtsne
- blockForest

**From Bioconductor:**
- ANF
- ConsensusClusterPlus
- SIMLR

**From GitHub:**
- CIMLR (`danro9685/CIMLR`)
- NNLM (`linxihui/NNLM`)
- NEMO (`Shamir-Lab/NEMO/NEMO`)
- LRACluster (`Zaoqu-Liu/LRAcluster`)

   
## 2. Setup
```bash
# Clone the repository
git clone https://github.com/tinnlab/CSIE.git
cd CSIE

# create result folders
cd PeformanceMetrics
mkdir Subtyping_Results
mkdir SubSurvClin
mkdir SP_Results

# create a data folder
cd ..
mkdir Data
cd /Data

# Download processed data
# Download data for CSIE
wget "   " -O CSIE_Main.zip
wget "https://seafile.tinnguyen-lab.com/f/f0b1f5ce16634e758a87/?dl=1" -O CSIE_Relevant.zip

# Download data for comparison methods
wget "   " -O Others_Main.zip
wget "https://seafile.tinnguyen-lab.com/f/5f1d638d32a641d08e56/?dl=1" -O Others_Relevant.zip

# unzip the downloaded files
```

## 3. Run analysis for CSIE and comparisons method

Please note that the CSIE_Main folder includes all datasets analyzed in this study, even the datasets with imputed omics types. Therefore, you can directly run the Ensemble Clustering module with these datasets to reproduce reported results of CSIE. However, we also provide you with guidances ro run the Data Inference so that you would understand more about the pipeline of this module.


# Run the Data Inference module of CSIE
# This module includes three autoencoders which are used for imputing three different data matrices: miRNArpm, miRNAiso, and DNA methylation.

```bash
wget "https://seafile.tinnguyen-lab.com/f/60ffdf2c5d4e4235acca/?dl=1" -O CSIE_Inference.zip

# unzip the downloaded files
```


```bash
cd CSIE/DataInference
cd mRNATPM_meth450
python run.py --quick ## this script is for training the model
python custom_inference.py ## this script is for inferring the target omics using the trained model
```

```bash
cd CSIE/DataInference
cd mRNATPM_miRNARPM
python run.py --quick ## this script is for training the model
python custom_inference.py ## this script is for inferring the target omics using the trained model
```

```bash
cd CSIE/DataInference
cd mRNATPM_miRNAiso
python run.py --quick ## this script is for training the model
python custom_inference.py ## this script is for inferring the target omics using the trained model
```

# We also uploaded the checkpoints of training the models in case you don't want to train the models from scratch. you can download the checkpoints via: 
```bash
wget "  " -O AllCheckpoints.zip
```
# After unzipping the downloaded file, please copy the corresponding checkpoint folder to the folder of each model. Then you just need to run the custom_inference.py script for data inference.


# Run the Ensemble Clustering module of CSIE

```bash
cd CSIE/EnsembleClustering
Rscript Run_CSIE.R --no-save
```

# Run Comparison methods
```bash
cd CSIE/ComparisonMethods
Rscript Run_Others.R --no-save
```

## 4. Calculate the performance metrics
```bash
cd CSIE/PerformanceMetrics

# Cox p-values and numbers of clusters
Rscript GetCoxPv.R --no-save

# Empirical p-values and numbers of clusters
Rscript GetEmpPv.R --no-save

# C-Indices
Rscript GetData_SP.R --no-save
Rscript trainpredict_all_SP.R --no-save
python Eval_Methods.py
```
