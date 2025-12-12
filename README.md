# CSIE

This repo contains the source code and scripts for reproducing the results reported in our paper "CSIE: Cancer Subtyping via Inference and Ensemble"
Please follow the below instructions:

## 1. Requirements

### Core Requirements (CSIE only)

To run CSIE you need an R-installed environment with the following packages. **Note:** The specified versions are required if you want to reproduce results of DSCC reported in the paper.

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

To run 13 comparison methods you need to install additional packages to the R environment. You also need a Python environment (we used python=3.11) with several packages. If you're using conda, you can directly install Python packages to the environment with R.

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
- MOVICS (`xlucpu/MOVICS`)
- CIMLR (`danro9685/CIMLR`)
- NNLM (`linxihui/NNLM`)
- NEMO (`Shamir-Lab/NEMO/NEMO`)
- LRACluster (`Zaoqu-Liu/LRAcluster`)

#### Python packages

Install using pip or conda:
- numpy
- pandas
- lifelines
- pycox
- scikit-survival
- torch
- matplotlib
- scikit-learn
   
## 2. Setup
```bash
# Clone the repository
git clone https://github.com/tinnlab/DSCC.git
cd DSCC

# create result folders
mkdir Subtyping_Results
mkdir SubSurvClin
mkdir SP_Results

# create a data folder
mkdir Data
cd /Data

# Download processed data
# Download data for DSCC
wget "https://seafile.tinnguyen-lab.com/f/9aa42f40d78247bc95db/?dl=1" -O DSCC_Main.zip
wget "https://seafile.tinnguyen-lab.com/f/2ad884778b00467ea690/?dl=1" -O DSCC_Relevant.zip

# Download data for comparison methods
wget "https://seafile.tinnguyen-lab.com/f/7c943f8328b3452db479/?dl=1" -O Others_Main.zip
wget "https://seafile.tinnguyen-lab.com/f/5f1d638d32a641d08e56/?dl=1" -O Others_Relevant.zip

# unzip the downloaded files
```

## 3. Run analysis for DSCC and comparisons method
```bash
# Please check the scripts for required packages before running them
# Run DSCC
Rscript Run_DSCC.R --no-save

# Run Comparison methods
Rscript Run_Other.R --no-save
```

## 4. Calculate the performance metrics
```bash
# Cox p-values and numbers of clusters
Rscript GetCoxPv.R --no-save

# Empirical p-values and numbers of clusters
Rscript GetEmpPv.R --no-save

# C-Indices
Rscript GetData_SP.R --no-save
Rscript trainpredict_all_SP.R --no-save
python Eval_Methods.py
```
