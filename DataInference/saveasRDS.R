RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1, VECLIB_MAXIMUM_THREADS = 1, NUMEXPR_NUM_THREADS = 1)

library(tidyverse)
library(matrixStats)
library(parallel)

datPath <- "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TestDataGene/DG4_1"
survPath <- "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/OtherSources/allbutTCGA_mapped"
savePath <- "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TestDataGene/DG4_1_rds"

if(!file.exists(savePath)){
    dir.create(savePath)
}

datasets <- c("GSE57495", "GSE103479", "GSE85916", "GSE72951", "GSE42669", "GSE13041", "GSE21501", "GSE62452", "GSE74187", "GSE61335", "GSE78229", "GSE4412", "GSE71729", "GSE17536", "GSE87211", 
"GSE17537", "GSE150615_2", "GSE1456", "GSE20685", "GSE150615_1")

# datasets <- c("GSE61335")

mclapply(datasets, mc.cores=10, function(dataset){
    print(dataset)
    dtypes <- list.files(file.path(datPath, dataset))
    dtypes <- gsub(".csv", "", dtypes)
    rds <- mclapply(dtypes, mc.cores=length(dtypes), function(dtype){
        data <- read.csv(file.path(datPath, dataset, paste0(dtype, ".csv")), row.names=1)
    }) %>% `names<-` (dtypes)
    survival <- readRDS(file.path(survPath, paste0(dataset, ".rds")))
    survival <- survival$survival
    rds$survival <- survival
    saveRDS(rds, file.path(savePath, paste0(dataset, ".rds")))
})