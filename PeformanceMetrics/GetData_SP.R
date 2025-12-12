library(tidyverse)
library(parallel)

subPath <- "./Subtyping_Results"
datPath <- "../Data/CSIE_Main"
savePath <- "./SubSurvClin"

datasets <- list.files(datPath)
datasets <- strsplit(datasets, ".rds")
datasets <- lapply(datasets, function(elm) { elm[1] }) %>% unlist()

### remove datasets with no suitable clinical variables
datasets <- setdiff(datasets, c("P23918603", "GSE62452", "GSE74187", "GSE78229", "GSE85916", "GSE57495", "GSE21501", "GSE72951"))

methods <- c("Baseline", "CSIE", "CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF", "NEMO", "MRGCN", "hMKL", "MDICC", "DLSF", "DSIR")
             
lapply(methods, function(method){
  print(method)
  if (!file.exists(file.path(savePath, method))){
    dir.create(file.path(savePath, method))
  }

  mclapply(datasets, mc.cores = 4, function(dataset){
    print(dataset)
    clindat <- readRDS(file.path(datPath, paste0(dataset, ".rds")))
    survival <- clindat$survival
    clindat <- clindat$clinicalImputedV2

    if(method == "Baseline"){
      subtypes <- NULL
    }else{
      subtypes <- readRDS(file.path(subPath, paste0(method, "-", dataset, ".rds")))
      subtypes <- subtypes$cluster

      if(length(subtypes) <= 1){
        subtypes <- NA
      }
    }

    rdsSaved <- list(cluster = subtypes, clinical = clindat, survival = survival)
    saveRDS(rdsSaved, file.path(savePath, method, paste0(dataset, ".rds")))
    return(NULL)
  })
  return(NULL)
})


