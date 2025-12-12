### split data into training and testing
### do 5-fold cross validation
### run 5 times
### use blockForest

library(tidyverse)
library(parallel)
library(survival)
library(matrixStats)
library(rsample)
library(blockForest)

datPath <- "./SubSurvClin"
methods <- c("Baseline", "CSIE", "CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF", "NEMO", "MRGCN", "hMKL", "MDICC", "DLSF", "DSIR")
savePath <- "./SP_Results"

allFiles <- list.files(file.path(datPath, methods[1]))
allFiles <- strsplit(allFiles, ".rds")
allFiles <- lapply(allFiles, function(elm) { elm[1] }) %>% unlist()

### remove variables that are meaningless or important to prognosis
rmpatterns <- c("barcode", "status_deceased", "whole_exome_sequenced", "whole_genome_sequenced", "event_distantrecurrence", "event_regionalrecurrence", "event_localrecurrence", "case_project_glss"
, "case_project_tcga", "tissue_source_", "histology", "histological", "grade", "stage", "classification", "group", "risk", "followup", "country", "treatment", "status", "cancer", "type", 
"response", "outcome", "rna", "genome", "subgroup", "patient:ch1_")

lapply(methods, function(method) {
  if (!file.exists(file.path(savePath, method))) {
    dir.create(file.path(savePath, method))
  }
  print(method)

  lapply(allFiles, function(file) {
    print(file)

    if (!dir.exists(file.path(savePath, method, file))) {
      dir.create(file.path(savePath, method, file))
    }

    rds <- readRDS(file.path(datPath, method, paste0(file, ".rds")))
    clin <- rds$clinical

    ### remove some vars

    colnames(clin) <- tolower(colnames(clin))
    allvars <- colnames(clin)
    rmvars <- lapply(rmpatterns, function(pattern){
        rmvar <- allvars[grep(pattern, allvars)]
    }) %>% do.call(what = c) %>% unique()

    allvars <- setdiff(allvars, rmvars)
    clin <- clin[, allvars, drop=F]

    ###

    surv <- rds$surv

    if("OStime" %in% colnames(surv)){
      surv <- surv[, c("OStime", "OSstatus")]
      colnames(surv) <- c("os", "isDead")
    }

    if ("os" %in% colnames(surv)) {
      surv <- surv[, c("os", "isDead")]
      colnames(surv) <- c("time", "status")
    }

    subs <- rds$cluster

    if (length(subs) > 1 | is.null(subs)) {
      cSamples <- intersect(rownames(surv), rownames(clin))

      if (is.null(subs)) {
        clin <- clin[cSamples, 1:ncol(clin), drop = FALSE]
        surv <- surv[cSamples,]
        data <- as.data.frame(clin)
      }else {
        cSamples <- intersect(names(subs), cSamples)
        clin <- clin[cSamples,]
        surv <- surv[cSamples,]
        subs <- subs[cSamples]

        one_hot <- model.matrix(~0 + factor(subs))
        colnames(one_hot) <- gsub("\\(", "", colnames(one_hot))
        colnames(one_hot) <- gsub("\\)", "", colnames(one_hot))
        rownames(one_hot) <- names(subs)

        data <- as.data.frame(cbind(clin, one_hot))
      }

      lapply(c(1:5), function(time) {
        if (!dir.exists(file.path(savePath, method, file, paste0("Time", time)))) {
          dir.create(file.path(savePath, method, file, paste0("Time", time)))
        }

        print(paste0('Running Time: ', time))
        set.seed(time)
        if (sum(surv$status == 0) == 0) {
          all_folds <- vfold_cv(surv, v = 5, repeats = 1)
        }else {
          all_folds <- vfold_cv(surv, v = 5, repeats = 1, strata = surv$status)
        }
        all_folds <- lapply(1:5, function(fold) {
          patientIDs <- rownames(surv)[all_folds$splits[[fold]]$in_id]
        })

        mclapply(c(1:5), mc.cores = 5, function(fold) {
          print(paste0('Running Fold: ', fold))
          trainIndex <- all_folds[[fold]]
          valIndex <- setdiff(rownames(surv), trainIndex)

          data_train <- list(data[trainIndex, 1:ncol(data), drop = FALSE])
          surv_train <- surv[trainIndex,]

          blockIndices <- rep(seq_along(data_train), sapply(data_train, ncol))

          blocks <- lapply(seq_along(data_train), function(i) which(blockIndices == i)) %>% `names<-`(paste0("block", seq_along(data_train)))
          data_train_used <- do.call(cbind, data_train)
          colnames(data_train_used) <- paste0('Var', c(1:ncol(data_train_used)))
          train_label <- Surv(surv_train$time, surv_train$status)

          set.seed(1234)
          bf <- try({ blockfor(data_train_used, train_label, block = blocks, block.method = "BlockForest", num.trees = 200, replace = TRUE,
                               nsets = 20, num.trees.pre = 100, splitrule = "extratrees", num.threads = 1) })

          if (is(bf, "try-error")) {
            predVal <- rep(NA, length(valIndex))
          }else {
            ### fix the problem with blockForest when there is only one predictor in the data
            if (ncol(data_train_used) == 1) {
              bf$forest$forest$independent.variable.names <- "Var1"
            }
            data_val_used <- data[valIndex, 1:ncol(data), drop = FALSE]
            colnames(data_val_used) <- paste0('Var', c(1:ncol(data_val_used)))
            predVal <- predict(bf$forest, data = data_val_used, block.method = "BlockForest")

            survFVal <- predVal$survival %>% `rownames<-`(valIndex)
            colnames(survFVal) <- paste0(rep("predVal", ncol(survFVal)), "_", predVal$unique.death.times)
            predVal <- survFVal
          }
          predVal <- as.data.frame(cbind(predVal, as.matrix(surv[valIndex,]))) %>% `rownames<-`(valIndex)
          write.csv(predVal, file.path(savePath, method, file, paste0('Time', time), paste0("Val_Res_", fold, ".csv")), row.names = T)
        })
      })
    }else {
      lapply(c(1:5), function(time) {
        if (!dir.exists(file.path(savePath, method, file, paste0("Time", time)))) {
          dir.create(file.path(savePath, method, file, paste0("Time", time)))
        }
        mclapply(c(1:5), mc.cores = 5, function(fold) {
          predVal <- as.data.frame(matrix(NA, 50, 2))
          write.csv(predVal, file.path(savePath, method, file, paste0('Time', time), paste0("Val_Res_", fold, ".csv")), row.names = T)
        })
      })
    }
  })
  # write.csv(alltimes, file.path(timerecPath, "blockForest", file, "TimeRec.csv"), row.names=T)
  return(NULL)
})





