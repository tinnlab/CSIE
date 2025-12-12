source("./Others_helper.R")
library(data.table)
library(matrixStats)
library(mltools)
library(survminer)
library(metap)

dataPath <- "../Data/Others_Main"
ptoPath <- "../Data/Others_Relevant"
resPath <- "../PerformanceMetrics/Subtyping_Results/"

datasets <- list.files(dataPath)
datasets <- strsplit(datasets, ".rds")
datasets <- lapply(datasets, function(elm) { elm[1] }) %>% unlist()

rmTypes <- c("clinical", "survival", "clinicalImputedV2", "clinicalMCA", "clinical_Clean", "clinicalImputedV3", "clinicalMCAV2")

ptoTypes <- c("PTO_Sum_SpectralCounts", "PTO_Sum_DistinctPeptides", "PTO_Sum_UnsharedPeptides", "PTO_Peptides_SpectralCount",
              "PTO_Sum_SpectralCounts_JHU", "PTO_Sum_DistinctPeptides_JHU", "PTO_Sum_UnsharedPeptides_JHU", "PTO_Peptides_SpectralCount_JHU")

if (!file.exists(resPath)) {
  dir.create(resPath)
}

methods <- c("CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF", "NEMO", "MRGCN", "hMKL", "MDICC", "DLSF", "DSIR")

intersect_methods <- c("CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF")
union_methods <- c("NEMO", "MRGCN")

for (dataset in datasets) {
# mclapply(datasets, mc.cores=length(datasets), function(dataset){
  (function() {
    message("Subtyping on: ", dataset)
    rds <- readRDS(file.path(dataPath, paste0(dataset, ".rds")))

    survival <- rds$survival
    if("OStime" %in% colnames(survival)){
      survival <- survival[, c("OStime", "OSstatus")]
      colnames(survival) <- c("os", "isDead")
    }

    survival <- survival[rowSums(is.na(survival)) == 0,]
    survival <- survival[survival$os >= 0,]
    survival$os[survival$os == 0] <- 1

    toUseDataList <- rds[setdiff(names(rds), c("survival"))]
    toUseDataList <- toUseDataList[!names(toUseDataList) %in% rmTypes]
    toUseDataList <- toUseDataList[!sapply(toUseDataList, is.null)]

    if (file.exists(file.path(ptoPath, paste0(dataset, ".rds")))) {
      ptodata <- readRDS(file.path(ptoPath, paste0(dataset, ".rds")))
      ptodata <- ptodata[names(ptodata) %in% ptoTypes]
      toUseDataList <- c(toUseDataList, ptodata)
    }

    toUseDataList <- lapply(toUseDataList, function(data) {
      data[is.na(data)] <- 0
      data <- data[rowSums(data) != 0, ]
      data
    }) %>% `names<-`(names(toUseDataList))

    rm(rds)
    # gc()
    # mclapply(methods, mc.cores=5, function(method){
    for (method in methods) {
      message(method, "\t", dataset)
      # if (file.exists(file.path(resPath, paste0(method, "-", dataset, ".rds")))) {
      #   next
      # }
      (function() {

        if (method %in% union_methods) {
          unionSamples <- Reduce(c, lapply(toUseDataList, rownames)) %>% unique()
          toUseSamples <- intersect(unionSamples, rownames(survival))

          toUseDataList <- lapply(toUseDataList, function(data) {
            sampkeep <- rownames(data)[rownames(data) %in% toUseSamples]
            if (length(sampkeep) < 5) {
              data <- NULL
            }else {
              data <- data[sampkeep,]
            }
            data
          }) %>% `names<-`(names(toUseDataList))

          toUseDataList <- toUseDataList[!sapply(toUseDataList, is.null)]
          unionSamples <- Reduce(c, lapply(toUseDataList, rownames)) %>% unique()
          toUseSamples <- intersect(unionSamples, rownames(survival))

          if (method == "MRGCN") {
            toUseDataList <- lapply(toUseDataList, function(data) {
              missingSamples <- setdiff(toUseSamples, rownames(data))
              if (length(missingSamples) > 0) {
                missingData <- matrix(0,
                                      nrow = length(missingSamples),
                                      ncol = ncol(data),
                                      dimnames = list(missingSamples, colnames(data)))
                if (is.data.frame(data)) {
                  missingData <- as.data.frame(missingData)
                }
                data <- rbind(data, missingData)
              }
              data <- data[toUseSamples, , drop = FALSE]
              return(data)
            }) %>% `names<-`(names(toUseDataList))
          }
        }else {
          intersectSamples <- Reduce(intersect, lapply(toUseDataList, rownames)) %>% unique()
          toUseSamples <- intersect(intersectSamples, rownames(survival))

          toUseDataList <- lapply(toUseDataList, function(data) {
            sampkeep <- rownames(data)[rownames(data) %in% toUseSamples]
            sampkeep <- sort(sampkeep, decreasing = T)
            if (length(sampkeep) < 5) {
              data <- NULL
            }else {
              data <- data[sampkeep,]
            }
            data
          }) %>% `names<-`(names(toUseDataList))

          toUseDataList <- toUseDataList[!sapply(toUseDataList, is.null)]
          intersectSamples <- Reduce(intersect, lapply(toUseDataList, rownames)) %>% unique()
          toUseSamples <- intersect(intersectSamples, rownames(survival))
        }


        toUseDataList <- lapply(names(toUseDataList), function(dataType) {
          df <- as.matrix(toUseDataList[[dataType]])
          if (!dataType %in% c("clinical", "clinicalImputedV2", "clinicalMCA")) {
            if (min(df) >= 0) {
              if (max(df) > 100) {
                df <- log2(df + 1)
              }
            }

            df <- df[, colVars(df) > 0, drop = FALSE]
            df <- df[, order(colVars(df), decreasing = TRUE)[1:min(8000, ncol(df))]]
          }
          df
        }) %>% `names<-`(names(toUseDataList))

        resFile <- paste0(resPath, method, "-", dataset, ".rds")
        runningTime <- Sys.time()
        cluster <- try({ get(paste0("run", method))(toUseDataList) })
        ncluster <- max(cluster[toUseSamples])
        runningTime <- Sys.time() - runningTime
        if (!is(cluster, "try-error")) {
          coxFit <- try({ summary(coxph(Surv(time = os, event = isDead) ~ as.factor(cluster[toUseSamples]), data = survival[toUseSamples,], ties = "exact")) })
          if (!is(coxFit, "try-error")) {
            pval <- round(coxFit$sctest[3], digits = 40)
            coxFit <- summary(coxFit)
          } else {
            pval <- NA
            coxFit <- NA
          }
          message("Result for", "\t", dataset, "\t", pval, "\t", ncluster)


          saveRDS(
            list(
              cluster = cluster[toUseSamples],
              nclusters = ncluster,
              pval_cox = pval,
              noSamples = length(toUseSamples),
              nDeaths = sum(survival[toUseSamples, "isDead"]),
              noDataTypes = length(toUseDataList),
              coxFit = coxFit,
              dataset = dataset,
              method = method,
              runningTime = runningTime
            ), file = resFile)
        } else {
          saveRDS(
            list(
              cluster = NA,
              nclusters = NA,
              pval_cox = NA,
              noSamples = length(toUseSamples),
              nDeaths = sum(survival[toUseSamples, "isDead"]),
              noDataTypes = length(toUseDataList),
              coxFit = NA,
              dataset = dataset,
              method = method,
              runningTime = runningTime
            ), file = resFile)
          
          print("error occured")
        }
      })()
      gc()
    # })
    }

  })()
  gc()
# })
}

