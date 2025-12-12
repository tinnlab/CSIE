source("./CSIE_helper.R")

processedDataPath <- "./Data/CSIE_Main/"
ptoPath <- "./Data/CSIE_Relevant"
resPath <- "./Subtyping_Results/"

datasets <- list.files(processedDataPath)
datasets <- strsplit(datasets, ".rds") %>% do.call(what = c)

rmTypes <- c("clinical", "survival", "clinicalImputedV2", "clinicalMCA", "clinical_Clean", "clinicalImputedV3", "clinicalMCAV2")

ptoTypes <- c("PTO_Sum_SpectralCounts", "PTO_Sum_DistinctPeptides", "PTO_Sum_UnsharedPeptides", "PTO_Peptides_SpectralCount",
              "PTO_Sum_SpectralCounts_JHU", "PTO_Sum_DistinctPeptides_JHU", "PTO_Sum_UnsharedPeptides_JHU", "PTO_Peptides_SpectralCount_JHU")

method <- "CSIE"

if (!file.exists(resPath)) {
  dir.create(resPath)
}

for (dataset in datasets) {
# mclapply(datasets, mc.cores=length(datasets), function(dataset){
  (function() {
    message("Subtyping on: ", dataset)
    rds <- readRDS(file.path(processedDataPath, paste0(dataset, ".rds")))

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

    rm(rds)

    toUseDataList <- lapply(toUseDataList, function(data) {
      data[is.na(data)] <- 0
      data <- data[rowSums(data) != 0, ]
      data
    }) %>% `names<-`(names(toUseDataList))

    toUseDataList <- toUseDataList[!sapply(toUseDataList, is.null)]
    unionSamples <- Reduce(c, lapply(toUseDataList, rownames)) %>% unique()
    toUseSamples <- intersect(unionSamples, rownames(survival))

    toUseDataList <- lapply(toUseDataList, function(data) {
      sampkeep <- rownames(data)[rownames(data) %in% toUseSamples]
      if (length(sampkeep) < 5) {
        data <- NULL
      }else {
        data <- data[sampkeep, ,drop=F]
      }
      data
    }) %>% `names<-`(names(toUseDataList))

    toUseDataList <- toUseDataList[!sapply(toUseDataList, is.null)]
    unionSamples <- Reduce(c, lapply(toUseDataList, rownames)) %>% unique()
    toUseSamples <- intersect(unionSamples, rownames(survival))

    toUseDataList <- lapply(names(toUseDataList), function(dataType) {
      data <- toUseDataList[[dataType]]
      data <- as.matrix(data)

      if (min(data) >= 0) {
        if (max(data) > 100) {
          data <- log2(data + 1)
        }
      }
      data
    }) %>% `names<-`(names(toUseDataList))

    noDeath <- survival[toUseSamples, "OSstatus"]
    noDeath <- sum(noDeath, na.rm = T)
    cat("Finished loading data\n")

    runningTime <- Sys.time()
    resFile <- paste0(resPath, method, "-", dataset, ".rds")

    cluster <- try({ runCSIE(dataList = toUseDataList, nSamples = length(toUseSamples)) })
    runningTime <- Sys.time() - runningTime
    if (!is(cluster, "try-error")) {
      ncluster <- max(cluster[toUseSamples])
      coxFit <- try({ summary(coxph(Surv(time = os, event = isDead) ~ as.factor(cluster[toUseSamples]), data = survival[toUseSamples,], ties = "exact")) })
      if (!is(coxFit, "try-error")) {
        pval <- round(coxFit$sctest[3], digits = 40)
        coxFit <- summary(coxFit)
      } else {
        pval <- NA
        coxFit <- NA
      }
      message(method, "\t", dataset, "\t", pval)
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
    }
  })()
  gc()
# })
}

