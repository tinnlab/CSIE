library(tidyverse)
library(parallel)
library(scales)
library(ggplot2)
library(tidyr)
library(matrixStats)
library(grid)
library(gridExtra)

resPath <- "./Subtyping_Results"
datasets <- list.files(resPath)
datasets <- datasets[grep("CSIE-", datasets)]
datasets <- gsub("CSIE-", "", datasets)
datasets <- strsplit(datasets, ".rds")
datasets <- lapply(datasets, function(elm) { elm[1] }) %>% unlist()

methods <- c("CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF", "NEMO", "MRGCN", "hMKL", "MDICC", "DLSF", "DSIR")

allRes_methods <- lapply(methods, function(method) {

  allRes <- lapply(datasets, function(dataset) {
    ResList <- readRDS(file.path(resPath, paste0(method, "-", dataset, ".rds")))
  }) %>% `names<-`(datasets)

  ResTable <- lapply(allRes, function(ResList) {
    data.frame(pvalue_cox = ResList$pval_cox,
               ncluster = ResList$ncluster,
               nSamples = ResList$noSamples,
               nDeath = ResList$nDeaths)
  }) %>%
    do.call(what = rbind) %>%
    `rownames<-`(names(allRes)) %>%
    as.data.frame()

  ResTable
}) %>% `names<-`(methods)

pva_cox_tab <- lapply(allRes_methods, function(table) {
  table$pvalue_cox
}) %>%
  do.call(what = cbind) %>%
  `colnames<-`(names(allRes_methods)) %>%
  `rownames<-`(rownames(allRes_methods[[1]]))


# ## number of clusters
nclus_tab <- lapply(allRes_methods, function(table) {
  table$ncluster
}) %>%
  do.call(what = cbind) %>%
  `colnames<-`(names(allRes_methods)) %>%
  `rownames<-`(rownames(allRes_methods[[1]]))

format_scientific_vector <- function(nums, digits = 2) {
  formatted <- sapply(nums, function(num) {
    fmt <- format(num, scientific = TRUE, digits = digits)
    fmt <- gsub("e", "E", fmt)
    return(fmt)
  })
  return(formatted)
}

pval_nclus_tab <- lapply(1:ncol(nclus_tab), function(i) {
  pval_col <- pva_cox_tab[, i]
  pval_col[!is.na(pval_col)] <- format_scientific_vector(pval_col[!is.na(pval_col)])
  pval_col[is.na(pval_col)] <- "NA"
  nclus_col <- nclus_tab[, i]
  nclus_col[is.na(nclus_col)] <- "NA"
  cb <- paste0(pval_col, " (", nclus_col, ")")
}) %>%
  do.call(what = cbind) %>%
  `colnames<-`(colnames(nclus_tab)) %>%
  `rownames<-`(rownames(nclus_tab))

print(pval_nclus_tab)


