library(tidyverse)
library(parallel)
library(matrixStats)
library(jsonlite)

datPath <- "../Data/CSIE_Main"
# savePath <- "/data/daotran/Cancer_Subtyping/BiB_Submission/Data/TCGA_csv"
savePath <- "../Data/TCGA_csv"

if(!file.exists(file.path(savePath))){
    dir.create(file.path(savePath))
}

dataTypes <- c("mRNATPM", "miRNA", "miRNAiso", "meth450")

allFiles <- list.files(datPath)
allFiles <- gsub(".rds", "", allFiles)
allFiles <- allFiles[grep("TCGA-", allFiles)]

mclapply(allFiles, mc.cores=8, function(file){
  print(file)
  if(!file.exists(file.path(savePath, file))){
    dir.create(file.path(savePath, file))
  }
  rds <- readRDS(file.path(datPath, paste0(file, ".rds")))
  rds <- rds[names(rds) %in% dataTypes]

  lapply(names(rds), function(dType){
    write.csv(rds[[dType]], file.path(savePath, file, paste0(dType, ".csv")), row.names=T, quote=F)
  })
})

### save the kegg pathway genes as csv
keggGeneSet <- readRDS("../Data/CSIE_Relevant/KEGGPathways.rds")
keggGeneSet <- lapply(names(keggGeneSet), function(name){
  gs <- keggGeneSet[[name]]
  gs[gs == ""] <- NA
  gs[gs == "NA"] <- NA
  gs[gs == "NULL"] <- NA
  gs <- gs[!is.na(gs)]
}) %>% `names<-` (names(keggGeneSet))
# genecount <- sapply(keggGeneSet, length)
# keggGeneSet <- keggGeneSet[genecount >= 20]
# genecount <- genecount[genecount >= 20]
# genecount <- genecount[genecount <= 300 & genecount >= 100]
# write_json(keggGeneSet[names(genecount)], "/nfs/blanche/share/daotran/Subtyping/data-analysis/NewKEGGgs.json", pretty = TRUE)
write_json(keggGeneSet, "../Data/CSIE_Relevant/NewKEGGgs.json", pretty = TRUE)