RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1, VECLIB_MAXIMUM_THREADS = 1, NUMEXPR_NUM_THREADS = 1)

# Wrapper to run each method
suppressMessages({ suppressWarnings({
  library(tidyverse)
  library(survival)
  library(matrixStats)
  library(parallel)
}) })

ResPath <- "./Subtyping_Results"
datPath <- "../Data/CSIE_Main"
methods <- c("CSIE", "CC", "CIMLR", "SNF", "LRACluster", "IntNMF", "ANF", "NEMO", "MRGCN", "hMKL", "MDICC", "DLSF", "DSIR")

alldts <- list.files(ResPath)
alldts <- alldts[grep("CSIE", alldts)]
alldts <- gsub("CSIE-", "", alldts)
alldts <- gsub(".rds", "", alldts)

AllEmP <- lapply(alldts, function(dts) {
  print(dts)
  datards <- readRDS(file.path(datPath, paste0(dts, ".rds")))
  survdat <- datards$survival
  if("OStime" %in% colnames(survdat)){
    survdat <- survdat[, c("OStime", "OSstatus")]
    colnames(survdat) <- c("os", "isDead")
  }

  rm(datards)
  mclapply(methods, mc.cores=length(methods), function(method) {
    print(method)
    resdat <- readRDS(file.path(ResPath, paste0(method, "-", dts, ".rds")))
    cluster <- resdat$cluster
    rm(resdat)

    toUseSamples <- intersect(names(cluster), rownames(survdat))
    # coxFit <- try({ summary(coxph(Surv(time = os, event = isDead) ~ as.factor(cluster[toUseSamples]), data = survdat[toUseSamples,], ties = "exact")) })
    logrank <- try({ survdiff( Surv(time = os, event = isDead) ~ as.factor(cluster[toUseSamples]), data = survdat[toUseSamples,])   })

    if (!is(logrank, "try-error")) {
      # obsstat <- round(coxFit$sctest[1], digits = 40)
      # org_pv <- round(coxFit$sctest[3], digits = 40)
      obsstat <- round(logrank$chisq, digits = 40)
      org_pv <- 1 - pchisq(logrank$chisq, length(logrank$n) - 1)  
    } else {
      obsstat <- NA
      org_pv <- NA
    }

    # rm(coxFit)
    rm(logrank)
    if (is.na(obsstat)){
      em_pv <- NA
    }else{
      # perm1 <- ceiling(min(max(10/org_pv, 1e4), 1e6))
      # perm2 <- min(2e7-perm1, 1e5)
      perm1 <- ceiling(min(1/org_pv, 5e3))
      # perm2 <- 2e4-perm1
      perm2 <- 1e4-perm1
      allperms <- perm1 + perm2

      allpermstat <- mclapply(1:allperms, mc.cores=16, function(i){
        set.seed(i)
        ranids <- sample(1:length(cluster))
        cluster_perm <- cluster
        names(cluster_perm) <- names(cluster)[ranids]
        # coxFit <- try({ summary(coxph(Surv(time = os, event = isDead) ~ as.factor(cluster_perm[toUseSamples]), data = survdat[toUseSamples,], ties = "exact")) })
        logrank <- try({ survdiff( Surv(time = os, event = isDead) ~ as.factor(cluster_perm[toUseSamples]), data = survdat[toUseSamples,])   })

        if (!is(logrank, "try-error")) {
          # permstat <- round(coxFit$sctest[1], digits = 40)
          permstat <- round(logrank$chisq, digits = 40)

        } else {
          permstat <- NA
        }
        names(permstat) <- NULL
        permstat
      }) %>% do.call(what = c)

      allpermstat[is.na(allpermstat)] <- -Inf
      
      pickid <- perm2

      pickpemrstat <- allpermstat[1:(perm1+pickid)]
      em_pv <- sum(pickpemrstat >= obsstat)/length(pickpemrstat)
    }
    print(em_pv)
    em_pv
  }) %>% do.call(what = c)
}) %>% do.call(what = rbind) %>% `rownames<-` (alldts) %>% `colnames<-` (methods)
sum(AllEmP < 0.05, na.rm=T)

write.csv(as.data.frame(AllEmP), "./Emp_Pvalue.csv", row.names=T)


