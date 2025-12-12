### you need to copy the path to python environment for several methods; search for "path-to-python-env"

options(mc.cores = 1)
RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1, VECLIB_MAXIMUM_THREADS = 1, NUMEXPR_NUM_THREADS = 1)

# Wrapper to run each method
suppressMessages({ suppressWarnings({
  library(ANF)
  library(PMA)
  library(IntNMF)
  library(wordspace)
  library(kernlab)
  library(tidyverse)
  library(survival)
  library(matrixStats)
  library(ConsensusClusterPlus)
  library(CIMLR)
  library(SNFtool)
  library(NEMO)
  library(NNLM)
  library(polycor)
  library(psych)
  library(FactoMineR)
  library(pbmcapply)
  library(LRAcluster)
  library(cluster)
  library(survminer)
  library(dplyr)
  library(reticulate)
  library(callr)
  library(SIMLR)
  library(parallel)
  library(Matrix)
  library(MASS)
  library(quadprog)
  library(Rtsne)
}) })

Sys.setenv(CUDA_VISIBLE_DEVICES = sample(0:5, 1))
source("./R/hMKL/load_hMKL.R")
source("./R/MDICC/Functions.R")
LRAClus_prob <- readRDS("../Data/Others_Relevant/LRAClus_meth.rds")

standardNormalization <- function(x) {
  x = as.matrix(x)
  mean = apply(x, 2, mean)
  sd = apply(x, 2, sd)
  sd[sd == 0] = 1
  xNorm = t((t(x) - mean) / sd)
  return(xNorm)
}

listNorm <- function(dataList) {
  newdataList <- lapply(names(dataList), function(dataType) {
    data <- dataList[[dataType]]
    if (!dataType %in% c("meth450", "clinicalMCA")) {
      if (min(data) >= 0) {
        data <- standardNormalization(data)
      }
    }
    return(as.matrix(data))
  }) %>% `names<-`(names(dataList))
  return(newdataList)
}


#### CC ####
runCC <- function(dataList) {

  triangle = function(m, mode = 1) {
    #mode=1 for CDF, vector of lower triangle.
    #mode==3 for full matrix.
    #mode==2 for calcICL; nonredundant half matrix coun
    #mode!=1 for summary
    n = dim(m)[1]
    nm = matrix(0, ncol = n, nrow = n)
    fm = m


    nm[upper.tri(nm)] = m[upper.tri(m)] #only upper half

    fm = t(nm) + nm
    diag(fm) = diag(m)

    nm = fm
    nm[upper.tri(nm)] = NA
    diag(nm) = NA
    vm = m[lower.tri(nm)]

    if (mode == 1) {
      return(vm) #vector
    }else if (mode == 3) {
      return(fm) #return full matrix
    }else if (mode == 2) {
      return(nm) #returns lower triangle and no diagonal. no double counts.
    }
  }

  maxK <- 10


  dataList <- listNorm(dataList)
  d = do.call(cbind, dataList)

  d = standardNormalization(t(d))
  set.seed(1234)
  result_All = ConsensusClusterPlus(d, maxK = maxK, reps = 50, pItem = 0.8, pFeature = 1, clusterAlg = "hc", distance = "pearson", plot = NULL, seed = 1)

  k = maxK
  breaks = 100
  areaK = c()
  for (i in 2:k) {
    v = triangle(result_All[[i]]$ml, mode = 1)

    #empirical CDF distribution. default number of breaks is 100
    h = hist(v, plot = FALSE, breaks = seq(0, 1, by = 1 / breaks))
    h$counts = cumsum(h$counts) / sum(h$counts)

    #calculate area under CDF curve, by histogram method.
    thisArea = 0
    for (bi in 1:(length(h$breaks) - 1)) {
      thisArea = thisArea + h$counts[bi] * (h$breaks[bi + 1] - h$breaks[bi]) #increment by height by width
      bi = bi + 1
    }
    areaK = c(areaK, thisArea)
  }

  #plot area under CDF change.
  deltaK = areaK[1] #initial auc at k=2
  for (i in 2:(length(areaK))) {
    #proportional increase relative to prior K.
    deltaK = c(deltaK, (areaK[i] - areaK[i - 1]) / areaK[i - 1])
  }

  k <- which.max(deltaK[2:(k - 1)]) + 2 #minimun no cluster is 3, first pos is not calculated (k=2)
  cluster <- result_All[[k]]$consensusClass

  cluster
}

#### SNF ####
runSNF <- function(dataList, norm = T) {
  ## First, set all the parameters:
  K = 10; ##number of neighbors, usually (10~30)
  alpha = 0.5; ##hyperparameter, usually (0.3~0.8)
  NIT = 20; ###Number of Iterations, usually (10~20)

  # Normalization
  if (norm) {
    dataList <- listNorm(dataList)
  }
  PS <- lapply(dataList, function(dat) SNFtool::dist2(as.matrix(dat), as.matrix(dat)))

  # Add code to handle the problem of low number of samples
  minSamp <- min(lapply(dataList, nrow) %>% do.call(what = c))
  if (minSamp <= 20) {
    K <- minSamp - 1
  }

  W <- lapply(PS, function(ps) affinityMatrix(ps, K, alpha))
  # W = SNF(W, K, NIT)
  if (length(W) > 1) {
    W = SNF(W, K, NIT)
  } else {
    W = W[[1]]
  }

  #####Clustering
  #Groups with SNF
  C = estimateNumberOfClustersGivenGraph(W, 2:10)[[1]]

  set.seed(1234)
  groupSNF = spectralClustering(W, C); #the final subtypes information
  names(groupSNF) <- rownames(W)
  groupSNF
}

#### CIMLR ####
runCIMLR <- function(dataList) {
  NUMC = 2:10
  dataList <- listNorm(dataList)
  dataList <- lapply(dataList, function(data) {
    data[data > 10] <- 10
    data[data < -10] <- -10
    t(data)
  })

  set.seed(1234)
  k <- CIMLR_Estimate_Number_of_Clusters(dataList, NUMC, cores.ratio = 0)
  k <- NUMC[which.min(k$K1)]

  # closeAllConnections()

  set.seed(1234)
  result <- CIMLR(dataList, k, cores.ratio = 0)
  cluster <- result$y$cluster
  names(cluster) <- colnames(dataList[[1]])
  cluster
}

### LRACluster ###
runLRACluster <- function(dataList, procmeth = T) {
  if (procmeth) {
    if ("meth450" %in% names(dataList)) {
      tmp <- dataList$meth450
      if(length(grep("cg", colnames(tmp))) > 0){
        dataList$meth450 <- tmp[, colnames(tmp) %in% LRAClus_prob]
      }
      rm(tmp)
    }
  }

  dataList <- lapply(dataList, t)
  types = c()
  for (data in dataList) {
    if (all(data %in% 0:1)) {
      types = c(types, "binary")
    }
    else {
      types = c(types, "gaussian")
    }
  }

  rlist <- LRAcluster::LRAcluster(data = dataList, types = types)
  df <- t(rlist$coordinate)

  silhouette_score <- function(k) {
    set.seed(1234)
    km <- kmeans(df, centers = k)
    ss <- silhouette(km$cluster, dist(df))
    mean(ss[, 3])
  }

  k <- 2:10
  avg_sil <- sapply(k, silhouette_score)

  set.seed(1234)
  clst <- kmeans(df, which.max(avg_sil) + 1)
  clst$cluster
}

### IntNMF ###
runIntNMF <- function(dataList) {

  # keep maximum 2000 genes
  maxGeneNo = 2000
  dataList <- lapply(dataList, function(dat) {
    dat <- as.matrix(dat)
    if (ncol(dat) > maxGeneNo) {
      sds <- colSds(dat)
      dat <- dat[, order(sds, decreasing = T)[1:maxGeneNo]]
    }
    dat
  })
  # making negative to positive
  dataList <- lapply(dataList, function(dat) {
    if (min(dat) < 0) {
      dat <- (dat - min(dat)) # make it non negative
    }
    dat[, colSds(dat) > 0 & colSums(dat) > 0] + .Machine$double.eps
  })

  # Finding the optimum number of cluster
  set.seed(1234)
  kList <- IntNMF::nmf.opt.k(dataList, n.runs = 5, n.fold = 5, k.range = 2:10, make.plot = F)

  k <- which.max(rowMeans(kList)) + 1

  set.seed(1234)
  clusters <- IntNMF::nmf.mnnals(dataList, k = k)$clusters

  clusters
}

### ANF ###
runANF <- function(dataList) {

  ## First, set all the parameters:
  K = 10; ##number of neighbors, usually (10~30)

  # Normalization
  dataList <- listNorm(dataList)
  PS <- lapply(dataList, function(dat) SNFtool::dist2(as.matrix(dat), as.matrix(dat)))
  WList <- lapply(PS, function(ps) ANF::affinity_matrix(ps, K))
  set.seed(1234)
  W <- ANF::ANF(WList, K = K)
  C = estimateNumberOfClustersGivenGraph(W, 2:10)[[1]]
  set.seed(1234)
  cluster = spectral_clustering(W, C); #the final subtypes information
  names(cluster) <- rownames(W)
  cluster

}

### NEMO ###
runNEMO <- function(dataList) {

  nemo.affinity.graph <- function(raw.data, k = NA) {
    if (is.na(k)) {
      k = as.numeric(lapply(1:length(raw.data), function(i) round(ncol(raw.data[[i]]) / NUM.NEIGHBORS.RATIO)))
    } else if (length(k) == 1) {
      k = rep(k, length(raw.data))
    }

    sim.data = lapply(1:length(raw.data), function(i) { affinityMatrix(SNFtool::dist2(as.matrix(t(raw.data[[i]])),
                                                                             as.matrix(t(raw.data[[i]]))), k[i], 0.5) })
    affinity.per.omic = lapply(1:length(raw.data), function(i) {
      sim.datum = sim.data[[i]]
      non.sym.knn = apply(sim.datum, 1, function(sim.row) {
        returned.row = sim.row
        threshold = sort(sim.row, decreasing = T)[k[i]]
        returned.row[sim.row < threshold] = 0
        row.sum = sum(returned.row)
        returned.row[sim.row >= threshold] = returned.row[sim.row >= threshold] / row.sum
        return(returned.row)
      })
      sym.knn = non.sym.knn + t(non.sym.knn)
      return(sym.knn)
    })
    patient.names = Reduce(union, lapply(raw.data, colnames))
    num.patients = length(patient.names)
    returned.affinity.matrix = matrix(0, ncol = num.patients, nrow = num.patients)
    rownames(returned.affinity.matrix) = patient.names
    colnames(returned.affinity.matrix) = patient.names

    shared.omic.count = matrix(0, ncol = num.patients, nrow = num.patients)
    rownames(shared.omic.count) = patient.names
    colnames(shared.omic.count) = patient.names

    for (j in 1:length(raw.data)) {
      curr.omic.patients = colnames(raw.data[[j]])
      returned.affinity.matrix[curr.omic.patients, curr.omic.patients] = returned.affinity.matrix[curr.omic.patients, curr.omic.patients] + affinity.per.omic[[j]][curr.omic.patients, curr.omic.patients]
      shared.omic.count[curr.omic.patients, curr.omic.patients] = shared.omic.count[curr.omic.patients, curr.omic.patients] + 1
    }

    final.ret = returned.affinity.matrix / shared.omic.count
    lower.tri.ret = final.ret[lower.tri(final.ret)]
    final.ret[shared.omic.count == 0] = mean(lower.tri.ret[!is.na(lower.tri.ret)])

    return(final.ret)
  }

  dataList <- listNorm(dataList)
  dataList <- lapply(dataList, function(data) {
    t(data)
  })
  graph <- nemo.affinity.graph(dataList)
  num.clusters <- nemo.num.clusters(graph, 2:10)
  clustering <- spectralClustering(graph, num.clusters)
  names(clustering) <- colnames(graph)
  clustering
}

### run moCluster ###
runmoCluster <- function(dataList) {
  dataList <- lapply(dataList, t)
  set.seed(1234)
  moa <- mbpca(dataList, ncomp = 10, method = "globalScore")
  B = 100
  while (B > 0) {
    set.seed(1234)
    r <- try({ bootMbpca(moa, plot = F, B = B, mc.cores = 4) })
    B <- B - 10
    if (!is(r, "try-error")) {
      break()
    }
  }

  if (is(r, "try-error")) {
    stop("Error when running bootMbpca")
  }
  # r <- bootMbpca(moa, plot = F)
  ncomp = which(colMeans(r) > moa@eig)[1] - 1
  if (ncomp < 2) {
    ncomp <- 2
  }

  set.seed(1234)
  moas <- mbpca(dataList, ncomp = ncomp, method = "globalScore")
  scrs <- moaScore(moas)

  set.seed(1234)
  gap <- moGap(moas, K.max = 10, cluster = "hcl", plot = F)

  g <- gap$Tab[, 3]
  gse <- c(gap$Tab[, 3] - gap$Tab[, 4], 0)[-1]

  candidateK <- which(g > gse)
  k = candidateK[which(candidateK > 1)][1]

  set.seed(1234)
  hcl <- hclust(dist(scrs))
  cutree(hcl, k = k)
}


### run MRGCN ###
runMRGCN <- function(dataList, conda_env = "path-to-python-env") {
  result <- r(function(dataList, conda_env) {
    library(tidyverse)
    library(reticulate)
    reticulate::use_condaenv(conda_env)
    reticulate::source_python("./Python/MRGCN/train.py")

    standardNormalization <- function(x) {
      x = as.matrix(x)
      mean = apply(x, 2, mean)
      sd = apply(x, 2, sd)
      sd[sd == 0] = 1
      xNorm = t((t(x) - mean) / sd)
      return(xNorm)
    }

    listNorm <- function(dataList) {
      newdataList <- lapply(names(dataList), function(dataType) {
        data <- dataList[[dataType]]
        if (!dataType %in% c("meth450", "clinicalMCA")) {
          if (min(data) >= 0) {
            data <- standardNormalization(data)
          }
        }
        return(as.matrix(data))
      }) %>% `names<-`(names(dataList))
      return(newdataList)
    }

    dataList <- listNorm(dataList)
    patients <- rownames(dataList[[1]])
    names(dataList) <- NULL

    run_cancer_clustering <- function(dataList, seed = 1234) {
      # Validate input
      if (!is.list(dataList) || length(dataList) == 0) {
        stop("dataList must be a non-empty list of data matrices/arrays")
      }

      # Data validation and conversion
      for (i in seq_along(dataList)) {
        # cat("  Modality", i, ":", paste(dim(dataList[[i]]), collapse=" x "))
        # cat(", class:", class(dataList[[i]]))
        # cat(", type:", typeof(dataList[[i]]))

        # Check for any problematic values
        if (any(is.na(dataList[[i]]))) {
          cat(", contains NA values")
          # Replace NA with 0
          dataList[[i]][is.na(dataList[[i]])] <- 0
        }

        if (any(is.infinite(dataList[[i]]))) {
          cat(", contains infinite values")
          # Replace infinite values with 0
          dataList[[i]][is.infinite(dataList[[i]])] <- 0
        }

        # Ensure data is numeric matrix/array
        if (!is.numeric(dataList[[i]])) {
          cat(", converting to numeric")
          dataList[[i]] <- as.numeric(dataList[[i]])
        }

        # Ensure it's a matrix
        if (!is.matrix(dataList[[i]])) {
          cat(", converting to matrix")
          dataList[[i]] <- as.matrix(dataList[[i]])
        }

        # Explicitly convert to double (R's default numeric type)
        storage.mode(dataList[[i]]) <- "double"

        cat("\n")
      }

      # Validate that all matrices have the same number of rows
      # n_samples <- nrow(dataList[[1]])
      # for (i in 2:length(dataList)) {
      #   if (nrow(dataList[[i]]) != n_samples) {
      #     stop(paste("All data modalities must have the same number of samples.",
      #                "Modality 1 has", n_samples, "samples, but modality", i,
      #                "has", nrow(dataList[[i]]), "samples."))
      #   }
      # }

      # cat("Data validation completed. All modalities have", n_samples, "samples.\n")

      # Use do.call to pass variable number of arguments to Python function
      # The data gets automatically converted from R to Python numpy arrays
      tryCatch({
        results <- do.call(run_clustering_with_data, c(dataList, list(seed = seed)))

        # Convert results back to R format
        r_results <- list(
          cluster_assignments = as.vector(results$cluster_assignments),
          soft_assignments = results$soft_assignments,
          shared_representation = results$shared_representation,
          cluster_centers = results$cluster_centers,
          n_clusters = results$n_clusters,
          n_modalities = results$n_modalities
        )

        cat("Clustering completed with", r_results$n_clusters, "clusters\n")

        return(r_results$cluster_assignments + 1)

      }, error = function(e) {
        cat("Error occurred during clustering:\n")
        cat("Error message:", conditionMessage(e), "\n")
        cat("Calling py_last_error() for more details:\n")
        print(reticulate::py_last_error())
        stop(e)
      })
    }

    cluster <- run_cancer_clustering(dataList)
    names(cluster) <- patients
    return(cluster)
  }, args = list(dataList = dataList, conda_env = conda_env))
  return(result)
}


### run hMKL ###
runhMKL <- function(dataList) {
  dataList <- listNorm(dataList)
  dataList <- lapply(dataList, function(data) {
    t(data)
  })

  dataList_noc <- lapply(dataList, function(data) {
    data_noc <- CIMLR_noc(data, cores.ratio = 0)
  }) %>% `names<-`(names(dataList))

  Ss_dataList_noc <- lapply(dataList_noc, function(data) {
    SS_data_noc <- data$S
  }) %>% `names<-`(names(dataList_noc))

  set_input <- function(kernel_mat) {
    output <- list(kernel = kernel_mat)
    class(output) <- "kernel"
    return(output)
  }

  input <- lapply(Ss_dataList_noc, function(data) {
    input_kernel_noc <- set_input(data)
  }) %>% `names<-`(names(Ss_dataList_noc))

  meta.kernel_sparse_noc <- do.call(combine.kernels2, c(input, list(method = "sparse-UMKL", scale = FALSE)))

  NUMC <- 2:10
  res <- SIMLR_Estimate_Clusters_W(meta.kernel_sparse_noc$kernel, NUMC = NUMC)
  full_K <- res$K1
  opt_k <- NUMC[which.min(full_K)]

  set.seed(1234)
  group_sparse_kmeans_noc <- kmeans(meta.kernel_sparse_noc$kernel, opt_k, nstart = 30)
  group_sparse_kmeans <- group_sparse_kmeans_noc$cluster
  group_sparse_kmeans
  names(group_sparse_kmeans) <- colnames(dataList[[1]])
  group_sparse_kmeans
}


### run MDICC ###
runMDICC <- function(dataList) {
  # dataList <- listNorm(dataList)
  dataList <- lapply(dataList, function(data) {
    t(data)
  })
  X <- lapply(dataList, function(data) {
    data <- as.matrix(data)
    data <- scale(data, center = TRUE, scale = TRUE)
    data <- t(data)
    dist <- as.matrix(dist(data))
  })

  aff <- list()
  for (i in 1:length(X)) {
    a <- as.matrix(X[[i]])
    xxx <- testaff(matrix = a)
    aff[[i]] <- xxx
  }

  test <- MDICC(X = aff)
  test_S <- as.matrix(test)

  cluster <- kmean_opt(test_S)
  names(cluster) <- colnames(dataList[[1]])
  return(cluster)
}


### run DLSF ###
runDLSF <- function(dataList, conda_env = "path-to-python-env") {
  result <- callr::r(function(dataList, conda_env) {
    library(tidyverse)
    library(reticulate)
    reticulate::use_condaenv(conda_env)
    reticulate::source_python("./Python/DLSF/run.py")

    # Debug: Check if function is loaded
    cat("Python function available:", exists("deep_multiomics_clustering"), "\n")

    patients <- rownames(dataList[[1]])
    names(dataList) <- NULL

    run_cancer_clustering <- function(dataList) {
      # Data validation and conversion

      if (!is.list(dataList) || length(dataList) == 0) {
        stop("dataList must be a non-empty list of data matrices/arrays")
      }

      # Data validation and conversion
      for (i in seq_along(dataList)) {
        # cat("  Modality", i, ":", paste(dim(dataList[[i]]), collapse=" x "))
        # cat(", class:", class(dataList[[i]]))
        # cat(", type:", typeof(dataList[[i]]))

        # Check for any problematic values
        if (any(is.na(dataList[[i]]))) {
          cat(", contains NA values")
          # Replace NA with 0
          dataList[[i]][is.na(dataList[[i]])] <- 0
        }

        if (any(is.infinite(dataList[[i]]))) {
          cat(", contains infinite values")
          # Replace infinite values with 0
          dataList[[i]][is.infinite(dataList[[i]])] <- 0
        }

        # Ensure data is numeric matrix/array
        if (!is.numeric(dataList[[i]])) {
          cat(", converting to numeric")
          dataList[[i]] <- as.numeric(dataList[[i]])
        }

        # Ensure it's a matrix
        if (!is.matrix(dataList[[i]])) {
          cat(", converting to matrix")
          dataList[[i]] <- as.matrix(dataList[[i]])
        }

        # Explicitly convert to double (R's default numeric type)
        storage.mode(dataList[[i]]) <- "double"

        cat("\n")
      }

      # dataList <- lapply(dataList, function(data){
      #   as.data.frame(data)
      # })

      tryCatch({
        cat("Calling Python function...\n")
        # Fixed function call - use positional argument
        r_results <- deep_multiomics_clustering(dataList)
        # r_results <- r_results + 1
        return(r_results)
      }, error = function(e) {
        cat("Error occurred during clustering:\n")
        cat("Error message:", conditionMessage(e), "\n")
        cat("Calling py_last_error() for more details:\n")
        print(reticulate::py_last_error())
        stop(e)
      })
    }

    cluster <- run_cancer_clustering(dataList)
    cluster <- cluster + 1
    names(cluster) <- patients
    return(cluster)
  }, args = list(dataList = dataList, conda_env = conda_env))

  # Debug: Check final result
  return(result)
}


### run DSIR ###
runDSIR <- function(dataList, conda_env = "path-to-python-env") {
  coeffMat <- r(function(dataList, conda_env) {
    library(tidyverse)
    library(reticulate)
    reticulate::use_condaenv(conda_env)
    reticulate::source_python("./Python/DSIR/train.py")

    standardNormalization <- function(x) {
      x = as.matrix(x)
      mean = apply(x, 2, mean)
      sd = apply(x, 2, sd)
      sd[sd == 0] = 1
      xNorm = t((t(x) - mean) / sd)
      return(xNorm)
    }

    listNorm <- function(dataList) {
      newdataList <- lapply(names(dataList), function(dataType) {
        data <- dataList[[dataType]]
        if (!dataType %in% c("meth450", "clinicalMCA")) {
          if (min(data) >= 0) {
            data <- standardNormalization(data)
          }
        }
        return(as.matrix(data))
      }) %>% `names<-`(names(dataList))
      return(newdataList)
    }

    dataList <- listNorm(dataList)
    names(dataList) <- NULL

    calcoeff <- function(dataList) {
      # Validate input
      if (!is.list(dataList) || length(dataList) == 0) {
        stop("dataList must be a non-empty list of data matrices/arrays")
      }

      # Data validation and conversion
      for (i in seq_along(dataList)) {
        # Check for any problematic values
        if (any(is.na(dataList[[i]]))) {
          cat(", contains NA values")
          # Replace NA with 0
          dataList[[i]][is.na(dataList[[i]])] <- 0
        }

        if (any(is.infinite(dataList[[i]]))) {
          cat(", contains infinite values")
          # Replace infinite values with 0
          dataList[[i]][is.infinite(dataList[[i]])] <- 0
        }

        # Ensure data is numeric matrix/array
        if (!is.numeric(dataList[[i]])) {
          cat(", converting to numeric")
          dataList[[i]] <- as.numeric(dataList[[i]])
        }

        # Ensure it's a matrix
        if (!is.matrix(dataList[[i]])) {
          cat(", converting to matrix")
          dataList[[i]] <- as.matrix(dataList[[i]])
        }

        # Explicitly convert to double (R's default numeric type)
        storage.mode(dataList[[i]]) <- "double"

        cat("\n")
      }

      tryCatch({
        coefmat <- compute_coef_matrix(dataList)
        return(coefmat)

      }, error = function(e) {
        cat("Error occurred during clustering:\n")
        cat("Error message:", conditionMessage(e), "\n")
        cat("Calling py_last_error() for more details:\n")
        print(reticulate::py_last_error())
        stop(e)
      })
    }

    outmat <- calcoeff(dataList)
    return(outmat)
  }, args = list(dataList = dataList, conda_env = conda_env))

  coeffMat <- as.matrix(coeffMat)

  DSIR.num.clusters <- function(W, NUMC=2:10) {
    if (min(NUMC) == 1) {
      warning("Note that we always assume there are more than one cluster.")
      NUMC = NUMC[NUMC > 1]
    }
    W = (W + t(W))/2
    diag(W) = 0
    if (length(NUMC) > 0) {
      degs = rowSums(W)
      degs[degs == 0] = .Machine$double.eps
      D = diag(degs)
      L = D - W
      Di = diag(1/sqrt(degs))
      L = Di %*% L %*% Di
      print(dim(L))
      eigs = eigen(L)
      eigs_order = sort(eigs$values, index.return = T)$ix
      eigs$values = eigs$values[eigs_order]
      eigs$vectors = eigs$vectors[, eigs_order]
      eigengap = abs(diff(eigs$values))
      eigengap = (1:length(eigengap)) * eigengap
      
      t1 <- sort(eigengap[NUMC], decreasing = TRUE, index.return = T)$ix
      return(NUMC[t1[1]])
    }
  }

  optk <- DSIR.num.clusters(coeffMat)
  S <- 0.5*(coeffMat + t(coeffMat))

  cluster <- spectralClustering(S, optk)
  names(cluster) <- rownames(dataList[[1]])
  return(cluster)
}

