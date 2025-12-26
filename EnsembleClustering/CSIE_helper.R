### always use LCA
### add another step of scaling

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1, VECLIB_MAXIMUM_THREADS = 1, NUMEXPR_NUM_THREADS = 1)

# Wrapper to run each method
suppressMessages({ suppressWarnings({
  library(tidyverse)
  library(survival)
  library(matrixStats)
  library(SNFtool)
  library(igraph)
  library(NEMO)
  library(parallel)
  library(cluster)
}) })

clust_to_conn <- function(clustering) {
  n <- length(clustering)
  connectivity_matrix <- matrix(0, nrow = n, ncol = n)

  unique_clusters <- unique(clustering)
  for (cluster in unique_clusters) {
    members <- which(clustering == cluster)
    connectivity_matrix[members, members] <- 1
  }
  return(connectivity_matrix)
}

angular_distance_matrix <- function(data) {
  norms <- sqrt(rowSums(data^2))
  if (any(norms == 0)) {
    stop("Cannot compute angular distance with zero vector")
  }
  normalized_data <- data / norms
  cos_sim_matrix <- normalized_data %*% t(normalized_data)
  cos_sim_matrix <- pmin(pmax(cos_sim_matrix, -1.0), 1.0)
  angle_matrix <- acos(cos_sim_matrix)
  dist_matrix <- angle_matrix / pi
  diag(dist_matrix) <- 0
  return(dist_matrix)
}

correlation_distance_matrix <- function(data) {
  matrix_data <- t(data)
  cor_matrix <- cor(matrix_data)
  dist_matrix <- 1 - cor_matrix
  return(dist_matrix)
}

distance_to_affinity <- function(dist_matrix, sigma = 1) {
  affinity_matrix <- exp(-(dist_matrix^2) / (2 * sigma^2))
  return(affinity_matrix)
}

merge_matrices <- function(matrices) {
  n <- nrow(matrices[[1]])
  k <- length(matrices)
  w <- array(0, dim = c(n, n, k))
  for (i in 1:k) w[, , i] <- matrices[[i]]
  return(w)
}

solve_lambda <- function(alpha, gamma, ini) {

  f <- function(x) {
    a <- rep(1, length(gamma))
    abs((sum(a / ((gamma - x) * alpha))^2) /
          (sum(a / ((gamma - x)^2 * alpha))) - 1)
  }

  k <- length(gamma)
  # Initial x0
  x0 <- if (ini == 1) -1
  else if (ini == k + 2) 1
  else round(gamma[1] * 100) / 100
  # Use optim instead of fsolve (R equivalent)
  set.seed(123)
  opt_result <- optim(x0, f, method = "Brent",
                      lower = -2, upper = 2)
  return(opt_result$par)
}

aasc_eigengap <- function(w, num_clusters = 10) {
  sizeK <- dim(w)[3]  # Number of affinity matrices
  s1 <- dim(w)[1]     # Number of nodes
  s2 <- dim(w)[2]     # Should be equal to s1

  num_clusters <- min(num_clusters, s1)

  # Initialize matrices
  D_k <- array(0, dim = c(s1, s2, sizeK))
  L_k <- array(0, dim = c(s1, s2, sizeK))

  # Initial equal weight
  weight <- rep(1 / sizeK, sizeK)

  # Precompute degree and Laplacian matrices for each input matrix
  for (i in 1:sizeK) {
    # Compute degree matrix
    s <- rowSums(w[, , i])
    D_k[, , i] <- diag(s)
    # Compute Laplacian matrix
    L_k[, , i] <- D_k[, , i] - w[, , i]
  }

  # Parameters
  iter_number <- 20
  best_eigengap <- -Inf  # Changed from best_eigensum to best_eigengap
  best_weights <- weight

  # Main iterative process
  for (iter in 1:iter_number) {
    w_reshaped <- matrix(w, nrow = s1 * s2, ncol = sizeK)
    # Multiply by weights and sum
    w_n_flat <- w_reshaped %*% weight
    # Reshape back to original dimensions
    w_n <- matrix(w_n_flat, nrow = s1, ncol = s2)

    # Compute degree matrix
    D <- diag(rowSums(w_n))

    # Compute normalized Laplacian matrix
    D_inv_sqrt <- diag(1 / sqrt(diag(D) + 1e-10))  # Add small constant to avoid division by zero
    L <- D - w_n
    L_norm <- D_inv_sqrt %*% L %*% D_inv_sqrt

    # Compute eigendecomposition
    eigen_result <- eigen(L_norm, symmetric = TRUE)
    eigenvalues <- eigen_result$values
    eigenvectors <- eigen_result$vectors

    # Sort eigenvalues and eigenvectors
    idx <- order(eigenvalues)
    eigenvalues <- eigenvalues[idx]
    eigenvectors <- eigenvectors[, idx]

    # Calculate all eigengaps (differences between consecutive eigenvalues)
    # We'll look at all possible gaps to find the most significant one
    eigengaps <- abs(diff(eigenvalues))
    eigengaps <- (1:length(eigengaps)) * eigengaps
    eigengap <- max(eigengaps)

    # Keep track of the best combination found - now maximizing eigengap
    if (eigengap > best_eigengap) {
      best_eigengap <- eigengap
      best_weights <- weight
    }

    # Compute coefficients for updating weights
    relevant_eigenvectors <- eigenvectors[, 2:num_clusters, drop = FALSE]

    alpha_k <- sapply(1:sizeK, function(k) {
      D_mult <- D_k[, , k] %*% relevant_eigenvectors
      colSums(relevant_eigenvectors * D_mult)
    })

    # Calculate beta_k values using sapply
    beta_k <- sapply(1:sizeK, function(k) {
      L_mult <- L_k[, , k] %*% relevant_eigenvectors
      colSums(relevant_eigenvectors * L_mult)
    })

    # Calculate gamma_k values in one operation
    gamma_k <- beta_k / alpha_k

    # Average across all eigenvectors of interest to get a combined measure
    alpha <- colMeans(alpha_k)
    gamma <- colMeans(gamma_k)

    # 1-D solution for weight optimization
    weight_1D <- matrix(0, nrow = sizeK + 2, ncol = sizeK)
    result_1D <- numeric(sizeK + 2)

    for (iter_1 in 1:(sizeK + 2)) {
      lambda_1 <- solve_lambda(alpha, gamma, iter_1)
      a <- rep(1, sizeK)
      b <- (gamma - lambda_1) * alpha
      lambda_2 <- 1 / sum(a / b)
      c <- sqrt(alpha) * (gamma - lambda_1)
      u <- a * lambda_2 / c
      # weight_1D[iter_1 + 1,] <- u / sqrt(alpha)
      weight_1D[iter_1,] <- u / sqrt(alpha)


      # Evaluate the quality of this weight combination - now using eigengap
      w_reshaped <- matrix(w, nrow = s1 * s2, ncol = sizeK)
      # test_w_n_flat <- w_reshaped %*% (weight_1D[iter_1 + 1,]^2)
      test_w_n_flat <- w_reshaped %*% (weight_1D[iter_1,]^2)

      test_w_n <- matrix(test_w_n_flat, nrow = s1, ncol = s2)

      test_D <- diag(rowSums(test_w_n))
      test_L <- test_D - test_w_n
      test_D_inv_sqrt <- diag(1 / sqrt(diag(test_D) + 1e-10))
      test_L_norm <- test_D_inv_sqrt %*% test_L %*% test_D_inv_sqrt
      test_eigen <- eigen(test_L_norm, symmetric = TRUE)
      test_eigenvalues <- sort(test_eigen$values)

      # Calculate all eigengaps and find the maximum
      test_eigengaps <- abs(diff(test_eigenvalues))
      test_eigengaps <- (1:length(test_eigengaps)) * test_eigengaps
      result_1D[iter_1] <- max(test_eigengaps)
    }

    # Select weights that give maximum eigengap (not minimum eigensum)
    max_idx <- which.max(result_1D)
    weight <- weight_1D[max_idx,]

    # Normalize weights to sum to 1
    weight <- weight / sum(weight)

    ### check if there is negative
    w_n_flat <- w_reshaped %*% weight
    # Reshape back to original dimensions
    w_n <- matrix(w_n_flat, nrow = s1, ncol = s2)

    if(sum(w_n < 0) > 0){
      break
    }
  }

  # Return the weights that gave the best result
  return(list(
    weights = best_weights,
    eigengap = best_eigengap # Return eigengap instead of eigensum
  ))
}


fuse_conn_matrices <- function(allSamples, conn.per.omic) {
  patient.names <- as.vector(allSamples)
  num.patients <- length(patient.names)
  returned.conn.matrix <- matrix(0, ncol = num.patients, nrow = num.patients)
  rownames(returned.conn.matrix) <- patient.names
  colnames(returned.conn.matrix) <- patient.names

  shared.omic.count <- matrix(0, ncol = num.patients, nrow = num.patients)
  rownames(shared.omic.count) <- patient.names
  colnames(shared.omic.count) <- patient.names

  for (j in 1:length(conn.per.omic)) {
    curr.omic.patients <- colnames(conn.per.omic[[j]])
    returned.conn.matrix[curr.omic.patients, curr.omic.patients] <- returned.conn.matrix[curr.omic.patients, curr.omic.patients] + conn.per.omic[[j]][curr.omic.patients, curr.omic.patients]
    shared.omic.count[curr.omic.patients, curr.omic.patients] <- shared.omic.count[curr.omic.patients, curr.omic.patients] + 1
  }

  final.ret <- returned.conn.matrix / shared.omic.count
  lower.tri.ret <- final.ret[lower.tri(final.ret)]
  final.ret[shared.omic.count == 0] <- mean(lower.tri.ret[!is.na(lower.tri.ret)])
  ## prevent the connectivity matrix from having NA values
  final.ret[is.na(final.ret)] <- 0
  diag(final.ret) <- 1
  final.ret
}

find_ncluster <- function(W, NUMC=2:15) {
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

filterGenesUsingGeneList <- function(M, geneList, delimeter) {

  if (is.null(delimeter)) {
    features <- colnames(M)
  } else {
    features <- colnames(M) %>%
      lapply(function(x) {
        stringr::str_split(x, delimeter)[[1]][1]
      }) %>%
      unlist() %>%
      toupper()
  }

  if (length(features) %in% c(0, 1)) {
    return(NULL)
  }

  keepFeatures <- features %in% (geneList %>% unique())

  M <- as.matrix(M[, keepFeatures]) %>%
    as.data.frame() %>%
    t() %>%
    unique() %>%
    t()

  rn <- rownames(M)
  cn <- colnames(M)

  M <- M %>% as.numeric() %>% matrix(nrow = nrow(M))
  colnames(M) <- cn
  rownames(M) <- rn
  M
}

runCSIE <- function(dataList, nSamples, defk = 10) {

  dataList <- lapply(1:length(dataList), function(i) {
    data <- dataList[[i]]

    data <- as.data.frame(data) %>% t() %>% unique() %>% t()
    filteredDat <- data[, colVars(data) > 0, drop = FALSE]
    filteredDat
  })

  ang_aff <- lapply(dataList, function(data) {
    dist_matrix <- angular_distance_matrix(as.matrix(data))
    aff <- distance_to_affinity(dist_matrix, 0.5)

    if (nrow(dist_matrix) <= defk){
      tmpk <- nrow(dist_matrix)
    }else{
      tmpk <- defk
    }
    if (nSamples >= 100) {
      non.sym.knn <- apply(aff, 1, function(sim.row) {
        returned.row <- sim.row
        threshold <- sort(sim.row, decreasing = T)[tmpk]
        returned.row[sim.row < threshold] <- 0
        row.sum <- sum(returned.row)
        returned.row[sim.row >= threshold] <- returned.row[sim.row >= threshold] / row.sum
        return(returned.row)
      })
      aff <- non.sym.knn + t(non.sym.knn)
    }
    aff
  })

  mag_aff <- lapply(dataList, function(data) {
    dist_matrix <- as.matrix(dist(as.matrix(data), method = "euclidean"))
    if (nrow(dist_matrix) <= defk){
      tmpk <- nrow(dist_matrix)
    }else{
      tmpk <- defk
    }
    non_zero_dists <- dist_matrix[dist_matrix > 0]
    if (length(non_zero_dists) > 0) {
      if(nSamples <= 500){
        sigma <- quantile(non_zero_dists, 0.6) / 1
      }else{
        sigma <- median(non_zero_dists) / 1  # A reasonable heuristic
      }
    } else {
      sigma <- 1.0  # Fallback
    }

    aff <- distance_to_affinity(dist_matrix, sigma)
    if (nSamples >= 100) {
      non.sym.knn <- apply(aff, 1, function(sim.row) {
        returned.row <- sim.row
        threshold <- sort(sim.row, decreasing = T)[tmpk]
        returned.row[sim.row < threshold] <- 0
        row.sum <- sum(returned.row)
        returned.row[sim.row >= threshold] <- returned.row[sim.row >= threshold] / row.sum
        return(returned.row)
      })
      aff <- non.sym.knn + t(non.sym.knn)
    }
    aff
  })

  aasc_res <- lapply(1:length(mag_aff), function(i) {
    array3D <- merge_matrices(list(mag_aff[[i]], ang_aff[[i]]))
    opt_weights <- aasc_eigengap(array3D, num_clusters = defk)$weights
    aff <- mag_aff[[i]] * (opt_weights[1]) + ang_aff[[i]] * (opt_weights[2])

    if(nSamples > 500){
        if (nrow(aff) <= 20){
          tmpk <- nrow(aff)
        }else{
          tmpk <- 20
        }
        non.sym.knn <- apply(aff, 1, function(sim.row) {
            returned.row <- sim.row
            threshold <- sort(sim.row, decreasing = T)[tmpk]
            returned.row[sim.row < threshold] <- 0
            row.sum <- sum(returned.row)
            returned.row[sim.row >= threshold] <- returned.row[sim.row >= threshold] / row.sum
            return(returned.row)
        })
        aff <- non.sym.knn + t(non.sym.knn)
    }
    rownames(aff) <- colnames(aff) <- rownames(mag_aff[[i]])

    num.clusters <- find_ncluster(aff, 2:15)
    cluster <- spectralClustering(aff, num.clusters)
    names(cluster) <- rownames(aff)
    conn <- clust_to_conn(cluster)
    rownames(conn) <- colnames(conn) <- rownames(aff)
    return(list(aff = aff, conn = conn))
  })

  allSamples <- lapply(mag_aff, function(sim) rownames(sim)) %>%
    unlist() %>%
    unique()

  affinity.per.omic <- lapply(aasc_res, function(res) { res$aff })
  conn.per.omic <- lapply(aasc_res, function(res) { res$conn })

  final.mag.aff <- fuse_conn_matrices(allSamples, mag_aff)
  final.ang.aff <- fuse_conn_matrices(allSamples, ang_aff)
  final.aff <- fuse_conn_matrices(allSamples, affinity.per.omic)
  final.conn <- fuse_conn_matrices(allSamples, conn.per.omic)

  ## Cluster 1
  num.clusters <- find_ncluster(final.mag.aff, 2:10)
  cluster1 <- spectralClustering(final.mag.aff, num.clusters)
  names(cluster1) <- allSamples
  dist_mat1 <- 1 - final.aff
  sh_1 <- try(mean(silhouette(cluster1, dist_mat1)[, "sil_width"]), silent = TRUE)

  ## Cluster 2
  num.clusters <- find_ncluster(final.ang.aff, 2:10)
  cluster2 <- spectralClustering(final.ang.aff, num.clusters)
  names(cluster2) <- allSamples
  dist_mat2 <- 1 - final.aff
  sh_2 <- try(mean(silhouette(cluster2, dist_mat2)[, "sil_width"]), silent = TRUE)

  ## Cluster 3
  probs <- seq(0.01, 0.99, by = 0.01) # Creates sequence from 1% to 99%
  quantiles <- quantile(final.conn, probs = probs)
  mod_optimum <- 0
  dist_mat3 <- 1 - final.aff

  for (connThreshold in quantiles) {
    conn <- final.conn
    conn[conn < connThreshold] <- 0
    graph <- graph_from_adjacency_matrix(conn, mode = "undirected", weighted = T)
    for (seed in 1:50) {
      set.seed(seed)
      louvain_res <- cluster_louvain(graph)
      cluster <- membership(louvain_res)  # Use membership() function
      names(cluster) <- allSamples
      current_modularity <- modularity(graph, cluster)
      if (!inherits(current_modularity, "try-error")) {
        if (max(cluster) <= 10 & current_modularity > mod_optimum) {
          mod_optimum <- current_modularity
          cluster3 <- cluster
        }
      }
    }
  }
  sh_3 <- try(mean(silhouette(cluster3, dist_mat3)[, "sil_width"]), silent = TRUE)
  if (inherits(sh_3, "try-error")) {
    sh_3 <- 0
    cluster3 <- NULL
  }

  all_sh <- c(sh_1, sh_2, sh_3)
  all_cluster <- list(cluster1, cluster2, cluster3)

  return(all_cluster[[ which.max(all_sh)[1] ]])
}

