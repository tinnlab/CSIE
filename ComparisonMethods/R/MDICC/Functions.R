### some functions originally in C
# Core fast simplex projection function (REQUIRED by all other functions)
simplex_projection_fast <- function(v) {
  n <- length(v)

  # Sort in descending order
  u <- sort(v, decreasing = TRUE)

  # Compute cumulative sum minus 1
  cssv <- cumsum(u) - 1

  # Find the breakpoint using vectorized operations
  ind <- cssv / (1:n)
  rho <- max(which(u > ind))

  # Compute threshold
  theta <- cssv[rho] / rho

  # Project (vectorized)
  result <- pmax(v - theta, 0)

  return(result)
}

# Fast implementation for small to medium matrices
projsplx_R_fast <- function(y, x = NULL) {
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  m <- nrow(y)
  n <- ncol(y)

  # Create output matrix if not provided
  if (is.null(x)) {
    x <- matrix(0, nrow = m, ncol = n)
  }

  # Vectorized processing for all columns at once

  # Step 1: Calculate column means and adjust (vectorized)
  col_means <- colMeans(y)
  s <- sweep(y, 2, (col_means - 1) / m, "-")

  # Step 2: Find columns that need iterative projection (have negative values)
  col_mins <- apply(s, 2, min)
  needs_projection <- col_mins < 0

  # Step 3: Handle columns that don't need projection (all non-negative)
  if (any(!needs_projection)) {
    x[, !needs_projection] <- s[, !needs_projection]
  }

  # Step 4: Handle columns that need iterative projection
  if (any(needs_projection)) {
    proj_cols <- which(needs_projection)

    for (k in proj_cols) {
      s_col <- s[, k]
      x[, k] <- simplex_projection_fast(s_col)
    }
  }

  return(x)
}

# Fully vectorized version (your existing code with dependencies)
projsplx_R_vectorized <- function(y) {
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  m <- nrow(y)
  n <- ncol(y)

  # Vectorized normalization
  col_means <- colMeans(y)
  s <- sweep(y, 2, (col_means - 1) / m, "-")

  # Check which columns need projection
  col_mins <- apply(s, 2, min)
  needs_projection <- col_mins < 0

  # Initialize result
  result <- s

  # Process columns that need projection
  if (any(needs_projection)) {
    proj_data <- s[, needs_projection, drop = FALSE]

    # Apply fast projection to each column using apply
    projected <- apply(proj_data, 2, simplex_projection_fast)

    # Handle single column case
    if (is.vector(projected) && sum(needs_projection) == 1) {
      projected <- matrix(projected, ncol = 1)
    }

    result[, needs_projection] <- projected
  }

  return(result)
}

# Parallel version (your existing code with dependencies)
projsplx_R_parallel <- function(y, n_cores = 4) {
  if (!requireNamespace("parallel", quietly = TRUE)) {
    warning("parallel package not available, using sequential version")
    return(projsplx_R_vectorized(y))
  }

  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  m <- nrow(y)
  n <- ncol(y)

  if (is.null(n_cores)) {
    n_cores <- min(parallel::detectCores() - 1, ncol(y))
  }

  # Vectorized normalization
  col_means <- colMeans(y)
  s <- sweep(y, 2, (col_means - 1) / m, "-")

  # Check which columns need projection
  col_mins <- apply(s, 2, min)
  needs_projection <- col_mins < 0

  # Initialize result
  result <- s

  # Process columns that need projection in parallel
  if (any(needs_projection)) {
    proj_indices <- which(needs_projection)

    # Parallel processing
    cl <- parallel::makeCluster(n_cores)
    on.exit(parallel::stopCluster(cl))

    # Export function to cluster
    parallel::clusterExport(cl, "simplex_projection_fast", envir = environment())

    # Apply projection in parallel
    projected_cols <- parallel::parLapply(cl, proj_indices, function(i) {
      simplex_projection_fast(s[, i])
    })

    # Assign results back
    for (i in seq_along(proj_indices)) {
      result[, proj_indices[i]] <- projected_cols[[i]]
    }
  }

  return(result)
}

# Smart wrapper with automatic method selection (your existing code)
projsplx_wrapper_fast <- function(y, method = "auto", n_cores = 4) {
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  n_elements <- nrow(y) * ncol(y)

  if (method == "auto") {
    if (n_elements < 1000) {
      method <- "fast"
    } else if (n_elements < 10000) {
      method <- "vectorized"
    } else {
      method <- "parallel"
    }
  }

  switch(method,
    "fast" = projsplx_R_fast(y),
    "vectorized" = projsplx_R_vectorized(y),
    "parallel" = projsplx_R_parallel(y, n_cores),
    stop("Unknown method. Use 'fast', 'vectorized', or 'parallel'")
  )
}

# Main interface function - use this one in your MDICC workflow
projsplx_R <- function(y, x = NULL) {

  # Use the fast wrapper for automatic method selection
  result <- projsplx_wrapper_fast(y, method = "auto")

  # If x was provided, copy result to x (for C compatibility)
  if (!is.null(x)) {
    x[] <- result
    return(x)
  }

  return(result)
}



### Network Fursion ###
# compute the dominate set for the matrix aff.matrix and NR.OF.KNN
dominate.set <- function( aff.matrix, NR.OF.KNN ) {

  # create the structure to save the results
  PNN.matrix = array(0,c(nrow(aff.matrix),ncol(aff.matrix)))

  # sort each row of aff.matrix in descending order and saves the sorted
  # array and a collection of vectors with the original indices
  res.sort = apply(t(aff.matrix),MARGIN=2,FUN=function(x) {return(sort(x, decreasing = TRUE, index.return = TRUE))})
  sorted.aff.matrix = t(apply(as.matrix(1:length(res.sort)),MARGIN=1,function(x) { return(res.sort[[x]]$x) }))
  sorted.indices = t(apply(as.matrix(1:length(res.sort)),MARGIN=1,function(x) { return(res.sort[[x]]$ix) }))

  # get the first NR.OF.KNN columns of the sorted array
  res = sorted.aff.matrix[,1:NR.OF.KNN]

  # create a matrix of NR.OF.KNN columns by binding vectors of
  # integers from 1 to the number of rows/columns of aff.matrix
  inds = array(0,c(nrow(aff.matrix),NR.OF.KNN))
  inds = apply(inds,MARGIN=2,FUN=function(x) {x=1:nrow(aff.matrix)})

  # get the first NR.OF.KNN columns of the indices of aff.matrix
  loc = sorted.indices[,1:NR.OF.KNN]

  # assign to PNN.matrix the sorted indices
  PNN.matrix[(as.vector(loc)-1)*nrow(aff.matrix)+as.vector(inds)] = as.vector(res)

  # compute the final results and return them
  PNN.matrix = (PNN.matrix + t(PNN.matrix))/2

  return(PNN.matrix)

}

# compute the transition field of the given matrix
transition.fields <- function( W ) {

  # get any index of columns with all 0s
  zero.index = which(apply(W,MARGIN=1,FUN=sum)==0)

  # compute the transition fields
  W = dn(W,'ave')

  w = sqrt(apply(abs(W),MARGIN=2,FUN=sum)+.Machine$double.eps)
  W = W / t(apply(array(0,c(nrow(W),ncol(W))),MARGIN=2,FUN=function(x) {x=w}))
  W = W %*% t(W)

  # set to 0 the elements of zero.index
  W[zero.index,] = 0
  W[,zero.index] = 0

  return(W)

}

# normalizes a symmetric kernel
dn <- function( w, type ) {

  # compute the sum of any column
  D = apply(w,MARGIN=2,FUN=sum)

  # type "ave" returns D^-1*W
  if(type=="ave") {
    D = 1 / (D + .Machine$double.eps)  # Added small epsilon for numerical stability
    D_temp = matrix(0,nrow=length(D),ncol=length(D))
    D_temp[cbind(1:length(D),1:length(D))] = D
    D = D_temp
    wn = D %*% w
  }
  # type "gph" returns D^-1/2*W*D^-1/2
  else if(type=="gph") {
    D = 1 / sqrt(D + .Machine$double.eps)  # Added small epsilon for numerical stability
    D_temp = matrix(0,nrow=length(D),ncol=length(D))
    D_temp[cbind(1:length(D),1:length(D))] = D
    D = D_temp
    wn = D %*% (w %*% D)
  }
  else {
    stop("Invalid type!")
  }

  return(wn)

}

# compute the eigenvalues and eigenvectors
eig1 <- function( A, c = NA, isMax = NA, isSym = NA ) {

  # set the needed parameters
  if(is.na(c)) {
    c = dim(A)[1]
  }
  if(c>dim(A)[1]) {
    c = dim(A)[1]
  }
  if(is.na(isMax)) {
    isMax = 1
  }
  if(is.na(isSym)) {
    isSym = 1
  }

  # compute the eigenvalues and eigenvectors of A
  if(isSym==1) {
    eigen_A = eigen(A,symmetric=TRUE)
  }
  else {
    eigen_A = eigen(A)
  }
  v = eigen_A$vectors
  d = eigen_A$values

  # sort the eigenvectors
  if(isMax == 0) {
    eigen_A_sorted = sort(d,index.return=TRUE)
  }
  else {
    eigen_A_sorted = sort(d,decreasing=TRUE,index.return=TRUE)
  }
  d1 = eigen_A_sorted$x
  idx = eigen_A_sorted$ix # the index after ranking
  idx1 = idx[1:c] # the index after ranking and pick top c¡®s index

  # compute the results
  eigval = d[idx1]
  eigvec = Re(v[,idx1])
  eigval_full = d[idx]

  return(list(eigval=eigval,eigvec=eigvec,eigval_full=eigval_full))

}

# compute the L2 distance - optimized version
L2_distance_1 <- function( a, b ) {

  if(dim(a)[1] == 1) {
    a = rbind(a,rep(0,dim(a)[2]))
    b = rbind(b,rep(0,dim(b)[2]))
  }

  # Vectorized computation for better performance
  aa = colSums(a * a)
  bb = colSums(b * b)
  ab = t(a) %*% b

  # Use outer product for more efficient computation
  d = outer(aa, bb, "+") - 2 * ab
  d = Re(d)
  d = pmax(d, 0)  # More efficient than mapply

  # Zero out diagonal
  diag(d) = 0

  return(d)

}

# umkl function
umkl <- function( D, beta = NA ) {

  # set some parameters
  if(is.na(beta)) {
    beta = 1 / length(D)
  }
  tol = 1e-4
  u = 20
  logU = log(u)

  # compute Hbeta
  res_hbeta = Hbeta(D, beta)
  H = res_hbeta$H
  thisP = res_hbeta$P

  betamin = -Inf
  betamax = Inf
  # evaluate whether the perplexity is within tolerance
  Hdiff = H - logU
  tries = 0
  while (abs(Hdiff) > tol && tries < 30) {
    #if not, increase or decrease precision
    if (Hdiff > 0) {
      betamin = beta
      if(abs(betamax)==Inf) {
        beta = beta * 2
      }
      else {
        beta = (beta + betamax) / 2
      }
    }
    else {
      betamax = beta
      if(abs(betamin)==Inf) {
        beta = beta * 2
      }
      else {
        beta = (beta + betamin) / 2
      }
    }
    # compute the new values
    res_hbeta = Hbeta(D, beta)
    H = res_hbeta$H
    thisP = res_hbeta$P
    Hdiff = H - logU
    tries = tries + 1
  }

  return(thisP)

}

Hbeta <- function( D, beta ) {

  D = (D - min(D)) / (max(D) - min(D) + .Machine$double.eps)
  P = exp(-D * beta)
  sumP = sum(P)
  H = log(sumP) + beta * sum(D * P) / sumP
  P = P / sumP

  return(list(H=H,P=P))

}

MDICC <- function( X, c=3, no.dim = NA, k = 10 ) {

  # set any required parameter to the defaults
  if(is.na(no.dim)) {
    no.dim = c
  }

  # start the clock to measure the execution time
  ptm = proc.time()

  # set some parameters
  NITER = 30
  num = ncol(X[[1]])
  r = -1
  beta = 0.8

  cat("Removing matrices with NA values...\n")
  has_na <- sapply(X, function(x) any(is.na(x)))
  if(any(has_na)) {
    cat("Removing", sum(has_na), "matrices with NA values\n")
    X <- X[!has_na]
  }

  # Convert to sparse matrices for better memory efficiency
  cat("Converting to sparse matrices...\n")
  for(i in 1:length(X)){
    X[[i]] = Matrix(X[[i]], sparse = TRUE)
  }

  X2 = X

  # Normalization step
  cat("Normalizing data...\n")
  for(i in 1:length(X)){
    d = Matrix::rowSums(X2[[i]])
    d = pmax(d, .Machine$double.eps)  # Avoid division by zero
    d1 = Diagonal(x = 1/d)
    X2[[i]] = d1 %*% X2[[i]]
  }

  D_Kernels = X2

  # set up some parameters
  alphaK = rep(1/length(D_Kernels), length(D_Kernels))

  # Compute initial average distance matrix
  cat("Computing initial distance matrix...\n")
  distX = Reduce("+", D_Kernels) / length(D_Kernels)
  distX = as.matrix(distX)  # Convert to regular matrix for sorting

  # Optimized sorting for all rows
  res = apply(distX, MARGIN=1, FUN=function(x) return(sort(x, index.return = TRUE)))
  distX1 = matrix(0, nrow=nrow(distX), ncol=ncol(distX))
  idx = matrix(0, nrow=nrow(distX), ncol=ncol(distX))

  for(i in 1:nrow(distX)) {
    distX1[i,] = res[[i]]$x
    idx[i,] = res[[i]]$ix
  }

  A = matrix(0, nrow=num, ncol=num)
  di = distX1[,2:(k+2)]
  rr = 0.5 * (k * di[,k+1] - rowSums(di[,1:k]))
  id = idx[,2:(k+2)]

  if(r<=0) {
    r = mean(rr)
  }
  lambda = max(mean(rr), 0)

  ###
  S0 = max(distX) - distX

  cat("Network fusion.\n")

  # compute dn(normalization)
  S0 = dn(S0,'ave')
  S = S0
  D0 = diag(colSums(S))  # More efficient than apply
  L0 = D0 - S

  eig1_res = eig1(L0,c,0)
  
  # cat("  Range distX:", range(distX, na.rm=T), "\n")
  # S0 = max(distX) - distX
  # cat("After S0 = max(distX) - distX:\n")
  # cat("  Range:", range(S0), "\n")
  # cat("  Has Inf:", any(is.infinite(S0)), "\n")
  # cat("  Has NaN:", any(is.nan(S0)), "\n")
  # cat("  Column sums range:", range(colSums(S0)), "\n")

  # # compute dn(normalization)
  # S0 = dn(S0,'ave')
  # cat("After S0 = dn(S0, 'ave'):\n")
  # cat("  Range:", range(S0), "\n")
  # cat("  Has Inf:", any(is.infinite(S0)), "\n")
  # cat("  Has NaN:", any(is.nan(S0)), "\n")
  
  # S = S0
  # D0 = diag(colSums(S))
  # cat("D0 diagonal range:", range(diag(D0)), "\n")
  
  # L0 = D0 - S
  # cat("L0 range:", range(L0), "\n")
  # cat("L0 has Inf:", any(is.infinite(L0)), "\n")
  # cat("L0 has NaN:", any(is.nan(L0)), "\n")

  # eig1_res = eig1(L0,c,0)

  ###

  F_eig1 = eig1_res$eigvec
  temp_eig1 = eig1_res$eigval
  evs_eig1 = eig1_res$eigval_full

  # perform the iterative procedure NITER times
  converge = vector()
  for(iter in 1:NITER) {

    cat("Iteration: ",iter,"\n")

    distf = L2_distance_1(t(F_eig1),t(F_eig1))
    A = matrix(0, nrow=num, ncol=num)
    b = idx[,2:ncol(idx)]
    a = matrix(rep(1:num, ncol(b)), nrow=num, ncol=ncol(b))
    inda = cbind(as.vector(a), as.vector(b))
    ad = (distX[inda] + lambda*distf[inda]) / (2*r)
    dim(ad) = c(num, ncol(b))

    # CRITICAL CHANGE: Replace C function call with R implementation
    c_input = -t(ad)
    # Use our fast R implementation instead of C function
    ad_projected = projsplx_R(c_input)  # This replaces .Call("projsplx_R",c_input,c_output)
    ad = t(ad_projected)

    A[inda] = as.vector(ad)
    A[is.nan(A)] = 0
    A = (A + t(A)) / 2
    S = (1 - beta) * S + beta * A

    # After updating S
    D = diag(colSums(S))  # More efficient than apply
    L = D - S
    F_old = F_eig1
    eig1_res = eig1(L,c,0)
    F_eig1 = eig1_res$eigvec
    temp_eig1 = eig1_res$eigval
    ev_eig1 = eig1_res$eigval_full
    evs_eig1 = cbind(evs_eig1,ev_eig1)

    # Compute DD for kernel weights
    DD = vector()
    for (i in 1:length(D_Kernels)) {
      temp1 = (.Machine$double.eps + as.matrix(D_Kernels[[i]])) * (S + .Machine$double.eps)
      temp2 = 0.5 * (.Machine$double.eps + as.matrix(D_Kernels[[i]])) * (as.matrix(D_Kernels[[i]]) + .Machine$double.eps)
      temp = temp1 - temp2
      DD[i] = mean(colSums(temp))  # More efficient than apply
    }

    alphaK0 = umkl(DD)
    alphaK0 = alphaK0 / sum(alphaK0)
    alphaK = (1-beta) * alphaK + beta * alphaK0
    alphaK = alphaK / sum(alphaK)

    fn1 = sum(ev_eig1[1:c])
    fn2 = sum(ev_eig1[1:(c+1)])
    converge[iter] = fn2 - fn1

    if (iter < 10) {
      if (ev_eig1[length(ev_eig1)] > 0.000001) {
        lambda = 1.5 * lambda
        r = r / 1.01
      }
    } else {
      if(converge[iter] > converge[iter-1]) {
        S = S_old
        break
      }
    }
    S_old = S

    # Update distance matrix with new weights
    distX = as.matrix(D_Kernels[[1]]) * alphaK[1]
    if (length(D_Kernels) >= 2){
      for (i in 2:length(D_Kernels)) {
        distX = distX + as.matrix(D_Kernels[[i]]) * alphaK[i]
      }
    }

    # sort distX for rows
    res = apply(distX, MARGIN=1, FUN=function(x) return(sort(x, index.return = TRUE)))
    for(i in 1:nrow(distX)) {
      distX1[i,] = res[[i]]$x
      idx[i,] = res[[i]]$ix
    }
  }

  LF = F_eig1
  D = diag(colSums(S))
  L = D - S

  # compute the eigenvalues and eigenvectors of final result
  eigen_L = eigen(L)
  U = eigen_L$vectors
  D_final = eigen_L$values

  # compute the execution time
  execution.time = proc.time() - ptm
  cat("Execution time:", execution.time[3], "seconds\n")

  # return the similarity matrix
  return(S)
}



### Local Affinity Matrix ###
kscale <- function(matrix, k = 7, dists = NULL, minval = 0.004) {
  r <- nrow(matrix)
  c <- ncol(matrix)
  scale <- matrix(0, nrow = r, ncol = c)

  for (i in 1:k) {
    sorted_indices <- apply(matrix, 1, order)
    d <- numeric(r)
    for (row in 1:r) {
      col_idx <- sorted_indices[i + 1, row]  # +1 because we skip the diagonal (self)
      d[row] <- matrix[row, col_idx]
    }

    if (is.null(dists)) {
      dists_to_use <- d
    } else {
      dists_to_use <- dists
    }

    d_matrix <- matrix(d, nrow = length(d), ncol = 1)
    scale1 <- d_matrix %*% t(d_matrix)
    scale <- scale1 + scale
  }

  scale <- scale / k
  scale <- pmax(scale, minval)
  return(scale)
}

affinity <- function(matrix, k=7) {
  scale <- kscale(matrix, k)
  msq <- matrix * matrix
  scaled <- -msq / (0.5 * scale + 0.5 * matrix)
  scaled[is.nan(scaled)] <- 0.0
  a <- exp(scaled)
  diag(a) <- 0.0
  return(a)
}

testaff <- function(matrix, k=7) {
  k <- as.integer(k)
  a <- matrix
  affi <- affinity(a, k)

  return(affi)
}



### k means and select optimal cluster ###
kmean_opt <- function(test_S, k_range = 2:10) {
  # best_k <- NULL
  best_score <- -999
  best_clusters <- NULL
  silhouette_scores <- numeric(length(k_range))
  names(silhouette_scores) <- k_range

  for(i in seq_along(k_range)) {
    k <- k_range[i]
    set.seed(1234)
    kmeans_result <- kmeans(test_S, centers = k)
    clusters <- kmeans_result$cluster
    dist_matrix <- max(test_S) - test_S
    diag(dist_matrix) <- 0
    dist_matrix <- as.dist(max(test_S) - test_S)

    # Calculate silhouette score
    sil_result <- cluster::silhouette(clusters, dist_matrix)
    avg_sil_score <- mean(sil_result[, "sil_width"])
    silhouette_scores[i] <- avg_sil_score

    # Update best result
    if(avg_sil_score > best_score) {
      best_score <- avg_sil_score
      # best_k <- k
      best_clusters <- clusters
    }
  }
  return(best_clusters)
}
