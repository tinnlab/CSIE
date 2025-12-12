# load_hMKL.R
hMKL_setup <- function() {
  # Load required packages
  required_packages <- c("Matrix", "kernlab", "cluster", "Rtsne")
  for(pkg in required_packages) {
    if(!require(pkg, character.only = TRUE)) {
      install.packages(pkg)
      library(pkg, character.only = TRUE)
    }
  }

  # Source all hMKL files
  hMKL_path <- "/data/daotran/Cancer_Subtyping/BiB_Submission/ShareCode/R/hMKL/R"
  r_files <- list.files(hMKL_path, pattern = "\\.R$", full.names = TRUE)

  for(file in r_files) {
    source(file)
  }

  cat("hMKL functions loaded successfully!\n")
}

# Run the setup
hMKL_setup()