#' Install Necessary Python Modules
#'
#' @description
#' Installs required Python modules for the \{transforEmotion\} package, using uv
#' for fast, reproducible environments. Optionally detects GPU and can add
#' GPU-oriented packages.
#'
#' @details
#' This function performs the following steps:
#' \itemize{
#'   \item Detects NVIDIA GPU availability automatically
#'   \item Installs core modules including transformers, torch, tensorflow, and other dependencies
#'   \item For GPU systems, adds GPU-specific packages (and optional extras via \code{setup_gpu_modules()})
#' }
#'
#' The function declares Python requirements via \code{\link[reticulate]{py_require}},
#' which uses uv to resolve and cache an ephemeral environment on first use. No
#' conda/Miniconda is required.
#'
#' @note
#' For GPU support, NVIDIA drivers must be properly installed on your system. If
#' you need vendor-specific wheels (e.g., for CUDA), configure package indexes
#' prior to calling this function (see Notes in documentation).
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
# Updated 07.01.2025

check_nvidia_gpu <- function() {
  # This function checks if an NVIDIA GPU is available before we have access to `torch`
  if (.Platform$OS.type == "windows") {
    # Windows: Check using nvidia-smi if available
    has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))
    if (!has_nvidia_smi) return(FALSE)
    rc <- suppressWarnings(system("nvidia-smi", ignore.stdout = TRUE, ignore.stderr = TRUE))
    return(rc == 0)
  } else {
    # Linux/macOS: Prefer command existence checks to avoid noisy "command not found"
    has_gpu <- FALSE
    has_lspci <- nzchar(Sys.which("lspci"))
    has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))

    if (has_lspci) {
      rc <- suppressWarnings(system("lspci | grep -i nvidia", ignore.stdout = TRUE, ignore.stderr = TRUE))
      has_gpu <- has_gpu || (rc == 0)
    }
    if (has_nvidia_smi) {
      rc2 <- suppressWarnings(system("nvidia-smi", ignore.stdout = TRUE, ignore.stderr = TRUE))
      has_gpu <- has_gpu || (rc2 == 0)
    }
    return(has_gpu)
  }
}


#' Setup Required Python Modules
#'
#' @description
#' Installs and configures required Python modules for transforEmotion,
#' optionally enabling GPU-accelerated variants when a compatible NVIDIA GPU
#' is detected. Uses reticulate's uv-backed ephemeral environment.
#'
#' @details
#' This function ensures required modules are available and can add
#' additional GPU-specific packages when requested. See also
#' \code{setup_gpu_modules()} for GPU add-ons.
#'
#' @return Invisibly returns NULL.
#' @export
setup_modules <- function() {
  # Ensure Python is initialized through uv with baseline dependencies
  ensure_te_py_env()

  # Configure Python encoding (best-effort)
  try(reticulate::py_run_string("import sys; sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')"), silent = TRUE)

  # Optional: GPU add-ons when GPU present (automatic; override with TE_FORCE_CPU=1)
  use_gpu <- te_should_use_gpu()
  initialized <- FALSE
  try({ initialized <- reticulate::py_available(initialize = FALSE) }, silent = TRUE)
  if (initialized && use_gpu) {
    message("Python is already initialized; restart R to switch to GPU packages.")
    return(invisible(NULL))
  }
  if (use_gpu) {
    # Add GPU-friendly packages. Note: PyTorch CUDA wheels may require custom
    # indexes which uv does not expose via py_require(); see README for tips.
    try(reticulate::py_require(packages = c(
      "tensorflow==2.14.1", "torch==2.1.1", "torchvision==0.16.1"
    )), silent = TRUE)
    try(setup_gpu_modules(), silent = TRUE)
    message("GPU detected; added GPU-related packages.")
  } else {
    message("Using CPU-only environment (set TE_FORCE_CPU=0 and ensure NVIDIA drivers for GPU).")
  }

  message("Python dependencies ensured via uv (reticulate::py_require).")
  invisible(NULL)
}
