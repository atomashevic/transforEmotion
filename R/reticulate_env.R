#' @noRd
te_should_use_gpu <- function() {
  # Allow explicit override via env var
  get_bool <- function(x) {
    x <- tolower(as.character(x))
    nzchar(x) && x %in% c("1", "true", "t", "yes", "y")
  }
  force_cpu <- get_bool(Sys.getenv("TRANSFOREMOTION_FORCE_CPU", unset = "")) ||
               get_bool(Sys.getenv("TE_FORCE_CPU", unset = ""))
  if (force_cpu) return(FALSE)

  explicit_gpu <- Sys.getenv("TRANSFOREMOTION_USE_GPU", unset = NA)
  if (!is.na(explicit_gpu) && nzchar(explicit_gpu)) return(get_bool(explicit_gpu))

  # macOS: skip CUDA decisions (MPS not handled here)
  os <- tolower(Sys.info()["sysname"])  # linux, windows, darwin
  if (identical(os, "darwin")) return(FALSE)

  # Default: detect NVIDIA GPU
  res <- FALSE
  try({ res <- check_nvidia_gpu() }, silent = TRUE)
  isTRUE(res)
}

#' @noRd
.configure_uv_env <- function(use_gpu = FALSE) {
  # Define baseline packages and versions (aligned with previous setup)
  base_modules <- c(
    "numpy==1.26",
    "scipy==1.10.1",
    "accelerate==0.29.3",
    "llama-index==0.10.30",
    "nltk==3.8.1",
    "timm",
    "einops",
    "safetensors==0.4.3",
    "opencv-python",
    "pandas==1.5.3",
    "pypdf==4.0.1",
    "pytz==2024.1",
    "qdrant-client==1.8.2",
    "sentencepiece==0.2.0",
    "sentence-transformers==2.2.2",
    "rank-bm25==0.2.2",
    "tokenizers==0.21.0",
    "findingemo-light",
    "transformers==4.51.0"
  )

  # Platform-specific additions
  OS <- tolower(Sys.info()["sysname"])  # linux, windows, darwin
  if (OS %in% c("linux", "windows")) {
    base_modules <- c(base_modules, "bitsandbytes==0.45.2")
  }

  # ML stack (CPU default)
  ml_modules <- if (isTRUE(use_gpu)) {
    # Note: Installing GPU wheels via uv may require configuring indices externally.
    c("tensorflow==2.14.1", "torch==2.1.1", "torchvision==0.16.1")
  } else {
    c("tensorflow-cpu==2.14.1", "torch==2.1.1", "torchvision==0.16.1")
  }

  modules <- c(base_modules, ml_modules)

  # For PyTorch wheels, set extra index for CPU/GPU flavors (best-effort)
  extra_index <- if (isTRUE(use_gpu)) "https://download.pytorch.org/whl/cu121" else "https://download.pytorch.org/whl/cpu"
  prev_pip_idx <- Sys.getenv("PIP_EXTRA_INDEX_URL", unset = "")
  prev_uv_idx  <- Sys.getenv("UV_EXTRA_INDEX_URL", unset = "")
  on.exit({
    if (nzchar(prev_pip_idx)) Sys.setenv(PIP_EXTRA_INDEX_URL = prev_pip_idx) else Sys.unsetenv("PIP_EXTRA_INDEX_URL")
    if (nzchar(prev_uv_idx))  Sys.setenv(UV_EXTRA_INDEX_URL  = prev_uv_idx)  else Sys.unsetenv("UV_EXTRA_INDEX_URL")
  }, add = TRUE)
  Sys.setenv(PIP_EXTRA_INDEX_URL = extra_index)
  Sys.setenv(UV_EXTRA_INDEX_URL  = extra_index)

  # Use Python 3.10 by default (allow latest micro): >=3.10,<3.11
  reticulate::py_require(
    packages = modules,
    python_version = ">=3.10,<3.11",
    action = "set"
    # You may add exclude_newer = "YYYY-MM-DD" here for strict reproducibility
  )

  invisible(TRUE)
}

#' @noRd
# Ensure Python is initialized via uv with the required packages
ensure_te_py_env <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(invisible(FALSE))

  # If Python already initialized, nothing to do
  initialized <- FALSE
  try({ initialized <- reticulate::py_available(initialize = FALSE) }, silent = TRUE)
  if (isTRUE(initialized)) return(invisible(TRUE))

  # Not initialized yet — configure uv environment (CPU default)
  use_gpu <- te_should_use_gpu()
  try(.configure_uv_env(use_gpu = use_gpu), silent = TRUE)
  invisible(TRUE)
}
