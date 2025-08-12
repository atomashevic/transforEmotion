#' Install Necessary Python Modules
#'
#' @description
#' Installs required Python modules for the \{transforEmotion\} package, with automatic GPU detection
#' and optional GPU-enabled module installation.
#'
#' @details
#' This function performs the following steps:
#' \itemize{
#'   \item Checks for NVIDIA GPU availability
#'   \item If GPU is detected, prompts user to choose between CPU or GPU installation
#'   \item Installs core modules including transformers, torch, tensorflow, and other dependencies
#'   \item For GPU installations, sets up additional GPU-specific modules via \code{setup_gpu_modules()}
#' }
#'
#' The function automatically manages dependencies and versions, ensuring compatibility
#' between CPU and GPU variants of packages like torch, tensorflow, and torchvision.
#' It uses \code{\link[reticulate]{conda_install}} for package management in the
#' 'transforEmotion' conda environment.
#'
#' @note
#' Ensure that miniconda is installed and properly configured before running this function.
#' For GPU support, NVIDIA drivers must be properly installed on your system.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
# Updated 07.01.2025

check_nvidia_gpu <- function() {
# This functions checks if a NVIDIA GPU is available before we have access to `torch`
if (.Platform$OS.type == "windows") {
  # Windows: Check using nvidia-smi
  gpu_check <- try(system("nvidia-smi", intern = TRUE, ignore.stderr = TRUE), silent = TRUE)
  has_gpu <- !inherits(gpu_check, "try-error")
} else {
  # Linux/MacOS: Check using lspci or nvidia-smi
  gpu_check_lspci <- suppressWarnings(
    system("lspci | grep -i nvidia", ignore.stdout = TRUE, ignore.stderr = TRUE)
  )
  gpu_check_nvidia <- suppressWarnings(
    system("nvidia-smi", ignore.stdout = TRUE, ignore.stderr = TRUE)
  )
  has_gpu <- gpu_check_lspci == 0 || gpu_check_nvidia == 0
}
return(has_gpu)
}


setup_modules <- function() {
  # Configure Python encoding
  reticulate::py_run_string("import sys; sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')")

  # Check for NVIDIA GPU first
  has_gpu <- check_nvidia_gpu()
  use_gpu <- FALSE

  if (has_gpu) {
    # Prompt user for GPU installation
    message("\nNVIDIA GPU detected. Do you want to install GPU modules? ([Y]es/[N]o)")
    user_response <- tolower(readline())
    use_gpu <- user_response %in% c("yes", "y", "Yes", "Y")
  }

  # Get OS information
  OS <- tolower(Sys.info()["sysname"])
  
  # Set necessary modules with their versions
  base_modules <- c(
    "numpy==1.26",   # Use a version compatible with torch 2.1.1
    "scipy==1.10.1", # Add scipy explicitly with compatible version
    "accelerate==0.29.3", # Required for memory optimizations with large models
    "llama-index==0.10.30",
    "nltk==3.8.1",
    "timm", "einops",
    "safetensors==0.4.2", # For loading optimized model weights
    "opencv-python", "pandas==1.5.3", "pypdf==4.0.1", "pytz==2024.1",
    "qdrant-client==1.8.2", "sentencepiece==0.2.0",
    "sentence-transformers==2.2.2",
    # "tokenizers==0.13.3",
    "tokenizers==0.21.0",
    "findingemo-light" # Latest version, for FindingEmo dataset
  )
  
  # Add platform-specific modules
  if (OS == "linux") {
    # triton is only available on Linux x86_64
    # bitsandbytes is available on Linux and Windows
    base_modules <- c(base_modules, "triton", "bitsandbytes==0.45.2")
  } else if (OS == "windows") {
    # bitsandbytes is available on Windows, but triton is not
    base_modules <- c(base_modules, "bitsandbytes==0.45.2")
    message("Note: Skipping 'triton' on Windows (Linux-only)")
  } else if (OS == "darwin") {
    # macOS: Skip both triton and bitsandbytes as they're not supported
    message("Note: Skipping 'triton' and 'bitsandbytes' on macOS (not supported)")
  }

  # Add appropriate torch and tensorflow versions based on GPU availability
  ml_modules <- if (use_gpu) {
    c(
      "tensorflow==2.14.1",
      "torch==2.1.1",
      "torchvision==0.16.1"
    )
  } else {
    c(
      "tensorflow-cpu==2.14.1",
      "torch==2.1.1+cpu",
      "torchvision==0.16.1+cpu"
    )
  }

  # Add remaining modules
  final_modules <- c(
    # "transformers==4.30",
    "transformers==4.47.0",
    "pytubefix" #always use the latest version due to frequent fixes
  )

  # Combine all modules
  modules <- c(base_modules, ml_modules, final_modules)

  # Setup progress bar
  pb <- progress::progress_bar$new(
    format = "  Installing [:bar] :percent eta: :eta",
    total = 4, # Major steps: OpenSSL, pip update, main modules, GPU modules (if needed)
    clear = FALSE,
    width = 60
  )

  # Determine whether any modules need to be installed
  installed_modules <- suppressMessages(
    reticulate::py_list_packages(envname = "transforEmotion")
  )

  # Extract installed package names without versions
  installed_packages <- installed_modules$package

  # Remove version numbers from modules list for comparison
  modules_no_versions <- sub("(.*)==.*", "\\1", modules)

  # Determine missing modules
  missing_modules <- modules[!modules_no_versions %in% installed_packages]

  # Check if OpenSSL is already installed
  openssl_installed <- "openssl" %in% installed_packages

  if (length(missing_modules) > 0 || !openssl_installed) {
    if (!openssl_installed) {
      message("\nInstalling OpenSSL...")
      pb$tick(0, tokens = list(what = "Installing OpenSSL"))
      suppressWarnings(
        reticulate::conda_install("transforEmotion", "openssl=3.0",
                                 pip = FALSE,
                                 conda = "auto",
                                 python_version = NULL,
                                 forge = TRUE)
      )
      pb$tick(1)
    }

    # Update pip (silently)
    pb$tick(0, tokens = list(what = "Updating pip"))
    suppressWarnings(
      reticulate::conda_install("transforEmotion",
                               packages = "pip",
                               pip_options = "--upgrade --quiet",
                               pip = TRUE)
    )
    pb$tick(1)

    # Set pip options based on GPU availability
    pip_options <- c("--upgrade", "--quiet")
    if (!use_gpu) {
      pip_options <- c(pip_options,
                      "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    }

    # Install modules with appropriate pip options (silently)
    pb$tick(0, tokens = list(what = "Installing main modules"))
    suppressWarnings(
      reticulate::conda_install(
        envname = "transforEmotion",
        packages = missing_modules,
        pip_options = pip_options,
        pip = TRUE
      )
    )
    pb$tick(1)

    # If GPU was selected, install additional GPU modules
    if (use_gpu) {
      pb$tick(0, tokens = list(what = "Installing GPU modules"))
      suppressWarnings(setup_gpu_modules())
      pb$tick(1)
    }

    message("\nInstallation complete!")
  } else {
    message("\nAll required modules are already installed.")

    # If GPU was selected and all base modules are installed, still check GPU modules
    if (use_gpu) {
      pb$tick(0, tokens = list(what = "Checking GPU modules"))
      suppressWarnings(setup_gpu_modules())
      pb$tick(1)
    }
  }
}
