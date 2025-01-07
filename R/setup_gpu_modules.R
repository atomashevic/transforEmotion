#' Install GPU Python Modules
#'
#' @description
#' Installs GPU-specific Python modules for the \{transforEmotion\} conda environment.
#'
#' @details
#' This function installs additional GPU-specific modules including:
#' \itemize{
#'   \item AutoAWQ for weight quantization
#'   \item Auto-GPTQ for GPU quantization
#'   \item Optimum for transformer optimization
#'   \item llama-cpp-python (Linux only) for CPU/GPU inference
#' }
#'
#' The function is typically called by \code{setup_modules()} when GPU installation
#' is selected, but can also be run independently to update GPU-specific modules.
#'
#' @note
#' This function requires NVIDIA GPU and drivers to be properly installed.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @export
#'
# Install GPU modules
# Updated 07.01.2025
setup_gpu_modules <- function() {
  # Set necessary modules with their versions
  modules <- c(
    "autoawq==0.2.5", "auto-gptq==0.7.1", "optimum==1.19.1"
  )

  # Check for Linux and add llama-cpp-python if applicable
  if (system.check()$OS == "linux") {
    # Set environment variables for llama-cpp-python build
    Sys.setenv(
      LDFLAGS = "-lstdc++fs",
      CMAKE_ARGS = "-DLLAMA_CUBLAS=on"  # Enable CUDA support
    )
    
    # Try installing pre-built wheel first, then fallback to source
    tryCatch({
      reticulate::conda_install(
        envname = "transforEmotion",
        packages = "llama-cpp-python",
        pip_options = c("--upgrade", "--quiet", "--prefer-binary"),
        pip = TRUE
      )
    }, error = function(e) {
      message("Pre-built wheel not available, attempting to build from source...")
      reticulate::conda_install(
        envname = "transforEmotion",
        packages = "llama-cpp-python",
        pip_options = c("--upgrade", "--quiet"),
        pip = TRUE
      )
    })
    
    # Remove llama-cpp-python from main modules list since we handled it separately
    modules <- modules[modules != "llama-cpp-python"]
  }

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

  # Only proceed if there are modules to install
  if (length(missing_modules) > 0) {
    # Set pip options for quiet installation
    pip_options <- c("--upgrade", "--quiet", "--prefer-binary")

    # Install modules silently
    suppressMessages(
      reticulate::conda_install(
        envname = "transforEmotion",
        packages = missing_modules,
        pip_options = pip_options,
        pip = TRUE
      )
    )
  }
}
