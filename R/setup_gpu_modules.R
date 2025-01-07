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
    modules <- c(modules, "llama-cpp-python")
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
    pip_options <- c("--upgrade", "--quiet")

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
