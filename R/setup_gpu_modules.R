#' Install GPU Python Modules
#'
#' @description
#' Installs GPU-specific Python modules using uv-managed environments.
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
#' is selected, but can also be run independently to add GPU-related packages.
#'
#' @note
#' This function requires NVIDIA GPU and drivers to be properly installed.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @export
#'
# Install GPU modules
# Updated 07.01.2025 (uv-based)
setup_gpu_modules <- function() {
  # Base GPU-friendly set; torch/tensorflow variants are handled in setup_modules.
  modules <- c("autoawq==0.2.5", "auto-gptq==0.7.1", "optimum==1.19.1")

  # Linux: llama-cpp-python (optional; may build from source)
  os_name <- tolower(Sys.info()["sysname"])  # linux, windows, darwin
  if (identical(os_name, "linux")) {
    Sys.setenv(LDFLAGS = "-lstdc++fs", CMAKE_ARGS = "-DLLAMA_CUBLAS=on")
    modules <- c(modules, "llama-cpp-python")
  }

  # Add packages to the current uv environment
  try(reticulate::py_require(packages = modules), silent = TRUE)
  invisible(NULL)
}
