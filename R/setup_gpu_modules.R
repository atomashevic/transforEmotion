#' Install GPU Python Modules
#'
#' @description Installs GPU modules for the {transforEmotion} conda environment
#'
#' @details Installs modules for miniconda using \code{\link[reticulate]{conda_install}}
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @export
#'
# Install GPU modules
# Updated 06.02.2024
setup_gpu_modules <- function()
{

  # Set necessary modules
  modules <- c(
    "autoawq==0.2.5", "auto-gptq==0.7.1", "optimum==1.19.1"
  )

# TODO freeze versions of modules to their current versions

  # Check for Linux
  if(system.check()$OS == "linux"){
    modules <- c(modules, "llama-cpp-python")
  }

  # Determine whether any modules need to be installed
  installed_modules <- reticulate::py_list_packages(envname = "transforEmotion")

  # Determine missing modules
  missing_modules <- modules[!modules %in% installed_modules$package]

  # Determine if modules need to be installed
  if(length(missing_modules) != 0){

    # Send message to user
    message("\nInstalling modules for 'transforEmotion'...")

    # Install modules
    reticulate::conda_install(
      "transforEmotion", packages = missing_modules, pip = TRUE
    )

  }

}
