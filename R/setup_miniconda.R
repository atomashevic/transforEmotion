#' Check if the "transforEmotion" conda environment exists
#'
#' This function checks if the "transforEmotion" conda environment exists by
#' running the command "conda env list" and searching for the environment name
#' in the output.
#'
#' @return A logical value indicating whether the "transforEmotion" conda
#' environment exists.
#'

conda_check <- function(){
  env_list <- reticulate::conda_list()$name
  tE_env <- sum(grepl("transforEmotion", env_list))
  return (tE_env!=0)
}

#' Install Miniconda and activate the transforEmotion environment
#'
#' @description Installs miniconda and activates the transforEmotion environment
#'
#' @details Installs miniconda using \code{\link[reticulate]{install_miniconda}} and activates the transforEmotion environment using \code{\link[reticulate]{use_condaenv}}. If the transforEmotion environment does not exist, it will be created using \code{\link[reticulate]{conda_create}}.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'         Aleksandar Tomasevic <atomashevic@gmail.com>
#'
#' @export
#'
# Install miniconda
# Updated 15.11.2023
setup_miniconda <- function()
{

  # Install miniconda
  path_to_miniconda <- try(
    install_miniconda(),
    silent = TRUE
  )

  if(any(class(path_to_miniconda) != "try-error")){
    message("\nTo uninstall miniconda, use `reticulate::miniconda_uninstall()`")
  }

  # Create transformEmotion enviroment if it doesn't exist
  te_ENV <- conda_check()
  if (!te_ENV){
  print("Creating 'transforEmotion' environment...")
  path_to_env <- try(
    conda_create("transforEmotion"),
    silent = TRUE
  )
  }
  # Activate the environment

  reticulate::use_condaenv("transforEmotion", required = TRUE)

  # Check if all required Python libraries are installed (pip-level)
  installed_modules <- suppressMessages(
    reticulate::py_list_packages(envname = "transforEmotion")
  )

  # Extract installed package names without versions
  installed_packages <- installed_modules$package

  # Define required pip modules (subset of setup_modules; exclude conda-only like 'openssl')
  OS <- tolower(Sys.info()["sysname"])  # linux, windows, darwin
  required_modules <- c(
    "numpy", "scipy", "transformers", "torch", "tensorflow-cpu",
    "llama-index", "accelerate", "pandas", "sentence-transformers"
  )
  # bitsandbytes is unsupported on macOS; include only where applicable
  if (OS %in% c("linux", "windows")) {
    required_modules <- c(required_modules, "bitsandbytes")
  }

  # Check for missing modules (pip)
  missing_modules <- required_modules[!required_modules %in% installed_packages]

  if (length(missing_modules) > 0) {
    print("Installing missing Python libraries...")
    setup_modules()
  } else {
    print("All required Python libraries are already installed.")
  }
}
