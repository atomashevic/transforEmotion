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
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = '3')
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
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = '3')
 
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
  
  print("Installing missing Python libraries...")
    setup_modules()
}

