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
    reticulate::install_miniconda(),
    silent = TRUE
  )
  
  # Check for try-error
  if(any(class(path_to_miniconda) != "try-error")){
    
  # Give user the deets
  message("\nTo uninstall miniconda, use `reticulate::miniconda_uninstall()`")
    
  }

  # Create transformEmotion enviroment if it doesn't exist

  if (sum(grepl("transforEmotion", reticulate::conda_list()$name)) == 0){
  path_to_env <- try(
    reticulate::conda_create("transforEmotion"),
    silent = TRUE
  )

  # Check for try-error
  if(any(class(path_to_env) != "try-error")){

  # Give user the deets
  message("\nNew Python virtual environment created. To remove it, use: \n `reticulate::conda_remove(\"transforEmotion\")`")
  }

  }

  # Activate the environment

  Sys.unsetenv("RETICULATE_PYTHON")
  reticulate::use_condaenv("transforEmotion", required = TRUE)

  # Check if the enviroment is activated

  if (grepl("transforEmotion", reticulate::py_config()$python)){
  message("\ntransforEmotion Python virtual environment activated")
  } else {
     # throw an error if the environment is not activated
    stop("Not able to setup and activate transforEmotion Python virtual environemnt")
  }

}