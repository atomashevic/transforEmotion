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

#' Check if required Python libraries are installed in a Conda environment
#'
#' This function checks if a list of required Python libraries are installed in a specified Conda environment.
#'
#' @return A logical value indicating whether all the required Python libraries are installed.
#' reticulate::py_module_available("transformers")
check_python_libs <- function() {

  python_libs <- c("transformers", "torch", "torchvision", "torchaudio", "tensorflow", "pytube", "pytz", "face_recognition", "cv2")

  # Extract the names of the libraries
  lib_names <- sapply(strsplit(python_libs, " "), `[`, 1)
  libs_installed <- logical(length(python_libs))

  # Check if each Python library is installed
  for (i in seq_along(python_libs)) {
    python_lib <- python_libs[i]
    if (!reticulate::py_module_available(python_lib)) {
      print(paste("Python library", python_lib, "is not installed in Conda environment"))
      libs_installed[i] <- FALSE
    } else{
      libs_installed[i] <- TRUE
    }
  }
  return(all(libs_installed))
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
    reticulate::install_miniconda(),
    silent = TRUE
  )
  
  # Check for try-error
  if(any(class(path_to_miniconda) != "try-error")){
    
  # Give user the deets
  message("\nTo uninstall miniconda, use `reticulate::miniconda_uninstall()`")
    
  }

  # Create transformEmotion enviroment if it doesn't exist
  te_ENV <- conda_check()

  if (!te_ENV){
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
  if (!check_python_libs())
  {
    print("Installing missing Python libraries...")
    setup_modules()
  }

  } else {
     # throw an error if the environment is not activated
    print("Your active Python environment is:")
    print(reticulate::py_config()$python)
    stop("Please activate the transforEmotion Python environment instead: `reticulate::use_condaenv(\"transforEmotion\", required = TRUE)`")
  }

}

