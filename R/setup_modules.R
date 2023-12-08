#' Install Necessary Python Modules
#'
#' @description Installs modules to compute \code{\link[transforEmotion]{transformer_scores}}. These include
#' \itemize{
#' 
#' \item{pytorch}
#' 
#' \item{torchvison}
#' 
#' \item{torchaudio}
#' 
#' \item{tensorflow}
#' 
#' \item{transformers}
#' 
#' }
#' 
#' @details Installs modules for miniconda using \code{\link[reticulate]{conda_install}}
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#' 
#' @export
#'
# Install modules
# Updated 13.04.2022
setup_modules <- function()
{
  # Check if transforEmotion conda env is being used
  if (!grepl("transforEmotion", reticulate::py_config()$python))
  {
    reticulate::use_condaenv("transforEmotion", required = TRUE)
  }
  
  # Install modules
  message("\nInstalling modules for 'transforEmotion'...")
  
  Sys.sleep(1) # one second pause before the console explodes with text
    reticulate::conda_install("transforEmotion", 
    packages = c(
      "torch", "torchvision",
      "torchaudio", "tensorflow",
      "transformers",
      "pytube",
      "pytz",
      "face_recognition",
      "opencv-python"
    ),
    pip = TRUE
  )
  if (!check_python_libs())
  {
    stop("Not all Python libraries were installed correctly. Please try again.")
  }
  else {
    message("All Python libraries were installed correctly.")
  }
}
