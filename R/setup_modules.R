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
#'
# Install modules
# Updated 11.12.2023

setup_modules <- function()
{
  
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
      "pandas",
      "opencv-python"
    ),
    pip = TRUE
  )
}
