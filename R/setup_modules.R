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
  
  # Install modules
  message("\nInstalling modules for 'transforEmotion'...")
  
  Sys.sleep(1) # one second pause before the console explodes with text
  
  # Actually install the modules
  reticulate::conda_install("transforEmotion", 
    packages = c(
      "torch", "torchvision",
      "torchaudio", "tensorflow",
      "transformers"
    ),
    pip = TRUE
  )
  
}