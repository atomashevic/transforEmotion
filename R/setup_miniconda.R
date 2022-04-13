#' Install Miniconda
#'
#' @description Installs miniconda
#'
#' @details Installs miniconda using \code{\link[reticulate]{install_miniconda}}
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#' 
#' @export
#'
# Install miniconda
# Updated 13.04.2022
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
  
}
