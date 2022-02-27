#' Setup Python
#'
#' @description Sets up Python
#' 
#' @param path_to_python Character.
#' Path to specify where "python.exe" is located on your computer.
#' Defaults to \code{NULL}, which will use \code{\link[reticulate]{py_available}}
#' to find available Python or Anaconda
#'
#' @return Returns a message either specifying help or enables Python
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @examples
#' python_setup()
#' 
#' @export
#'
# Setup Python
# Updated 27.02.2022
python_setup <- function(path_to_python = NULL)
{
  
  if(is.null(path_to_python)){
    
    if(reticulate::py_available(TRUE)){
      path_to_python <- reticulate::py_config()$python
    }else{
      message("\nNo path to python configured.\n\nPlease see `browseVignettes(\"transforEmotion\")`: 'Python Setup'")
      return(NULL)
    }
    
  }
  
  # Use python
  reticulate::use_python(path_to_python)
  
  # Message
  message(
    paste("Using Python path:", path_to_python)
  )
  
  # Return path
  return(path_to_python)
  
}
