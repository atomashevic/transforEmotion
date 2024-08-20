#' Install Necessary Python Modules
#'
#' @description Installs modules for the \{transforEmotion\} conda environment
#'
#' @details Installs modules for miniconda using \code{\link[reticulate]{conda_install}}
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @export
#'
# Install modules
# Updated 02.08.2024
setup_modules <- function()
{

  # Set necessary modules
  modules <- c(
    "accelerate==0.29.3", "llama-index==0.10.30", "nltk==3.8.1",
    "opencv-python", "pandas==2.1.3", "pypdf==4.0.1", "pytz==2024.1", "qdrant-client==1.8.2",
    "sentencepiece==0.2.0", "sentence-transformers==2.7.0",
    "tensorflow==2.14.1", "torch==2.1.1", "transformers==4.35.0",
    "pytubefix==6.9.2"
  )

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
