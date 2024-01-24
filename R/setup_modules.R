#' Install Necessary Python Modules
#'
#' @description Installs modules for the {transforEmotion} conda environment
#'
#' @details Installs modules for miniconda using \code{\link[reticulate]{conda_install}}
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @noRd
#'
# Install modules
# Updated 24.01.2024
setup_modules <- function()
{

  # Set necessary modules
  modules <- c(
    "accelerate", "llama-index", "nltk",
    "opencv-python", "pandas", "pypdf",
    "pytube", "pytz", "qdrant-client",
    "tensorflow", "torch", "torchaudio",
    "torchvision", "transformers"
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
