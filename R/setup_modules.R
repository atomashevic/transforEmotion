#' Install Necessary Python Modules
#'
#' @description Installs modules for the \{transforEmotion\} conda environment.
#'
#' @details This function installs the required Python modules for the \{transforEmotion\} conda environment using \code{\link[reticulate]{conda_install}}. Ensure that miniconda is installed and properly configured before running this function.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
# Updated 13.09.2024

setup_modules <- function() {
  # Set necessary modules
  modules <- c(
    "accelerate==0.29.3", "llama-index==0.10.30", "nltk==3.8.1",
    "opencv-python", "pandas==2.1.3", "pypdf==4.0.1", "pytz==2024.1",
    "qdrant-client==1.8.2", "sentencepiece==0.2.0",
    "sentence-transformers==2.7.0", "tensorflow-cpu==2.14.1",
    "torch==2.1.1+cpu", "transformers==4.35.0", "pytubefix==6.9.2"
  )

  # Determine whether any modules need to be installed
  installed_modules <- reticulate::py_list_packages(envname = "transforEmotion")

  # Extract installed package names without versions
  installed_packages <- installed_modules$package

  # Remove version numbers from modules list for comparison
  modules_no_versions <- sub("(.*)==.*", "\\1", modules)

  # Determine missing modules
  missing_modules <- modules[!modules_no_versions %in% installed_packages]

  # Determine if modules need to be installed
  if (length(missing_modules) != 0) {
    # Send message to user about how many modules are being installed
    message("\nThere are ", length(missing_modules), " modules that need to be installed.")

    # Update pip
    message("\nUpdating pip first...")
    reticulate::conda_install("transforEmotion", packages = "pip", pip_options = "--upgrade", pip = TRUE)

    message("\nInstalling modules for 'transforEmotion'...")

    # Install modules with pip options for PyTorch CPU wheels
    reticulate::conda_install(
      envname = "transforEmotion",
      packages = missing_modules,
      pip_options = c(
        "--upgrade",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu"
      ),
      pip = TRUE
    )
  } else {
    message("\nAll modules are already installed.")
  }
}
