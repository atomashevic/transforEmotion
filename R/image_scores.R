#' Calculate image scores using a Hugging Face CLIP model
#'
#' This function takes an image file and a vector of classes as input and calculates
#' the scores for each class using a specified Hugging Face CLIP model.
#' Primary use of the function is to calculate FER scores - Facial Expression
#' Detection of emotions based on detected facial expression in images. In case
#' there are more than one face in the image, the function will return the scores
#' of the face selected using the face_selection parameter.
#' If there is no face in the image, the function will return NA for all classes.
#' Function uses reticulate to call the Python functions in the image.py file.
#' If you run this package/function for the first time it will take some time
#' for the package to setup a functioning Python virtual environment in the
#' background. This includes installing Python libraries for facial recognition
#' and emotion detection in text, images and video. Please be patient.
#'
#' Data Privacy: All processing is done locally with the downloaded model,
#' and your images are never sent to any remote server or third-party.
#'
#' @param image The path to the image file or URL of the image.
#' @param classes A character vector of classes to classify the image into.
#' @param face_selection The method to select the face in the image. Can be "largest",
#'   "left", "right", or "none". Default is "largest" and will select the largest face
#'   in the image. "left" and "right" will select the face on the far left or the
#'   far right side of the image. "none" will use the whole image without cropping.
#'   Face_selection method is irrelevant if there is
#'   only one face in the image.
#' @param model A string specifying the CLIP model to use. Options are:
#'   \itemize{
#'     \item \code{"oai-base"}: "openai/clip-vit-base-patch32" (default)
#'     \item \code{"oai-large"}: "openai/clip-vit-large-patch14"
#'     \item \code{"eva-8B"}: "BAAI/EVA-CLIP-8B-448" (quantized version for reduced memory usage)
#'     \item \code{"jina-v2"}: "jinaai/jina-clip-v2"
#'     \item Any valid HuggingFace model ID
#'   }
#'   Note: Using custom HuggingFace model IDs beyond the recommended models is done at your own risk.
#'   Large models may cause memory issues or crashes, especially on systems with limited resources.
#'   The package has been optimized and tested with the recommended models listed above.
#' @param local_model_path Optional. Path to a local directory containing a pre-downloaded 
#'   HuggingFace model. If provided, the model will be loaded from this directory instead
#'   of being downloaded from HuggingFace. This is useful for offline usage or for using
#'   custom fine-tuned models. 
#'   
#'   On Linux/Mac, look in ~/.cache/huggingface/hub/ folder for downloaded models. 
#'   Navigate to the snapshots folder for the relevant model and point to the directory 
#'   which contains the config.json file. For example: 
#'   "/home/username/.cache/huggingface/hub/models--cross-encoder--nli-distilroberta-base/snapshots/b5b020e8117e1ddc6a0c7ed0fd22c0e679edf0fa/"
#'   
#'   On Windows, the base path is C:\\Users\\USERNAME\\.cache\\huggingface\\transformers\\
#'   
#'   Warning: Using very large models from local paths may cause memory issues or crashes 
#'   depending on your system's resources.
#' @return A data frame containing the scores for each class.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @importFrom reticulate source_python
#' @importFrom reticulate py
#' @export

image_scores <- function(image, classes, face_selection = "largest", model = "oai-base", local_model_path = NULL) {
  
  # Suppress TensorFlow messages
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
  
  # Try to import required Python module
  module_import <- try({
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    image_module <- reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
    image_module
  }, silent = TRUE)
  
  # If import fails, try setting up modules
  if(inherits(module_import, "try-error")) {
    message("Required Python modules not found. Setting up modules...")
    setup_modules()
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    image_module <- reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  }

  # check if image has image file extension
  if (!grepl("\\.(jpg|jpeg|png|bmp)$", image)) {
    stop("Image file name must have an image file extension: jpg, jpeg, png, bmp")
  }
  # if not url check if file exists
  if (!grepl("^http", image)) {
    if (!file.exists(image)) {
      stop("Image file does not exist. If path is an URL make sure it includes http:// or https://")
    }
  }
  # check if classes is a character vector
  if (!is.character(classes)) {
    stop("Classes must be a character vector.")
  }
  # check if we have at least 2 classes
  if (length(classes) < 2) {
    stop("Classes must have at least 2 elements.")
  }

  # check if face_selection is valid
  if (!face_selection %in% c("largest", "left", "right", "none")) {
  stop("Argument face_selection must be one of: largest, left, right, none")
  }
  
  # Check if model is valid when using predefined shortcuts
  valid_models <- c("oai-base", "oai-large", "eva-18B", "eva-8B", "jina-v2")
  if (model %in% valid_models) {
    # Using a predefined model
    available_models <- c(
      "oai-base" = "openai/clip-vit-base-patch32",
      "oai-large" = "openai/clip-vit-large-patch14",
      "eva-8B" = "BAAI/EVA-CLIP-8B-448",
      "jina-v2" = "jinaai/jina-clip-v2"
    )
  } else if (!is.null(local_model_path)) {
    # Using a local model path - validate it exists
    if (!dir.exists(local_model_path)) {
      stop("The specified local_model_path directory does not exist.")
    }
    message("Using local model from: ", local_model_path)
  } else {
    # Assume it's a direct HuggingFace model ID
    message("Using model directly from HuggingFace Hub: ", model)
  }
  
  # Check if local_model_path exists if provided
  if (!is.null(local_model_path) && !dir.exists(local_model_path)) {
    stop("The specified local_model_path directory does not exist: ", local_model_path)
  }

  result <- reticulate::py$classify_image(
    image = image, 
    labels = classes, 
    face = face_selection, 
    model_name = model,
    local_model_path = local_model_path
  )
  if (is.null(result)) {
    # No face found or classification failed; return NA row for each class
    out <- as.list(rep(NA_real_, length(classes)))
    names(out) <- classes
    return(as.data.frame(out))
  }
  result <- as.data.frame(result)
  return(result)
}

#' Calculate image scores for all images in a directory (fast batch)
#'
#' This function scans a directory for image files and computes scores for each
#' image using a Hugging Face CLIP model. It loads the model once and reuses
#' text embeddings for speed, returning one row per image with the filename as
#' image_id and probability columns for each class.
#'
#' @param dir Path to a directory containing images.
#' @param classes Character vector of labels/classes (length >= 2).
#' @param face_selection Face selection strategy: "largest", "left", "right", or "none".
#' @param pattern Optional regex to filter images (default supports common formats).
#' @param recursive Whether to search subdirectories (default FALSE).
#' @param model CLIP model alias or HuggingFace model id (see image_scores()).
#' @param local_model_path Optional local path to a pre-downloaded model.
#' @return A data.frame with columns: image_id and one column per class.
#' @export
image_scores_dir <- function(dir,
                             classes,
                             face_selection = "largest",
                             pattern = "\\.(jpg|jpeg|png|bmp)$",
                             recursive = FALSE,
                             model = "oai-base",
                             local_model_path = NULL) {

  # Suppress TensorFlow messages
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

  # Validate inputs
  if (!dir.exists(dir)) stop("Directory does not exist: ", dir)
  if (!is.character(classes)) stop("Classes must be a character vector.")
  if (length(classes) < 2) stop("Classes must have at least 2 elements.")
  if (!face_selection %in% c("largest", "left", "right", "none")) {
  stop("Argument face_selection must be one of: largest, left, right, none")
  }

  # Validate model inputs similar to image_scores
  valid_models <- c("oai-base", "oai-large", "eva-18B", "eva-8B", "jina-v2")
  if (model %in% valid_models) {
    available_models <- c(
      "oai-base" = "openai/clip-vit-base-patch32",
      "oai-large" = "openai/clip-vit-large-patch14",
      "eva-8B" = "BAAI/EVA-CLIP-8B-448",
      "jina-v2" = "jinaai/jina-clip-v2"
    )
  } else if (!is.null(local_model_path)) {
    if (!dir.exists(local_model_path)) {
      stop("The specified local_model_path directory does not exist.")
    }
    message("Using local model from: ", local_model_path)
  } else {
    message("Using model directly from HuggingFace Hub: ", model)
  }

  # Discover images
  files <- list.files(dir, pattern = pattern, full.names = TRUE, recursive = recursive, ignore.case = TRUE)
  if (length(files) == 0) {
    warning("No images found in directory: ", dir)
    return(data.frame(image_id = character(0), check.names = FALSE))
  }

  # Import python module (use same environment as image_scores)
  module_import <- try({
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  }, silent = TRUE)
  if (inherits(module_import, "try-error")) {
    message("Required Python modules not found. Setting up modules...")
    setup_modules()
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  }

  # Call batch classifier in Python
  df <- reticulate::py$classify_images_batch(
    images = files,
    labels = classes,
    face = face_selection,
    model_name = model,
    local_model_path = local_model_path
  )

  # Ensure data.frame and column order
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  # If Python returns no rows, create an empty frame with expected columns
  expected_cols <- c("image_id", classes)
  for (col in expected_cols) {
    if (!col %in% names(df)) df[[col]] <- NA_real_
  }
  df <- df[expected_cols]
  return(df)
}

