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
#' @param face_selection The face(s) to score. Can be "largest", "left", "right",
#'   "none", "all", face numbers, or a data frame of faces. Default is "largest"
#'   and will select the largest face in the image. "left" and "right" will select
#'   the face on the far left or the far right side of the image. "none" will use
#'   the whole image without cropping. These four return one row, as before.
#'   "all" scores every face, face numbers (such as \code{c(1, 3)}) score those
#'   faces, numbered from left to right as \code{\link{detect_faces}} numbers
#'   them, and a data frame from \code{\link{detect_faces}} or
#'   \code{\link{select_faces}} scores its faces in this image without detecting
#'   them again; these return one row per face (see \code{\link{face_scores}}).
#' @param model A string specifying the vision model to use. Options include:
#'   \itemize{
#'     \item Built-in models: "oai-base" (default), "oai-large", "eva-8B", "jina-v2",
#'       "oai-base-fer" and "oai-large-fer" (OpenAI CLIP with the vision encoder
#'       fine-tuned on the FER2013 facial-expression dataset)
#'     \item Any valid HuggingFace model ID, including checkpoints that hold only
#'       a fine-tuned CLIP vision encoder; their text encoder and processor come
#'       from the base CLIP model named in the checkpoint's config
#'     \item Custom registered models (see \code{\link{register_vision_model}})
#'   }
#'   Use \code{\link{list_vision_models}} to see all available models.
#'   Note: Using large or untested models may cause memory issues or crashes.
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
#' @param face_detector \code{"yunet"} (default), OpenCV's YuNet face detector,
#'   or \code{"haar"}, the Haar cascade used before version 0.1.8. Use
#'   \code{"haar"} to reproduce scores from earlier versions.
#' @return A data frame containing the scores for each class.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @importFrom reticulate source_python
#' @importFrom reticulate py
#' @export

image_scores <- function(image, classes, face_selection = "largest", model = "oai-base", local_model_path = NULL,
                         face_detector = c("yunet", "haar")) {
  # Declare Python requirements
  ensure_te_py_env()
  if (.te_uses_eva_adapter(model) && .te_uses_gpu()) .te_require("gpu")
  
  # Suppress TensorFlow messages
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
  
  # Try to import required Python module
  module_import <- try({
    image_module <- reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
    image_module
  }, silent = TRUE)
  
  # If import fails, try setting up modules
  if(inherits(module_import, "try-error")) {
    message("Required Python modules not found. Setting up modules...")
    setup_modules(download_models = FALSE)
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
  selection <- .te_face_selection(face_selection)
  face_detector <- match.arg(face_detector)

  # Validate model using registry system
  # Resolve model name to actual model ID for Python
  actual_model_id <- model
  model_config <- NULL
  model_architecture <- NULL
  
  # Checked outside tryCatch() so the error is not replaced by "not recognized"
  if (!is.null(local_model_path) && !dir.exists(local_model_path)) {
    stop("The specified local_model_path directory does not exist: ", local_model_path)
  }

  tryCatch({
    if (is_vision_model_registered(model)) {
      # Model is registered - get the actual model ID
      model_config <- get_vision_model_config(model)
      actual_model_id <- model_config$model_id
      message("Using registered model: ", model_config$description)
      model_architecture <- model_config$architecture
    } else if (!is.null(local_model_path)) {
      message("Using local model from: ", local_model_path)
    } else {
      # Assume it's a direct HuggingFace model ID
      message("Using model directly from HuggingFace Hub: ", model)
      message("Note: For better support, consider registering custom models with register_vision_model()")
    }
  }, error = function(e) {
    # If registry system fails, provide helpful error message
    available_models <- tryCatch(list_vision_models(), error = function(e2) data.frame())
    if (nrow(available_models) > 0) {
      stop("Model '", model, "' not recognized. Available models: ",
           paste(available_models$name, collapse = ", "),
           "\nUse list_vision_models() to see details or register_vision_model() to add custom models.")
    } else {
      warning("Model registry system not available. Proceeding with model: ", model)
    }
  })
  
  # All faces, face numbers or a faces table: one row per face
  if (selection$mode != "rule") {
    faces <- .te_image_faces(image, selection, face_detector)
    vision <- list(model_id = actual_model_id, architecture = model_architecture)
    return(.te_face_scores(faces, classes, vision, local_model_path))
  }

  result <- without_hf_token({
    reticulate::py$classify_image(
      image = image, 
      labels = classes, 
      face = face_selection, 
      model_name = actual_model_id,  # Use resolved model ID
      local_model_path = local_model_path,
      model_architecture = model_architecture,
      face_detector = face_detector
    )
  })
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
#' @param face_selection Face selection strategy: "largest", "left", "right", or
#'   "none" (one row per image), "all" (one row per face, with faces of the same
#'   person sharing a \code{person_id}; see \code{\link{detect_faces}}), or a
#'   data frame from \code{\link{detect_faces}} (its faces in these images).
#' @param pattern Optional regex to filter images (default supports common formats).
#' @param recursive Whether to search subdirectories (default FALSE).
#' @param model CLIP model alias or HuggingFace model id (see image_scores()).
#' @param local_model_path Optional local path to a pre-downloaded model.
#' @param face_detector \code{"yunet"} (default) or \code{"haar"}; see
#'   \code{\link{image_scores}}.
#' @return A data.frame with columns: image_id and one column per class. With
#'   "all" or a data frame of faces, one row per face, with the columns of
#'   \code{\link{face_scores}} after image_id.
#' @export
image_scores_dir <- function(dir,
                             classes,
                             face_selection = "largest",
                             pattern = "\\.(jpg|jpeg|png|bmp)$",
                             recursive = FALSE,
                             model = "oai-base",
                             local_model_path = NULL,
                             face_detector = c("yunet", "haar")) {
  # Declare Python requirements
  ensure_te_py_env()
  if (.te_uses_eva_adapter(model) && .te_uses_gpu()) .te_require("gpu")

  # Suppress TensorFlow messages
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

  # Validate inputs
  if (!dir.exists(dir)) stop("Directory does not exist: ", dir)
  if (!is.character(classes)) stop("Classes must be a character vector.")
  if (length(classes) < 2) stop("Classes must have at least 2 elements.")
  selection <- .te_face_selection(face_selection, allow_ids = FALSE)
  face_detector <- match.arg(face_detector)

  # Resolve model name to actual model ID for Python (same as image_scores)
  actual_model_id <- model
  model_config <- NULL
  model_architecture <- NULL
  
  # Checked outside tryCatch() so the error is not replaced by "not recognized"
  if (!is.null(local_model_path) && !dir.exists(local_model_path)) {
    stop("The specified local_model_path directory does not exist: ", local_model_path)
  }

  tryCatch({
    if (is_vision_model_registered(model)) {
      model_config <- get_vision_model_config(model)
      actual_model_id <- model_config$model_id
      message("Using registered model: ", model_config$description)
      model_architecture <- model_config$architecture
    } else if (!is.null(local_model_path)) {
      message("Using local model from: ", local_model_path)
    } else {
      message("Using model directly from HuggingFace Hub: ", model)
      message("Note: For better support, consider registering custom models with register_vision_model()")
    }
  }, error = function(e) {
    available_models <- tryCatch(list_vision_models(), error = function(e2) data.frame())
    if (nrow(available_models) > 0) {
      stop("Model '", model, "' not recognized. Available models: ",
           paste(available_models$name, collapse = ", "),
           "\nUse list_vision_models() to see details.")
    } else {
      warning("Model registry system not available. Proceeding with model: ", model)
    }
  })

  # Discover images
  files <- list.files(dir, pattern = pattern, full.names = TRUE, recursive = recursive, ignore.case = TRUE)
  if (length(files) == 0) {
    warning("No images found in directory: ", dir)
    return(data.frame(image_id = character(0), check.names = FALSE))
  }
  # image_id is the file name, so files with the same name in different
  # subdirectories cannot be told apart
  dup <- unique(basename(files)[duplicated(basename(files))])
  if (length(dup)) {
    warning("Several images share a file name, so their image_id is the same: ",
            paste(utils::head(dup, 5), collapse = ", "),
            if (length(dup) > 5) ", ...", call. = FALSE)
  }

  # All faces or a faces table: one row per face
  if (selection$mode != "rule") {
    faces <- if (selection$mode == "table") {
      table <- selection$value
      table[normalizePath(table$source, mustWork = FALSE) %in% normalizePath(files), , drop = FALSE]
    } else {
      detect_faces(files, face_detector = face_detector)
    }
    vision <- list(model_id = actual_model_id, architecture = model_architecture)
    out <- .te_face_scores(faces, classes, vision, local_model_path)
    return(cbind(image_id = basename(out$source), out, stringsAsFactors = FALSE))
  }

  # Import python module (use same environment as image_scores)
  module_import <- try({
    reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  }, silent = TRUE)
  if (inherits(module_import, "try-error")) {
    message("Required Python modules not found. Setting up modules...")
    setup_modules(download_models = FALSE)
    reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  }

  # Call batch classifier in Python (avoid sending tokens for non-gated models)
  df <- without_hf_token({
    reticulate::py$classify_images_batch(
      images = files,
      labels = classes,
      face = face_selection,
      model_name = actual_model_id,  # Use resolved model ID
      local_model_path = local_model_path,
      model_architecture = model_architecture,
      face_detector = face_detector
    )
  })

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

#' @noRd
# Faces of one image for image_scores(): detected (all of them, or the given
# face numbers) or taken from a detect_faces() table
.te_image_faces <- function(image, selection, face_detector) {
  if (selection$mode == "table") {
    faces <- selection$value
    same <- faces$source == image |
      normalizePath(faces$source, mustWork = FALSE) == normalizePath(image, mustWork = FALSE)
    if (!any(same)) {
      stop("face_selection has no faces in this image (", image, ").", call. = FALSE)
    }
    return(select_faces(faces[same, , drop = FALSE]))
  }
  faces <- detect_faces(image, face_detector = face_detector, identify = FALSE)
  if (selection$mode == "ids") faces <- select_faces(faces, ids = selection$value)
  faces
}
