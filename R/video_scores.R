#' Run FER on a YouTube video using a Hugging Face CLIP model
#'
#' This function retrieves facial expression recognition (FER) scores from a specific number of frames extracted from a YouTube video using a specified Hugging Face CLIP model. It utilizes Python libraries for facial recognition and emotion detection in text, images, and video.
#'
#' @param video The URL of the YouTube video to analyze.
#' @param classes A character vector specifying the classes to analyze.
#' @param nframes The number of frames to analyze in the video. Default is 100.
#' @param face_selection The method for selecting faces in the video. Options are
#'   "largest", "left", "right", or "none". Default is "largest". Use "none"
#'   to classify the whole frame without face cropping.
#' @param start The start time of the video range to analyze. Default is 0.
#' @param end The end time of the video range to analyze. Default is -1 and this means that video won't be cut. If end is a positive number greater than start, the video will be cut from start to end.
#' @param uniform Logical indicating whether to uniformly sample frames from the video. Default is FALSE.
#' @param ffreq The frame frequency for sampling frames from the video. Default is 15.
#' @param save_video Logical indicating whether to save the analyzed video. Default is FALSE.
#' @param save_frames Logical indicating whether to save the analyzed frames. Default is FALSE.
#' @param save_dir The directory to save the analyzed frames. Default is "temp/".
#' @param video_name The name of the analyzed video. Default is "temp".
#' @param model A string specifying the vision model to use. Options include:
#'   \itemize{
#'     \item Built-in models: "oai-base" (default), "oai-large", "eva-8B", "jina-v2"
#'     \item Any valid HuggingFace model ID
#'     \item Custom registered models (see \code{\link{register_vision_model}})
#'   }
#'   Use \code{\link{list_vision_models}} to see all available models.
#'   Note: Video processing is memory-intensive, so use caution with large models.
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
#'   depending on your system's resources, especially when processing videos with many frames.
#' @return A result object containing the analyzed video scores.
#'
#' @section Data Privacy:
#'   All processing is done locally with the downloaded model,
#'   and your video frames are never sent to any remote server or third-party.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @import reticulate
#'
#' @export
#

video_scores <- function(video, classes, nframes = 100, face_selection = "largest",
                         start = 0, end = -1, uniform = FALSE, ffreq = 15,
                         save_video = FALSE, save_frames = FALSE, save_dir = "temp/",
                         video_name = "temp", model = "oai-base", local_model_path = NULL) {
  # Ensure reticulate uses the transforEmotion conda environment
  ensure_te_py_env()

  # Suppress TensorFlow messages
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

  # Try to import required Python modules
  modules_import <- try({
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    image_module <- reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
    video_module <- reticulate::source_python(system.file("python", "video.py", package = "transforEmotion"))
    list(image = image_module, video = video_module)
  }, silent = TRUE)

  # If import fails, try setting up modules
  if(inherits(modules_import, "try-error")) {
    message("Required Python modules not found. Setting up modules...")
    setup_modules()
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    image_module <- reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
    video_module <- reticulate::source_python(system.file("python", "video.py", package = "transforEmotion"))
  }

  # check if classes is a character vector
  if(!is.character(classes)){
    stop("Classes must be a character vector.")
  }

  # check if we have at least 2 classes
  if(length(classes) < 2){
    stop("Classes must have at least 2 elements.")
  }

  # Check if face_selection is valid
  if(!face_selection %in% c("largest", "left", "right", "none")){
    stop("Argument face_selection must be one of: largest, left, right, none")
  }

  # Validate model using registry system
  model_architecture <- NULL
  actual_model_id <- model
  tryCatch({
    if (is_vision_model_registered(model)) {
      model_config <- get_vision_model_config(model)
      message("Using registered model: ", model_config$description)
      actual_model_id <- model_config$model_id
      model_architecture <- model_config$architecture
      if (model_config$requires_special_handling) {
        message("Note: This model requires special handling and may be memory-intensive for video processing")
      }
    } else if (!is.null(local_model_path)) {
      if (!dir.exists(local_model_path)) {
        stop("The specified local_model_path directory does not exist.")
      }
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
           "\nUse list_vision_models() to see details or register_vision_model() to add custom models.")
    } else {
      warning("Model registry system not available. Proceeding with model: ", model)
    }
  })

  # create save_dir
  if(!dir.exists(save_dir)){
      dir.create(save_dir)
  }

  if (!grepl("^http", video)) {
    if(!file.exists(video)){
      stop("Video file does not exist. If path is an URL make sure it includes http:// or https://")
    }}
  else { if(!grepl("youtu", video)){
    stop("You need to provide a YouTube video URL.")
  }
  }

  result <- reticulate::py$yt_analyze(
    url = video,
    nframes = nframes,
    labels = classes,
    side = face_selection,
    start = start,
    end = end,
    uniform = uniform,
    ff = ffreq,
    frame_dir = save_dir,
    video_name = video_name,
    model_name = actual_model_id,
    model_architecture = model_architecture,
    local_model_path = local_model_path
  )

  if (!save_video && grepl("youtu", video)){
    file.remove(paste0(save_dir, video_name, ".mp4"))
  }
  if(!save_frames){
     file.remove(paste0(save_dir, list.files(save_dir, pattern = ".jpg")))
  }
  return(result)
}
