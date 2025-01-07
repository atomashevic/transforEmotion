#' Run FER on a YouTube video using a Hugging Face CLIP model
#'
#' This function retrieves facial expression recognition (FER) scores from a specific number of frames extracted from a YouTube video using a specified Hugging Face CLIP model. It utilizes Python libraries for facial recognition and emotion detection in text, images, and video.
#'
#' @param video The URL of the YouTube video to analyze.
#' @param classes A character vector specifying the classes to analyze.
#' @param nframes The number of frames to analyze in the video. Default is 100.
#' @param face_selection The method for selecting faces in the video. Options are "largest", "left", or "right". Default is "largest".
#' @param start The start time of the video range to analyze. Default is 0.
#' @param end The end time of the video range to analyze. Default is -1 and this means that video won't be cut. If end is a positive number greater than start, the video will be cut from start to end.
#' @param uniform Logical indicating whether to uniformly sample frames from the video. Default is FALSE.
#' @param ffreq The frame frequency for sampling frames from the video. Default is 15.
#' @param save_video Logical indicating whether to save the analyzed video. Default is FALSE.
#' @param save_frames Logical indicating whether to save the analyzed frames. Default is FALSE.
#' @param save_dir The directory to save the analyzed frames. Default is "temp/".
#' @param video_name The name of the analyzed video. Default is "temp".
#' @param model A string specifying the CLIP model to use. Options are:
#'   \itemize{
#'     \item \code{"oai-base"}: "openai/clip-vit-base-patch32" (default)
#'     \item \code{"oai-large"}: "openai/clip-vit-large-patch14"
#'     \item \code{"eva-8B"}: "BAAI/EVA-CLIP-8B-448" (quantized version for reduced memory usage)
#'     \item \code{"jina-v2"}: "jinaai/jina-clip-v2"
#'   }
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
                         video_name = "temp", model = "oai-base") {
  
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

  # Check if model is valid
  valid_models <- c("oai-base", "oai-large", "eva-18B", "eva-8B", "jina-v2")
  if (!model %in% valid_models) {
    stop(paste("Invalid model specified. Allowed models are:", paste(valid_models, collapse = ", ")))
  }
  if(!face_selection %in% c("largest", "left", "right")){
    stop("Argument face_selection must be one of: largest, left, right")
  }

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

  available_models <- c(
    "oai-base" = "openai/clip-vit-base-patch32",
    "oai-large" = "openai/clip-vit-large-patch14",
    "eva-8B" = "BAAI/EVA-CLIP-8B-448",
    "jina-v2" = "jinaai/jina-clip-v2"
  )

  result <- reticulate::py$yt_analyze(url = video, nframes = nframes, labels = classes,
             side = face_selection, start = start, end = end, uniform = uniform, ff = ffreq,
             frame_dir = save_dir, video_name = video_name, model_name = model)

  if (!save_video && grepl("youtu", video)){
    file.remove(paste0(save_dir, video_name, ".mp4"))
  }
  if(!save_frames){
     file.remove(paste0(save_dir, list.files(save_dir, pattern = ".jpg")))
  }
  return(result)
}
