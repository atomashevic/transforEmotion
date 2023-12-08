#' Run FER on YouTube video
#'
#' This function retrieves FER scores a specific number of frames extracted from YouTube video. It uses Python libraries for facial recognition and emotion detection in text, images, and videos.
#'
#' @param video The URL of the YouTube video to analyze.
#' @param classes A character vector specifying the classes to analyze.
#' @param nframes The number of frames to analyze in the video. Default is 100.
#' @param face_selection The method for selecting faces in the video. Options are "largest", "left", or "right". Default is "largest".
#' @param cut Logical indicating whether to cut the video to a specific time range. Default is FALSE.
#' @param start The start time of the video range to analyze. Default is 0.
#' @param end The end time of the video range to analyze. Default is 60.
#' @param uniform Logical indicating whether to uniformly sample frames from the video. Default is FALSE.
#' @param ffreq The frame frequency for sampling frames from the video. Default is 15.
#' @param save_video Logical indicating whether to save the analyzed video. Default is FALSE.
#' @param save_frames Logical indicating whether to save the analyzed frames. Default is FALSE.
#' @param save_dir The directory to save the analyzed frames. Default is "temp/".
#' @param video_name The name of the analyzed video. Default is "temp".
#'
#' @return A result object containing the analyzed video scores.
#'
#' @import reticulate
#'
#' @export
#'
#' @examples
#' # Not run: 
#' result <- video_scores("https://www.youtube.com/watch?v=dQw4w9WgXcQ", c("happy", "sad"), nframes = 200, face_selection = "left", cut = TRUE, start = 30, end = 90, uniform = TRUE, ffreq = 10, save_video = TRUE, save_frames = TRUE, save_dir = "output/", video_name = "analysis")
#'

video_scores <- function(video, classes, nframes=100,
                         face_selection = "largest", cut = FALSE, start = 0, end = 60, uniform = FALSE, ffreq = 15, save_video = FALSE, save_frames = FALSE, save_dir = "temp/", video_name = "temp"){
    ################################################################
    # TODO this piece of code needs to be a function, parametrized by the use of Python libraries: text, image, video
    if (!conda_check()){
    print("Creating and switching to transforEmotion virtual Python environment...")
    Sys.sleep(1)
    setup_miniconda()
  } else
  {
    reticulate::use_condaenv("transforEmotion", required = FALSE)
  }
  if (!check_python_libs()){
    print("Some Python libraries are not available in the transforEmotion conda environment. We are going to set them up. We need them for facial recognition and emotion detection in text, images and video. This may take a while, please be patient.")
    Sys.sleep(1)
    setup_modules()
  }
  
  ################################################################
  reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  reticulate::source_python(system.file("python", "video.py", package = "transforEmotion"))
  if (!grepl("youtu", video)){
    stop("You need to provide a YouTube video URL.")
  }
    # check if classes is a character vector
  if(!is.character(classes)){
    stop("Classes must be a character vector.")
  }
  # check if we have at least 2 classes
  if(length(classes) < 2){
    stop("Classes must have at least 2 elements.")
  }
  # check if face_selection is valid
  if(!face_selection %in% c("largest", "left", "right")){
    stop("Argument face_selection must be one of: largest, left, right")
  }
  # create save_dir
  if(!dir.exists(save_dir)){
      dir.create(save_dir)
  }
  
  result = yt_analyze(url = video, nframes = nframes, labels = classes,
             side= face_selection, cut= cut, start = start, end = end, uniform = uniform, ff = ffreq, save_video = save_video, save_frames = save_frames, frame_dir = save_dir, video_name = video_name)

  if (!save_video){
    # remove video from frame_dir
    file.remove(paste0(save_dir, video_name, ".mp4"))
  }
  if(!save_frames){
     file.remove(list.files(save_dir, pattern = ".jpg"))
  }
  return(result)
}
