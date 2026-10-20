#' Speech Emotion Scores for Segments of an Audio or Video File
#'
#' Splits the audio track of an audio file, a video file or a YouTube video
#' into segments and scores each segment against user-defined classes with a
#' CLAP audio-text model (zero-shot). Each segment's scores are the softmax over
#' the classes of the similarity between the segment's audio embedding and each
#' class's text embedding, as \code{\link{image_scores}} does with CLIP.
#'
#' @param audio Path to an audio or video file (any format FFmpeg reads, such
#'   as WAV, MP3, FLAC, OGG, M4A or MP4), or the URL of a YouTube video (only
#'   its audio is downloaded, to a temporary file that is removed afterwards).
#' @param classes A character vector of at least two classes. CLAP compares
#'   the audio with each class as text, so short descriptions of the voice
#'   (for example \code{"a person speaking angrily"}) can work better than
#'   single words.
#' @param nsegments The maximum number of segments to score. Default is 100.
#' @param segment_length Segment length in seconds, at most 10 (CLAP reads
#'   10 seconds of audio). Default is 5.
#' @param start The start time, in seconds, of the audio range to analyze.
#'   Default is 0.
#' @param end The end time, in seconds, of the audio range to analyze. Default
#'   is -1: the end of the audio.
#' @param uniform Logical. \code{FALSE} (default) scores consecutive segments
#'   from \code{start}, at most \code{nsegments}. \code{TRUE} spreads at most
#'   \code{nsegments} segments evenly over the range, without overlap.
#' @param model The CLAP model: \code{"laion-clap"} (default,
#'   \code{laion/larger_clap_music_and_speech}, trained on general audio,
#'   music and speech) or the Hugging Face ID of another CLAP model, such as
#'   \code{"laion/larger_clap_general"}.
#' @param local_model_path Optional. Path to a local directory that holds a
#'   downloaded CLAP model (the folder with \code{config.json}); it is used
#'   instead of \code{model}.
#'
#' @details
#' The audio is converted to mono at the model's sampling rate (48 kHz).
#' With \code{uniform = FALSE}, a last segment shorter than one second is not
#' scored, unless it is the only segment. A range no longer than
#' \code{segment_length} is scored as one segment. Segments shorter than 10
#' seconds are repeated to fill CLAP's 10-second input.
#'
#' General-purpose CLAP models separate speech emotions much less well than
#' CLIP separates facial expressions. On RAVDESS (1440 acted English clips,
#' eight emotions), \code{"laion-clap"} had an unweighted average recall of
#' 0.13 with one-word classes such as \code{"angry"} and 0.17 with classes
#' such as \code{"a person speaking angrily"}, where chance is 0.125. Check
#' the scores against labelled examples from your own data before you use
#' them.
#'
#' @return A data frame with one row per segment: \code{segment} (1, 2, ...),
#'   \code{start} and \code{end} (the segment's times in the file, in seconds)
#'   and one column of scores per class. The scores of a row sum to 1.
#'
#' @section Data Privacy:
#'   All processing is done locally with the downloaded model,
#'   and your audio is never sent to any remote server or third-party.
#'
#' @seealso \code{\link{video_scores}} for the frames of the same video.
#'
#' @examples
#' \dontrun{
#' speech_scores("interview.wav",
#'               classes = c("a person speaking angrily", "a person speaking happily",
#'                           "a person speaking sadly", "a person speaking calmly"),
#'               nsegments = 20, segment_length = 5)
#' }
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @export
speech_scores <- function(audio, classes, nsegments = 100, segment_length = 5,
                          start = 0, end = -1, uniform = FALSE, model = "laion-clap",
                          local_model_path = NULL) {
  if (!is.character(audio) || length(audio) != 1 || is.na(audio)) {
    stop("audio must be one file path or YouTube URL.", call. = FALSE)
  }
  is_youtube <- .te_is_youtube_url(audio)
  if (grepl("^https?://", audio, ignore.case = TRUE) && !is_youtube) {
    stop("You need to provide a YouTube video URL or a local file.", call. = FALSE)
  }
  if (!is_youtube && !file.exists(audio)) {
    stop("Audio file does not exist: ", audio,
         ". If it is a URL, make sure it includes http:// or https://", call. = FALSE)
  }
  .te_check_speech_classes(classes)
  if (!is.numeric(nsegments) || length(nsegments) != 1 || is.na(nsegments) ||
      nsegments < 1 || nsegments != round(nsegments)) {
    stop("nsegments must be a positive whole number.", call. = FALSE)
  }
  if (!is.numeric(segment_length) || length(segment_length) != 1 ||
      is.na(segment_length) || segment_length <= 0 || segment_length > 10) {
    stop("segment_length must be a number of seconds above 0 and at most 10.",
         call. = FALSE)
  }
  if (!is.numeric(start) || length(start) != 1 || is.na(start) || start < 0) {
    stop("start must be a number of seconds, 0 or more.", call. = FALSE)
  }
  if (!is.numeric(end) || length(end) != 1 || is.na(end) ||
      (end != -1 && end <= start)) {
    stop("end must be -1 (the end of the audio) or greater than start.", call. = FALSE)
  }
  if (!is.null(local_model_path) && !dir.exists(local_model_path)) {
    stop("The specified local_model_path directory does not exist: ", local_model_path)
  }
  model_id <- .te_speech_model_id(model)

  .te_require(c("audio", if (is_youtube) "youtube"))
  te_audio <- .te_audio_py()

  if (is_youtube) {
    dir <- tempfile("te-speech-")
    dir.create(dir)
    on.exit(unlink(dir, recursive = TRUE), add = TRUE)
    audio <- te_audio$download_youtube_audio(audio, dir, "audio")
  }

  res <- without_hf_token(
    te_audio$speech_analyze(
      path = normalizePath(audio, mustWork = TRUE),
      labels = as.list(classes),
      nsegments = as.integer(nsegments),
      segment_length = as.numeric(segment_length),
      start = as.numeric(start),
      end = as.numeric(end),
      uniform = isTRUE(uniform),
      model_name = model_id,
      local_model_path = local_model_path
    )
  )
  .te_speech_frame(res, classes)
}

# Short names for CLAP models
.te_speech_models <- c("laion-clap" = "laion/larger_clap_music_and_speech")

# Columns speech_scores() adds before the class scores
.te_speech_columns <- c("segment", "start", "end")

#' @noRd
.te_speech_model_id <- function(model) {
  if (!is.character(model) || length(model) != 1 || is.na(model) || !nzchar(model)) {
    stop("model must be \"laion-clap\" or a Hugging Face model ID.", call. = FALSE)
  }
  if (model %in% names(.te_speech_models)) .te_speech_models[[model]] else model
}

#' @noRd
.te_check_speech_classes <- function(classes) {
  if (!is.character(classes)) stop("Classes must be a character vector.", call. = FALSE)
  if (length(classes) < 2) stop("Classes must have at least 2 elements.", call. = FALSE)
  if (anyNA(classes) || any(!nzchar(classes))) {
    stop("Classes must not be empty or NA.", call. = FALSE)
  }
  if (anyDuplicated(classes)) stop("Classes must be unique.", call. = FALSE)
  reserved <- intersect(classes, .te_speech_columns)
  if (length(reserved)) {
    stop("Classes cannot be named ", paste(reserved, collapse = ", "),
         ": these are columns of the result.", call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
# The data frame speech_scores() returns, from what speech_analyze() returns
.te_speech_frame <- function(res, classes) {
  starts <- as.numeric(unlist(res$start))
  scores <- matrix(as.numeric(res$scores), nrow = length(starts),
                   ncol = length(classes))
  colnames(scores) <- classes
  data.frame(segment = seq_along(starts), start = starts,
             end = as.numeric(unlist(res$end)), scores, check.names = FALSE)
}

#' @noRd
.te_audio_py <- function() {
  reticulate::import_from_path("te_audio", system.file("python", package = "transforEmotion"))
}
