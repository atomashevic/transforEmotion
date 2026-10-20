#' Detect faces in images or a video
#'
#' Finds every face in a set of images or in frames sampled from a video, so
#' you can see which faces the package found, pick the ones to analyse (with
#' \code{\link{select_faces}} or ordinary subsetting) and score only those with
#' \code{\link{face_scores}}.
#'
#' Faces are numbered from left to right within each image or frame. In a
#' video, a face is followed from frame to frame and keeps its
#' \code{track_id}. With \code{identify = TRUE}, faces of the same person get
#' the same \code{person_id}: across all images, or across all tracks of the
#' video, so a person who leaves the picture and comes back, or who appears
#' in several images, is one person.
#'
#' @param x Image paths or URLs, a directory of images, or one video (a file
#'   or a YouTube URL).
#' @param face_detector \code{"yunet"} (default), OpenCV's YuNet detector, or
#'   \code{"haar"}, the Haar cascade transforEmotion used before version
#'   0.1.8. YuNet finds small, tilted and partly turned faces and reports a
#'   confidence score and five landmarks; Haar finds frontal faces only.
#' @param min_confidence YuNet's confidence threshold, between 0 and 1.
#'   Ignored for Haar.
#' @param identify Logical. Give faces of the same person the same
#'   \code{person_id}, using OpenCV's SFace face recognizer. Needs YuNet. The
#'   recognition model (10 MB) is downloaded on first use; see Details.
#' @param identity_threshold Cosine similarity between faces (or video tracks)
#'   above which they count as the same person. The default, 0.363, is the
#'   threshold SFace's authors recommend; raise it if different people share
#'   a \code{person_id}, lower it if one person gets several.
#' @param nframes,start,end,uniform,ffreq How frames are sampled from a video;
#'   see \code{\link{video_scores}}. Use the same values in both functions to
#'   get the same frames.
#' @param save_frames Logical. Keep the extracted video frames in
#'   \code{save_dir}. Default is \code{FALSE}.
#' @param save_video Logical. Keep a downloaded YouTube video in
#'   \code{save_dir}. Default is \code{FALSE}. The video has no sound when
#'   YouTube offers no stream with video and audio in one file.
#' @param save_dir Directory for video frames. Default is a temporary
#'   directory.
#' @param video_name Name used for the frame files
#'   (\code{<video_name>-frame-<i>.jpg}).
#' @param pattern,recursive When \code{x} is a directory: a regular expression
#'   for the image files and whether to search subdirectories.
#'
#' @details
#' Face recognition compares faces through SFace embeddings (128 numbers per
#' face). Faces are grouped by average-linkage clustering on cosine
#' similarity; a video track counts as one face, the average of its frames.
#' Two faces in the same image can be the same person (mirrors, screens,
#' posters). The embeddings stay in memory and are never returned or written
#' to disk, and \code{person_id}s only label faces within one call: they do
#' not name anyone and do not match ids from other calls.
#'
#' Grouping faces by person is processing of biometric data under the EU
#' General Data Protection Regulation (Article 9); make sure your ethics
#' approval covers it, or use \code{identify = FALSE}.
#'
#' The YuNet model ships with the package. The SFace model
#' (\code{face_recognition_sface_2021dec_int8.onnx}, from OpenCV's model zoo,
#' Apache 2.0 licence) is downloaded into the package cache on first use, or
#' by \code{\link{setup_modules}}. To use a copy elsewhere, for example on a
#' computer without internet access, set the environment variable
#' \code{TRANSFOREMOTION_SFACE_MODEL} to its path. Without the model,
#' \code{person_id} is \code{NA}.
#'
#' @return A data frame with one row per face and the columns:
#' \describe{
#'   \item{source}{The image path or URL, or the video.}
#'   \item{frame, frame_index, time}{For a video: the sample number (as in
#'     the frame file name), the frame's index in the video and its time in
#'     seconds. \code{NA} for images.}
#'   \item{face}{Face number within the image or frame, from left to right.}
#'   \item{track_id}{For a video: the face's track. \code{NA} for images.}
#'   \item{person_id}{\code{"p1"}, \code{"p2"}, ... numbered by first
#'     appearance; \code{NA} without recognition.}
#'   \item{x, y, width, height}{The face box in pixels, from the top-left
#'     corner of the image.}
#'   \item{confidence}{YuNet's score; \code{NA} for Haar.}
#'   \item{detector}{\code{"yunet"} or \code{"haar"}.}
#'   \item{image_width, image_height}{Size of the image or frame in pixels.}
#'   \item{right_eye_x, ..., mouth_left_y}{The five YuNet landmarks
#'     (\code{NA} for Haar).}
#' }
#' The attribute \code{"media"} holds one row per image or frame, with the
#' number of faces found (\code{n_faces}), so images without faces are not
#' lost. The attribute \code{"settings"} records the detector and, for a
#' video, how frames were sampled, which \code{\link{face_scores}} uses to
#' find the same frames again.
#'
#' @section Data Privacy:
#'   All processing is done locally; images and frames are never sent to a
#'   remote server.
#'
#' @seealso \code{\link{select_faces}}, \code{\link{face_scores}},
#'   \code{\link{draw_faces}}
#'
#' @examples
#' \dontrun{
#' image <- system.file("extdata", "two-people.jpg", package = "transforEmotion")
#' faces <- detect_faces(image)
#' faces[, c("face", "person_id", "x", "y", "width", "height", "confidence")]
#' draw_faces(faces)
#'
#' # Every face of one person in a video, scored once per sampled frame
#' video <- system.file("extdata", "two-people.mp4", package = "transforEmotion")
#' faces <- detect_faces(video, nframes = 20)
#' table(faces$track_id, faces$person_id)
#' scores <- face_scores(select_faces(faces, persons = "p1"),
#'                       classes = c("happy", "sad", "angry", "neutral"))
#' }
#'
#' @export
detect_faces <- function(x, face_detector = c("yunet", "haar"), min_confidence = 0.6,
                         identify = TRUE, identity_threshold = 0.363,
                         nframes = 100, start = 0, end = -1, uniform = FALSE,
                         ffreq = 15, save_frames = FALSE, save_video = FALSE,
                         save_dir = tempdir(), video_name = "temp",
                         pattern = "\\.(jpg|jpeg|png|bmp)$", recursive = FALSE) {
  face_detector <- match.arg(face_detector)
  input <- .te_face_inputs(x, pattern, recursive)
  ensure_te_py_env()
  if (input$type == "youtube") .te_require("youtube")
  te_faces <- .te_faces_py()

  sface <- NULL
  if (isTRUE(identify)) {
    if (face_detector == "yunet") {
      sface <- .te_sface_model()
    } else {
      message("Face recognition needs face_detector = \"yunet\"; person_id is NA.")
    }
  }
  settings <- list(type = if (input$type == "images") "images" else "video",
                   face_detector = face_detector, min_confidence = min_confidence)

  if (input$type == "images") {
    res <- te_faces$faces_table(
      as.list(input$paths), detector = face_detector, min_confidence = min_confidence,
      yunet_model = .te_yunet_model(), sface_model = sface,
      identity_threshold = identity_threshold
    )
    return(.te_faces_frame(res, input$paths, face_detector, settings))
  }

  # Video: extract the frames video_scores() would sample
  if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE)
  is_youtube <- input$type == "youtube"
  cleanup <- .te_video_cleanup_plan(save_dir, video_name, nframes, save_frames,
                                    save_video || !is_youtube)
  on.exit(cleanup(), add = TRUE)
  video_path <- if (is_youtube) {
    te_faces$download_youtube(input$paths, save_dir, video_name)
  } else {
    input$paths
  }
  frames <- te_faces$extract_frames(video_path, nframes, start = start, end = end,
                                    uniform = uniform, ff = ffreq,
                                    frame_dir = save_dir, video_name = video_name)
  settings <- c(settings, list(nframes = nframes, start = start, end = end,
                               uniform = uniform, ffreq = ffreq))
  paths <- vapply(frames, function(f) f$path, character(1))
  res <- te_faces$faces_table(
    as.list(paths), detector = face_detector, min_confidence = min_confidence,
    yunet_model = .te_yunet_model(), frames = frames, sface_model = sface,
    identity_threshold = identity_threshold
  )
  out <- .te_faces_frame(res, rep(input$paths, length(paths)), face_detector, settings)
  media <- attr(out, "media")
  media$frame_path <- if (isTRUE(save_frames)) paths else rep(NA_character_, length(paths))
  attr(out, "media") <- media
  out
}

#' Select faces to analyse
#'
#' Keeps the faces of a \code{\link{detect_faces}} result that a rule, face
#' numbers, tracks or people pick. The result can be passed to
#' \code{\link{face_scores}}, or as \code{face_selection} to
#' \code{\link{image_scores}} or \code{\link{video_scores}}. Ordinary
#' subsetting (for example \code{faces[faces$person_id == "p1", ]}) works as
#' well.
#'
#' @param faces A data frame returned by \code{\link{detect_faces}}.
#' @param rule \code{"all"} (default), or \code{"largest"}, \code{"left"} or
#'   \code{"right"} to keep one face per image or frame, with the rules of
#'   \code{face_selection} in \code{\link{image_scores}}.
#' @param ids Face numbers to keep (the \code{face} column).
#' @param tracks Video track ids to keep (the \code{track_id} column).
#' @param persons Person ids to keep (the \code{person_id} column), for
#'   example \code{"p1"}.
#'
#' @return The selected rows of \code{faces}, with its attributes.
#'
#' @examples
#' faces <- data.frame(
#'   source = "group.jpg", frame = NA, face = 1:3, person_id = c("p1", "p2", "p3"),
#'   x = c(10, 200, 400), y = 50, width = c(80, 120, 60), height = c(80, 120, 60)
#' )
#' select_faces(faces, rule = "largest")
#' select_faces(faces, persons = c("p1", "p3"))
#'
#' @export
select_faces <- function(faces, rule = c("all", "largest", "left", "right"),
                         ids = NULL, tracks = NULL, persons = NULL) {
  rule <- match.arg(rule)
  .te_check_faces_table(faces)
  for (arg in c("ids", "tracks", "persons")) {
    column <- c(ids = "face", tracks = "track_id", persons = "person_id")[[arg]]
    if (!is.null(get(arg)) && !column %in% names(faces)) {
      stop("faces has no ", column, " column to select ", arg, " from.", call. = FALSE)
    }
  }
  keep <- rep(TRUE, nrow(faces))
  if (!is.null(ids)) keep <- keep & faces$face %in% ids
  if (!is.null(tracks)) keep <- keep & faces$track_id %in% tracks
  if (!is.null(persons)) keep <- keep & faces$person_id %in% persons
  if (rule != "all" && any(keep)) {
    frame <- if ("frame" %in% names(faces)) faces$frame else NA
    group <- paste(faces$source, frame, sep = "\r")
    pick <- switch(rule,
      largest = function(i) i[which.max(faces$width[i] * faces$height[i])],
      left = function(i) i[which.min(faces$x[i])],
      right = function(i) i[which.max(faces$x[i])]
    )
    rows <- which(keep)
    chosen <- unlist(lapply(split(rows, group[rows]), pick), use.names = FALSE)
    keep <- seq_len(nrow(faces)) %in% chosen
  }
  out <- faces[keep, , drop = FALSE]
  rownames(out) <- NULL
  for (a in c("media", "settings")) attr(out, a) <- attr(faces, a)
  class(out) <- class(faces)
  out
}

#' Score selected faces
#'
#' Scores each face of a \code{\link{detect_faces}} result with a CLIP model,
#' so you can analyse exactly the faces you picked, for example one person
#' across a whole video.
#'
#' @param faces A data frame returned by \code{\link{detect_faces}} (or a
#'   subset of its rows).
#' @param classes A character vector of classes to score.
#' @param model,local_model_path The vision model; see
#'   \code{\link{image_scores}}.
#' @param padding Share of the face box's width and height added on each side
#'   before scoring. Default is 0.25.
#'
#' @details
#' For a video, the frames are extracted again with the sampling settings
#' \code{\link{detect_faces}} used (and the video downloaded again from
#' YouTube), unless the frames were kept with \code{save_frames = TRUE}.
#'
#' @return \code{faces} without the landmark columns, plus one column of
#'   scores per class.
#'
#' @section Data Privacy:
#'   All processing is done locally with the downloaded model.
#'
#' @seealso \code{\link{detect_faces}}, \code{\link{select_faces}}
#'
#' @examples
#' \dontrun{
#' image <- system.file("extdata", "boris-1.png", package = "transforEmotion")
#' faces <- detect_faces(image)
#' face_scores(faces, classes = c("happy", "sad", "angry", "neutral"))
#' }
#'
#' @export
face_scores <- function(faces, classes, model = "oai-base", local_model_path = NULL,
                        padding = 0.25) {
  .te_check_faces_table(faces)
  if (!is.character(classes)) stop("Classes must be a character vector.")
  if (length(classes) < 2) stop("Classes must have at least 2 elements.")
  ensure_te_py_env()
  vision <- .te_resolve_vision_model(model, local_model_path)
  .te_face_scores(faces, classes, vision, local_model_path, padding)
}

#' @noRd
# face_scores() for an already resolved model: list(model_id, architecture)
.te_face_scores <- function(faces, classes, vision, local_model_path = NULL,
                            padding = 0.25) {
  out <- as.data.frame(unclass(faces)[setdiff(names(faces), .te_landmark_columns)],
                       stringsAsFactors = FALSE, check.names = FALSE)
  for (cls in classes) out[[cls]] <- rep(NA_real_, nrow(out))
  if (!nrow(faces)) return(out)

  settings <- attr(faces, "settings")
  is_video <- .te_is_video_faces(faces, settings)
  paths <- if (is_video) .te_face_frame_paths(faces, settings) else faces$source
  if (is_video) {
    cleanup <- attr(paths, "cleanup")
    on.exit(cleanup(), add = TRUE)
  }
  paths <- as.character(paths)

  reticulate::source_python(system.file("python", "image.py", package = "transforEmotion"))
  # Rows of one image are scored together, so each image is read once
  order_rows <- order(paths)
  scores <- without_hf_token({
    reticulate::py$classify_face_boxes(
      images = as.list(unname(paths[order_rows])),
      boxes = lapply(order_rows, function(i) {
        c(faces$x[i], faces$y[i], faces$width[i], faces$height[i])
      }),
      labels = as.list(classes),
      model_name = vision$model_id,
      local_model_path = local_model_path,
      model_architecture = vision$architecture,
      padding = padding
    )
  })
  scores <- as.data.frame(scores, check.names = FALSE)
  for (cls in classes) out[[cls]][order_rows] <- scores[[cls]]
  out
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

.te_landmark_columns <- paste0(
  rep(c("right_eye", "left_eye", "nose", "mouth_right", "mouth_left"), each = 2),
  c("_x", "_y")
)

#' @noRd
# The te_faces Python module (inst/python/te_faces.py)
.te_faces_py <- function() {
  reticulate::import_from_path("te_faces", system.file("python", package = "transforEmotion"))
}

#' @noRd
.te_yunet_model <- function() {
  system.file("extdata", "face_detection_yunet_2023mar.onnx", package = "transforEmotion",
              mustWork = TRUE)
}

# OpenCV model zoo, tag 4.10.0
.te_sface <- list(
  file = "face_recognition_sface_2021dec_int8.onnx",
  url = paste0("https://media.githubusercontent.com/media/opencv/opencv_zoo/4.10.0/",
               "models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx"),
  size = 9896933,
  md5 = "83938fe16d629ff1dbfdccff1004bf0f"
)

#' @noRd
# Path of the SFace face recognition model, downloaded on first use into the
# package cache (or to TRANSFOREMOTION_SFACE_MODEL, when that file does not
# exist yet). NULL, with a message, when it cannot be had.
.te_sface_model <- function(download = TRUE) {
  path <- Sys.getenv("TRANSFOREMOTION_SFACE_MODEL", unset = "")
  if (!nzchar(path)) path <- file.path(.te_user_cache_dir(), "face_models", .te_sface$file)
  if (file.exists(path)) return(normalizePath(path))
  unavailable <- function(why) {
    message("Face recognition model not available (", why, "); person_id is NA. ",
            "See ?detect_faces.")
    NULL
  }
  if (!download) return(unavailable("not downloaded"))
  if (.te_hf_offline()) return(unavailable("HF_HUB_OFFLINE is set"))

  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  part <- paste0(path, ".part")
  on.exit(unlink(part), add = TRUE)
  message("Downloading the face recognition model (10 MB) ...")
  ok <- tryCatch(
    utils::download.file(.te_sface$url, part, mode = "wb", quiet = TRUE) == 0,
    error = function(e) FALSE, warning = function(w) FALSE
  )
  if (!ok || !file.exists(part)) return(unavailable("download failed"))
  if (file.size(part) != .te_sface$size ||
      unname(tools::md5sum(part)) != .te_sface$md5) {
    return(unavailable("downloaded file is corrupt"))
  }
  file.rename(part, path)
  normalizePath(path)
}

#' @noRd
# What detect_faces() was given: images (paths or URLs, or a directory), a
# video file, or a YouTube URL
.te_face_inputs <- function(x, pattern, recursive) {
  if (!is.character(x) || !length(x) || anyNA(x)) {
    stop("x must be image paths or URLs, a directory, or one video.", call. = FALSE)
  }
  video_ext <- "\\.(mp4|avi|mov|mkv|webm|m4v|mpg|mpeg|wmv)$"
  if (length(x) == 1 && .te_is_youtube_url(x)) {
    return(list(type = "youtube", paths = x))
  }
  if (length(x) == 1 && grepl(video_ext, x, ignore.case = TRUE)) {
    if (!file.exists(x)) stop("Video file does not exist: ", x, call. = FALSE)
    return(list(type = "video", paths = x))
  }
  if (length(x) == 1 && dir.exists(x)) {
    files <- list.files(x, pattern = pattern, full.names = TRUE,
                        recursive = recursive, ignore.case = TRUE)
    if (!length(files)) stop("No images found in directory: ", x, call. = FALSE)
    return(list(type = "images", paths = files))
  }
  local <- !grepl("^https?://", x)
  missing <- x[local & !file.exists(x)]
  if (length(missing)) {
    stop("Image file does not exist: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  list(type = "images", paths = x)
}

#' @noRd
# Python's faces_table() result -> the detect_faces() data frame
.te_faces_frame <- function(res, sources, face_detector, settings) {
  col <- function(d, name, n) {
    v <- unlist(d[[name]])
    if (is.null(v)) v <- rep(NA, n)
    if (is.numeric(v)) v[is.nan(v)] <- NA
    v
  }
  faces <- res[[1]]
  media <- res[[2]]
  n <- length(unlist(faces$item))
  item <- as.integer(col(faces, "item", n))
  person <- as.character(col(faces, "person_id", n))
  person[!nzchar(person)] <- NA_character_
  out <- data.frame(
    source = as.character(sources[item]),
    frame = as.integer(col(faces, "frame", n)),
    frame_index = as.integer(col(faces, "frame_index", n)),
    time = as.numeric(col(faces, "time", n)),
    face = as.integer(col(faces, "face", n)),
    track_id = as.integer(col(faces, "track_id", n)),
    person_id = person,
    x = as.numeric(col(faces, "x", n)),
    y = as.numeric(col(faces, "y", n)),
    width = as.numeric(col(faces, "width", n)),
    height = as.numeric(col(faces, "height", n)),
    confidence = as.numeric(col(faces, "confidence", n)),
    detector = rep(face_detector, n),
    image_width = as.integer(col(faces, "image_width", n)),
    image_height = as.integer(col(faces, "image_height", n)),
    stringsAsFactors = FALSE
  )
  for (lm in .te_landmark_columns) out[[lm]] <- as.numeric(col(faces, lm, n))

  m <- length(unlist(media$item))
  attr(out, "media") <- data.frame(
    source = as.character(sources[as.integer(col(media, "item", m))]),
    frame = as.integer(col(media, "frame", m)),
    frame_index = as.integer(col(media, "frame_index", m)),
    time = as.numeric(col(media, "time", m)),
    image_width = as.integer(col(media, "image_width", m)),
    image_height = as.integer(col(media, "image_height", m)),
    n_faces = as.integer(col(media, "n_faces", m)),
    stringsAsFactors = FALSE
  )
  attr(out, "settings") <- settings
  class(out) <- c("te_faces", "data.frame")
  out
}

#' Keep the attributes of a detect_faces() table when it is subset
#' @param x A data frame returned by \code{\link{detect_faces}}.
#' @param ... Passed to the data frame method.
#' @return The subset, with the \code{"media"} and \code{"settings"} attributes
#'   \code{\link{face_scores}} needs to find video frames again.
#' @keywords internal
#' @export
`[.te_faces` <- function(x, ...) {
  out <- NextMethod()
  if (is.data.frame(out)) {
    for (a in c("media", "settings")) attr(out, a) <- attr(x, a)
    class(out) <- class(x)
  }
  out
}

#' @noRd
.te_check_faces_table <- function(faces) {
  needed <- c("source", "x", "y", "width", "height")
  if (!is.data.frame(faces) || !all(needed %in% names(faces))) {
    stop("faces must be a data frame from detect_faces(), with the columns ",
         paste(needed, collapse = ", "), ".", call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
# Validate face_selection: a rule, "all", face numbers, or a detect_faces()
# data frame. Returns list(mode, value).
.te_face_selection <- function(face_selection, allow_ids = TRUE) {
  rules <- c("largest", "left", "right", "none")
  if (is.data.frame(face_selection)) {
    .te_check_faces_table(face_selection)
    return(list(mode = "table", value = face_selection))
  }
  if (is.character(face_selection) && length(face_selection) == 1 && !is.na(face_selection)) {
    if (face_selection %in% rules) return(list(mode = "rule", value = face_selection))
    if (face_selection == "all") return(list(mode = "all", value = "all"))
  }
  if (allow_ids && is.numeric(face_selection) && length(face_selection) &&
      !anyNA(face_selection) && all(face_selection >= 1) &&
      all(face_selection == round(face_selection))) {
    return(list(mode = "ids", value = as.integer(face_selection)))
  }
  stop("Argument face_selection must be one of: largest, left, right, none, all",
       if (allow_ids) ", face numbers", ", or a data frame from detect_faces()",
       call. = FALSE)
}

#' @noRd
# Model id and architecture for a vision model alias, registered model or
# Hugging Face id (the same resolution image_scores() does)
.te_resolve_vision_model <- function(model, local_model_path = NULL) {
  if (!is.null(local_model_path) && !dir.exists(local_model_path)) {
    stop("The specified local_model_path directory does not exist: ", local_model_path)
  }
  if (is_vision_model_registered(model)) {
    config <- get_vision_model_config(model)
    message("Using registered model: ", config$description)
    return(list(model_id = config$model_id, architecture = config$architecture))
  }
  if (!is.null(local_model_path)) {
    message("Using local model from: ", local_model_path)
  } else {
    message("Using model directly from HuggingFace Hub: ", model)
  }
  list(model_id = model, architecture = NULL)
}

#' @noRd
# Frame image paths for the rows of a video faces table: the kept frames, or
# frames extracted again with the table's sampling settings. The returned
# vector carries a "cleanup" function that removes re-extracted frames.
.te_face_frame_paths <- function(faces, settings) {
  sources <- unique(faces$source)
  if (length(sources) != 1) {
    stop("A video faces table must come from one video.", call. = FALSE)
  }
  media <- attr(faces, "media")
  kept <- if (!is.null(media) && "frame_path" %in% names(media)) media else NULL
  if (!is.null(kept)) {
    paths <- kept$frame_path[match(faces$frame_index, kept$frame_index)]
    if (!anyNA(paths) && all(file.exists(paths))) {
      return(structure(paths, cleanup = function() invisible(NULL)))
    }
  }
  if (is.null(settings$nframes)) {
    stop("The faces table has no frame sampling settings; run detect_faces() on ",
         "the video again.", call. = FALSE)
  }
  dir <- tempfile("te_frames_")
  dir.create(dir)
  cleanup <- function() unlink(dir, recursive = TRUE)
  ok <- FALSE
  on.exit(if (!ok) cleanup(), add = TRUE)
  te_faces <- .te_faces_py()
  video <- sources
  if (.te_is_youtube_url(video)) {
    .te_require("youtube")
    video <- te_faces$download_youtube(video, dir, "video")
  }
  frames <- te_faces$extract_frames(video, settings$nframes, start = settings$start,
                                    end = settings$end, uniform = settings$uniform,
                                    ff = settings$ffreq, frame_dir = dir,
                                    video_name = "frame")
  index <- vapply(frames, function(f) as.integer(f$frame_index), integer(1))
  all_paths <- vapply(frames, function(f) f$path, character(1))
  paths <- all_paths[match(faces$frame_index, index)]
  if (anyNA(paths)) {
    stop("Could not extract the frames these faces were found in.", call. = FALSE)
  }
  ok <- TRUE
  structure(paths, cleanup = cleanup)
}
