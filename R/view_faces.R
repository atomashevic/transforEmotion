#' Look at detected faces in the browser, and choose faces to analyse
#'
#' Writes a small web page that shows every image, or every sampled video
#' frame, with its face boxes, and opens it in your web browser. Faces of the
#' same person share a colour, and a People panel lists each person with the
#' images, frames and tracks they appear in. With \code{select = TRUE}, you
#' can click faces, video tracks or people to choose the faces to analyse,
#' export the choice, and read it back with \code{read_face_selection()}.
#'
#' The page is a folder with \code{index.html}, a few script files and small
#' copies of the images or frames. It needs no server and no internet
#' connection, and it makes no outside requests, so it also works after you
#' copy the folder from a remote computer (for example a cluster) to your
#' own.
#'
#' @param faces A data frame returned by \code{\link{detect_faces}} or
#'   \code{\link{select_faces}}.
#' @param select Logical. Let the viewer choose faces and export the choice.
#'   Default is \code{FALSE}.
#' @param selected Faces to start with selected, as rows of \code{faces} (for
#'   example \code{select_faces(faces, "largest")}). Only with
#'   \code{select = TRUE}.
#' @param dir A new or empty folder for the page. Default is a new folder in
#'   the temporary directory, which R deletes when the session ends.
#' @param open Logical. Open the page in the web browser. Default is
#'   \code{TRUE} in an interactive session.
#' @param title Title shown at the top of the page.
#' @param max_side Longest side, in pixels, of the image and frame copies in
#'   the page. Boxes are drawn in the original pixels, whatever the size of
#'   the copies.
#'
#' @details
#' Images without a face are shown too when \code{faces} is a complete
#' \code{\link{detect_faces}} result; for a subset, only images with a face
#' in it are shown. For a video, every sampled frame is shown, and the frames
#' are extracted again (as in \code{\link{face_scores}}), unless
#' \code{\link{detect_faces}} kept them with \code{save_frames = TRUE}.
#'
#' In select mode, clicking a face in a still image selects that face;
#' clicking a face in a video selects its whole track; clicking a person
#' selects every face of that person. \strong{Export selection} downloads
#' \code{selection-<task>.json}, and shows the same text in the page to copy
#' if downloads are blocked.
#'
#' The page lists file names and shows the images, so share the folder only
#' with people who may see them. It contains no face embeddings and no
#' scores.
#'
#' @return The path of the page's \code{index.html}, invisibly.
#'
#' @seealso \code{\link{draw_faces}} for the same boxes as plots in R,
#'   \code{\link{select_faces}} to choose faces by rule, number or person.
#'
#' @examples
#' \dontrun{
#' images <- system.file("extdata", c("two-people.jpg", "boris-2.png", "trump2.jpg"),
#'                       package = "transforEmotion")
#' faces <- detect_faces(images)
#' view_faces(faces)
#'
#' # Choose faces in the browser, export, then read the downloaded file
#' view_faces(faces, select = TRUE)
#' chosen <- read_face_selection("~/Downloads/selection-faces-20261011-101500.json", faces)
#' face_scores(chosen, classes = c("happy", "sad", "angry", "neutral"))
#' }
#'
#' @export
view_faces <- function(faces, select = FALSE, selected = NULL, dir = NULL,
                       open = interactive(), title = NULL, max_side = 1280) {
  .te_check_faces_table(faces)
  if (!nrow(faces)) stop("faces has no rows to show.", call. = FALSE)
  if (!is.null(selected)) {
    if (!isTRUE(select)) stop("selected needs select = TRUE.", call. = FALSE)
    .te_check_faces_table(selected)
  }
  if (is.null(dir)) dir <- tempfile("te-faces-")
  if (dir.exists(dir) && length(list.files(dir, all.files = TRUE, no.. = TRUE))) {
    stop("dir must be a new or empty folder: ", dir, call. = FALSE)
  }
  dir.create(file.path(dir, "media"), recursive = TRUE, showWarnings = FALSE)

  settings <- attr(faces, "settings")
  is_video <- .te_is_video_faces(faces, settings)
  media_ids <- .te_media_ids(faces)
  shown <- .te_shown_media(faces, is_video)

  ensure_te_py_env()
  te_faces <- .te_faces_py()
  if (is_video) {
    frame_table <- data.frame(source = shown$source, frame_index = shown$frame_index,
                              stringsAsFactors = FALSE)
    attr(frame_table, "media") <- attr(faces, "media")
    images <- .te_face_frame_paths(frame_table, settings)
    cleanup <- attr(images, "cleanup")
    on.exit(cleanup(), add = TRUE)
    shown$file <- sprintf("media/%s-f%06d.jpg", media_ids$media_id[1], shown$frame_index)
  } else {
    images <- shown$source
    shown$file <- paste0("media/", shown$media_id, ".jpg")
  }
  sizes <- te_faces$display_copies(as.list(as.character(images)),
                                   as.list(file.path(dir, shown$file)),
                                   max_side = as.integer(max_side))
  shown$img_w <- vapply(sizes, function(s) as.integer(s[[1]]), integer(1))
  shown$img_h <- vapply(sizes, function(s) as.integer(s[[2]]), integer(1))

  mode <- if (isTRUE(select)) "select" else "view"
  task <- .te_face_task(faces, shown, is_video, mode, title,
                        selected_ids = if (is.null(selected)) character() else .te_face_ids(selected))
  json <- jsonlite::toJSON(task, auto_unbox = TRUE, null = "null", na = "null", digits = 4)
  writeLines(paste0("window.TE_TASK = ", json, ";"), file.path(dir, "data.js"), useBytes = TRUE)
  app <- system.file("www", "face-annotator", .te_face_app_files, package = "transforEmotion",
                     mustWork = TRUE)
  file.copy(app, dir, overwrite = TRUE)

  index <- normalizePath(file.path(dir, "index.html"))
  if (isTRUE(open)) utils::browseURL(index)
  if (mode == "select") {
    message("Select faces in the browser and click Export selection. Then read the ",
            "downloaded file with read_face_selection(\"<path>/selection-", task$task_id,
            ".json\", faces).")
  }
  invisible(index)
}

#' Read faces chosen in the browser
#'
#' Reads the file that \strong{Export selection} in
#' \code{\link{view_faces}}\code{(select = TRUE)} downloads, and returns the
#' chosen faces.
#'
#' @param file Path of the exported \code{selection-<task>.json} file, or its
#'   text (as copied from the page).
#' @param faces The same data frame that was passed to
#'   \code{\link{view_faces}}, or the full \code{\link{detect_faces}} result
#'   it was taken from.
#'
#' @return The chosen rows of \code{faces}, which \code{\link{face_scores}}
#'   scores and \code{face_selection} in \code{\link{image_scores}},
#'   \code{\link{image_scores_dir}} and \code{\link{video_scores}} accepts.
#'
#' @seealso \code{\link{view_faces}}
#'
#' @examples
#' \dontrun{
#' faces <- detect_faces(system.file("extdata", "two-people.jpg", package = "transforEmotion"))
#' view_faces(faces, select = TRUE)
#' chosen <- read_face_selection("~/Downloads/selection-faces-20261011-101500.json", faces)
#' }
#'
#' @export
read_face_selection <- function(file, faces) {
  .te_check_faces_table(faces)
  if (!is.character(file) || length(file) != 1 || is.na(file)) {
    stop("file must be one path or the exported text.", call. = FALSE)
  }
  is_text <- grepl("^\\s*\\{", file)
  if (!is_text && !file.exists(file)) stop("File does not exist: ", file, call. = FALSE)
  sel <- tryCatch(jsonlite::fromJSON(file, simplifyVector = TRUE),
                  error = function(e) stop("Cannot read the selection: ", conditionMessage(e),
                                           call. = FALSE))
  if (!identical(sel$schema, "transforEmotion.face_selection")) {
    stop("This is not a face selection exported by view_faces().", call. = FALSE)
  }
  chosen <- as.character(unlist(sel$selected_face_ids))
  if (is.data.frame(sel$media) && all(c("media_id", "label") %in% names(sel$media))) {
    media <- .te_media_ids(faces)
    both <- match(sel$media$media_id, media$media_id)
    if (any(!is.na(both) & basename(media$source[both]) != sel$media$label)) {
      stop("The selection was made on other images or another video than faces.", call. = FALSE)
    }
  }
  ids <- .te_face_ids(faces)
  unknown <- setdiff(chosen, ids)
  if (length(unknown)) {
    warning(length(unknown), " selected face(s) are not in faces (",
            paste(utils::head(unknown, 5), collapse = ", "),
            if (length(unknown) > 5) ", ...", "); was the selection made on another table?",
            call. = FALSE)
  }
  faces[ids %in% chosen, , drop = FALSE]
}

# The page's files in inst/www/face-annotator, besides data.js and media/
.te_face_app_files <- c("index.html", "style.css", "logic.js", "app.js")

#' @noRd
# Opaque, stable id of each face's image or video: "m01", "m02", ... in the
# order of the "media" attribute, so subsets of a table keep their ids
.te_media_ids <- function(faces) {
  media <- attr(faces, "media")
  sources <- unique(if (!is.null(media)) media$source else faces$source)
  sources <- c(sources, setdiff(unique(faces$source), sources))
  ids <- sprintf("m%0*d", max(2, nchar(length(sources))), seq_along(sources))
  data.frame(source = sources, media_id = ids, stringsAsFactors = FALSE)
}

#' @noRd
# face_id: "<media_id>#<frame_index>#<face>", frame 0 for still images
.te_face_ids <- function(faces) {
  media <- .te_media_ids(faces)
  media_id <- media$media_id[match(faces$source, media$source)]
  frame_index <- if ("frame_index" %in% names(faces)) faces$frame_index else rep(NA_integer_, nrow(faces))
  frame_index[is.na(frame_index)] <- 0L
  face <- if ("face" %in% names(faces)) {
    faces$face
  } else {
    stats::ave(seq_len(nrow(faces)), media_id, frame_index, FUN = seq_along)
  }
  ids <- sprintf("%s#%d#%d", media_id, as.integer(frame_index), as.integer(face))
  if (anyDuplicated(ids)) {
    stop("faces has two rows for the same face (source, frame_index and face).", call. = FALSE)
  }
  ids
}

#' @noRd
# The images, or video frames, the page shows: one row each, with media_id
.te_shown_media <- function(faces, is_video) {
  media <- attr(faces, "media")
  ids <- .te_media_ids(faces)
  if (is_video) {
    if (length(unique(faces$source)) != 1) {
      stop("A video faces table must come from one video.", call. = FALSE)
    }
    frames <- if (!is.null(media)) media[media$source == faces$source[1], , drop = FALSE] else NULL
    if (is.null(frames) || !nrow(frames)) {
      frames <- unique(as.data.frame(faces)[, c("source", "frame", "frame_index", "time")])
    }
    frames <- frames[order(frames$frame_index), c("source", "frame", "frame_index", "time")]
    frames$media_id <- ids$media_id[match(frames$source, ids$source)]
    return(frames)
  }
  # Images without faces only when the table is complete
  complete <- !is.null(media) && "n_faces" %in% names(media) &&
    nrow(faces) == sum(media$n_faces, na.rm = TRUE)
  sources <- if (complete) unique(media$source) else unique(faces$source)
  data.frame(source = sources, media_id = ids$media_id[match(sources, ids$source)],
             stringsAsFactors = FALSE)
}

#' @noRd
# The task view_faces() writes to data.js (window.TE_TASK)
.te_face_task <- function(faces, shown, is_video, mode, title, selected_ids = character()) {
  face_ids <- .te_face_ids(faces)
  column <- function(name, fill = NA) if (name %in% names(faces)) faces[[name]] else rep(fill, nrow(faces))
  media_id <- shown$media_id[match(faces$source, shown$source)]
  if (is_video) {
    size <- shown[1, c("img_w", "img_h")]
    media <- list(list(
      media_id = shown$media_id[1], media_type = "video", label = basename(shown$source[1]),
      img_w = size$img_w, img_h = size$img_h,
      frames = data.frame(frame = shown$frame, frame_index = shown$frame_index,
                          timestamp_s = shown$time, src = shown$file,
                          stringsAsFactors = FALSE)
    ))
    img_w <- rep(size$img_w, nrow(faces))
    img_h <- rep(size$img_h, nrow(faces))
  } else {
    media <- lapply(seq_len(nrow(shown)), function(i) list(
      media_id = shown$media_id[i], media_type = "image", label = basename(shown$source[i]),
      src = shown$file[i], img_w = shown$img_w[i], img_h = shown$img_h[i]
    ))
    img_w <- shown$img_w[match(media_id, shown$media_id)]
    img_h <- shown$img_h[match(media_id, shown$media_id)]
  }
  person <- as.character(column("person_id"))
  person[!is.na(person) & !nzchar(person)] <- NA_character_
  face_rows <- data.frame(
    face_id = face_ids, media_id = media_id,
    media_type = if (is_video) "video" else "image",
    frame = as.integer(column("frame")), frame_index = as.integer(column("frame_index")),
    timestamp_s = as.numeric(column("time")),
    img_w = img_w, img_h = img_h,
    x = faces$x, y = faces$y, w = faces$width, h = faces$height,
    confidence = as.numeric(column("confidence")), detector = as.character(column("detector")),
    track_id = as.integer(column("track_id")), person_id = person,
    selected = face_ids %in% selected_ids,
    stringsAsFactors = FALSE
  )
  instructions <- if (mode == "select") {
    paste("Click faces, video tracks or people to select them, then Export selection",
          "and read the file in R with read_face_selection().")
  } else {
    "Faces of the same person share a colour. Click an image or frame to see it large."
  }
  list(
    schema = "transforEmotion.face_task", schema_version = "1.0.0",
    task_id = format(Sys.time(), "faces-%Y%m%d-%H%M%S"), mode = mode,
    title = if (is.null(title)) "Faces" else as.character(title),
    instructions = instructions,
    media = media, faces = face_rows
  )
}
