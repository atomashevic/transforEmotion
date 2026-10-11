#' Draw detected faces
#'
#' Draws the box and a label for each face on its image or video frame, so you
#' can check what \code{\link{detect_faces}} found: which faces were
#' detected, how they are numbered, and which faces were recognised as the
#' same person. Faces of the same person share a colour. \code{plot()} on a
#' \code{\link{detect_faces}} result does the same.
#'
#' @param faces A data frame returned by \code{\link{detect_faces}} or
#'   \code{\link{select_faces}}, or a subset of its rows.
#' @param labels Text drawn above each box: one per row of \code{faces}, or
#'   one for all. By default the face number and person
#'   (\code{"face 1, p1"}), or for a video the track and person
#'   (\code{"track 1, p1"}). Use \code{""} for boxes only.
#' @param max_images Most images or video frames to draw. When there are
#'   more, they are picked evenly from the first to the last.
#' @param dir Directory for the drawn images (PNG files). Default is a
#'   temporary directory.
#' @param plot Logical. Show the drawn images in the plot window, or below
#'   the cell in a notebook. Default is \code{TRUE}.
#'
#' @details
#' Only images and frames with a face in \code{faces} are drawn. The
#' \code{"media"} attribute of a \code{\link{detect_faces}} result lists every
#' image or frame with its number of faces, including those without any.
#'
#' For a video, the frames are extracted again (as in
#' \code{\link{face_scores}}), unless \code{\link{detect_faces}} kept them
#' with \code{save_frames = TRUE}.
#'
#' @return The paths of the drawn images, invisibly.
#'
#' @seealso \code{\link{detect_faces}}, \code{\link{select_faces}}
#'
#' @examples
#' \dontrun{
#' image <- system.file("extdata", "two-people.jpg", package = "transforEmotion")
#' faces <- detect_faces(image)
#' draw_faces(faces)
#'
#' # A video: a few frames, with each face's track and person
#' video <- system.file("extdata", "two-people.mp4", package = "transforEmotion")
#' plot(detect_faces(video, nframes = 20), max_images = 4)
#' }
#'
#' @export
draw_faces <- function(faces, labels = NULL, max_images = 20, dir = tempdir(),
                       plot = TRUE) {
  .te_check_faces_table(faces)
  if (!is.null(labels) && !length(labels) %in% c(1, nrow(faces))) {
    stop("labels must have one entry per row of faces (or one for all).", call. = FALSE)
  }
  if (!nrow(faces)) {
    message("No faces to draw.")
    return(invisible(character()))
  }
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)

  settings <- attr(faces, "settings")
  is_video <- .te_is_video_faces(faces, settings)
  labels <- if (is.null(labels)) .te_face_labels(faces, is_video) else rep_len(labels, nrow(faces))
  colors <- .te_face_colors(faces)

  # One group of rows per image or video frame, at most max_images of them
  group <- if (is_video) faces$frame_index else faces$source
  groups <- unique(group)
  if (length(groups) > max_images) {
    groups <- groups[unique(round(seq(1, length(groups), length.out = max_images)))]
  }
  rows <- which(group %in% groups)

  ensure_te_py_env()
  te_faces <- .te_faces_py()
  images <- if (is_video) .te_face_frame_paths(faces[rows, , drop = FALSE], settings) else faces$source[rows]
  if (is_video) {
    cleanup <- attr(images, "cleanup")
    on.exit(cleanup(), add = TRUE)
  }
  images <- as.character(images)

  stem <- gsub("[^A-Za-z0-9._-]+", "_", tools::file_path_sans_ext(basename(faces$source[rows])))
  if (is_video) stem <- sprintf("%s-frame-%d", stem, faces$frame[rows])
  # Different images with the same file name get different output files
  key <- group[rows]
  stem <- make.unique(stem[match(groups, key)], sep = "-")[match(key, groups)]
  out_names <- paste0(stem, "-faces.png")

  paths <- character(length(groups))
  previews <- vector("list", length(groups))
  titles <- character(length(groups))
  for (g in seq_along(groups)) {
    r <- which(key == groups[g])
    i <- rows[r]
    paths[g] <- file.path(dir, out_names[r[1]])
    previews[[g]] <- te_faces$draw(
      images[r[1]],
      boxes = lapply(i, function(k) c(faces$x[k], faces$y[k], faces$width[k], faces$height[k])),
      labels = as.list(as.character(labels[i])),
      colors = unname(colors[i]),
      out_path = paths[g]
    )
    titles[g] <- if (is_video) {
      sprintf("frame %d (%.1f s)", faces$frame[i[1]], faces$time[i[1]])
    } else {
      basename(faces$source[i[1]])
    }
  }
  if (isTRUE(plot)) .te_plot_images(previews, titles)
  invisible(normalizePath(paths))
}

#' @rdname draw_faces
#' @param x A data frame returned by \code{\link{detect_faces}}.
#' @param ... Passed to \code{draw_faces()}.
#' @export
plot.te_faces <- function(x, ...) {
  draw_faces(x, ...)
}

#' @noRd
# Whether a faces table holds faces of video frames
.te_is_video_faces <- function(faces, settings = attr(faces, "settings")) {
  identical(settings$type, "video") ||
    ("frame_index" %in% names(faces) && any(!is.na(faces$frame_index)))
}

#' @noRd
# Default labels: face number (track for a video) and person
.te_face_labels <- function(faces, is_video) {
  n <- nrow(faces)
  base <- if (is_video && "track_id" %in% names(faces) && any(!is.na(faces$track_id))) {
    ifelse(is.na(faces$track_id), "", paste("track", faces$track_id))
  } else if ("face" %in% names(faces)) {
    ifelse(is.na(faces$face), "", paste("face", faces$face))
  } else {
    rep("", n)
  }
  person <- if ("person_id" %in% names(faces)) faces$person_id else rep(NA_character_, n)
  ifelse(is.na(person), base, ifelse(nzchar(base), paste0(base, ", ", person), person))
}

# Colour-blind safe colours (Okabe and Ito) that keep white text readable
.te_face_palette <- list(
  c(0, 114, 178), c(213, 94, 0), c(0, 158, 115), c(204, 121, 167),
  c(230, 159, 0), c(86, 180, 233), c(0, 0, 0), c(128, 128, 128)
)

#' @noRd
# One colour per person, or per track or face number without person ids
.te_face_colors <- function(faces) {
  n <- nrow(faces)
  pick <- function(column, prefix) {
    if (column %in% names(faces)) ifelse(is.na(faces[[column]]), NA, paste0(prefix, faces[[column]]))
    else rep(NA_character_, n)
  }
  key <- pick("person_id", "")
  for (alt in list(pick("track_id", "t"), pick("face", "f"))) {
    key <- ifelse(is.na(key), alt, key)
  }
  key[is.na(key)] <- "?"
  index <- (match(key, unique(key)) - 1) %% length(.te_face_palette) + 1
  .te_face_palette[index]
}

#' @noRd
# Plot RGB arrays (0-255) with titles, up to six to a page
.te_plot_images <- function(images, titles) {
  op <- graphics::par(mfrow = .te_image_grid(images), mar = c(0.2, 0.2, 1.4, 0.2))
  on.exit(graphics::par(op))
  for (k in seq_along(images)) {
    img <- images[[k]]
    h <- dim(img)[1]
    w <- dim(img)[2]
    graphics::plot.new()
    graphics::plot.window(xlim = c(0, w), ylim = c(0, h), asp = 1)
    graphics::rasterImage(grDevices::as.raster(img / 255), 0, 0, w, h)
    graphics::title(main = titles[k], cex.main = 0.8, font.main = 1)
  }
  invisible(NULL)
}

#' @noRd
# Rows and columns that show up to six images largest on the current device
.te_image_grid <- function(images) {
  n <- min(length(images), 6)
  ratio <- stats::median(vapply(images, function(img) dim(img)[2] / dim(img)[1], numeric(1)))
  device <- grDevices::dev.size("in")
  width <- vapply(seq_len(n), function(cols) {
    min(device[1] / cols, device[2] / ceiling(n / cols) * ratio)
  }, numeric(1))
  cols <- which.max(width)
  c(ceiling(n / cols), cols)
}
