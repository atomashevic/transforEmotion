#' Look at detected faces in the browser, and choose faces to analyse
#'
#' Starts a small web page from R that shows every image, or every sampled
#' video frame, with its face boxes, and opens it in your web browser. Faces
#' of the same person share a colour, and a People panel lists each person
#' with the images, frames and tracks they appear in. With
#' \code{select = TRUE}, you click faces, video tracks or people, then
#' \strong{Send to R}, and \code{view_faces()} returns the chosen faces.
#'
#' The page is served by R itself (with \pkg{httpuv}) on your own computer
#' only (\code{127.0.0.1}), at an address with a random part that only you
#' get. It makes no outside requests. While R waits for a selection, the
#' console is busy; press Esc (Ctrl+C in a terminal) to stop waiting.
#'
#' With \code{live = FALSE}, no server is started: the page is written as a
#' folder that also works when opened from disk, for example after copying it
#' from a remote computer to yours. Its \strong{Export selection} button then
#' downloads a file that \code{\link{read_face_selection}()} reads.
#'
#' In a Jupyter notebook, including Google Colab, the page is shown inside the
#' notebook, below the cell (\code{inline = TRUE}, the default there). A page
#' served from R cannot be reached from a notebook running on another
#' computer, so in select mode the page gives a line of R code instead:
#' click \strong{Copy R code}, paste the line into a new cell and run it, and
#' \code{\link{read_face_selection}()} returns the chosen faces.
#'
#' @param faces A data frame returned by \code{\link{detect_faces}} or
#'   \code{\link{select_faces}}.
#' @param select Logical. Let the viewer choose faces and return them.
#'   Default is \code{FALSE}.
#' @param selected Faces to start with selected, as rows of \code{faces} (for
#'   example \code{select_faces(faces, "largest")}). Only with
#'   \code{select = TRUE}.
#' @param live Logical. Serve the page from R (the default). \code{FALSE}
#'   only writes the folder.
#' @param open Logical. Open the page in the web browser. Default is
#'   \code{TRUE} in an interactive session.
#' @param dir A new or empty folder for the page. Default is a new folder in
#'   the temporary directory, which R deletes when the session ends.
#' @param title Title shown at the top of the page.
#' @param max_side Longest side, in pixels, of the image and frame copies in
#'   the page. Default is 1280, or 640 inside a notebook, where the copies are
#'   stored in the notebook itself. Boxes are drawn in the original pixels,
#'   whatever the size of the copies.
#' @param port Port of the local server. Default is a free port chosen at
#'   random; set one to forward it over SSH when R runs on a remote computer.
#' @param inline Logical. Show the page inside the notebook, below the cell,
#'   instead of in a browser tab. Default is \code{TRUE} in a Jupyter
#'   notebook (Google Colab, JupyterLab, VS Code) and \code{FALSE} elsewhere.
#'   \code{live} and \code{open} are then ignored.
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
#' selects every face of that person.
#'
#' When R runs on a remote computer over SSH, forward the port first (the
#' message printed by \code{view_faces()} gives the command), or use
#' \code{live = FALSE} and copy the folder.
#'
#' The page lists file names and shows the images, so share it only with
#' people who may see them. It contains no face embeddings and no scores.
#'
#' @return With \code{select = TRUE}: the chosen rows of \code{faces}, which
#'   \code{\link{face_scores}} scores and \code{face_selection} in
#'   \code{\link{image_scores}}, \code{\link{image_scores_dir}} and
#'   \code{\link{video_scores}} accepts, or \code{NULL} if you click
#'   \strong{Cancel}. Otherwise, invisibly, the page: a list with its
#'   \code{url}, its folder \code{dir}, and \code{stop()}, which stops the
#'   server. With \code{live = FALSE}, the path of the page's
#'   \code{index.html}, invisibly. With \code{inline = TRUE}, invisibly, the
#'   path of the page as one self-contained file, \code{faces.html} in
#'   \code{dir}; in select mode the chosen faces come from the copied line of
#'   code.
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
#' # Click the faces to keep, then Send to R
#' chosen <- view_faces(faces, select = TRUE)
#' face_scores(chosen, classes = c("happy", "sad", "angry", "neutral"))
#'
#' # Without a server: a folder to copy, and a file to read back
#' view_faces(faces, select = TRUE, live = FALSE, dir = "faces-page")
#' chosen <- read_face_selection("~/Downloads/selection-faces-20261011-101500.json", faces)
#'
#' # In a notebook (Colab): the page appears below the cell. Click the faces,
#' # then Copy R code, and run the copied line in a new cell, for example
#' view_faces(faces, select = TRUE)
#' chosen <- read_face_selection("te-selection:eyJzY2hlbWEiOi...", faces)
#' }
#'
#' @export
view_faces <- function(faces, select = FALSE, selected = NULL, live = TRUE,
                       open = interactive(), dir = NULL, title = NULL, max_side = NULL,
                       port = NULL, inline = .te_in_notebook()) {
  # The name the copied line of code uses for the faces table
  faces_name <- paste(deparse(substitute(faces)), collapse = " ")
  if (!identical(make.names(faces_name), faces_name)) faces_name <- "faces"
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
  mode <- if (isTRUE(select)) "select" else "view"

  if (isTRUE(inline)) {
    task <- .te_write_face_page(faces, dir, mode, title, if (is.null(max_side)) 640 else max_side,
                                selected, notebook = faces_name)
    page <- file.path(dir, "faces.html")
    writeLines(.te_single_file_page(dir), page, useBytes = TRUE)
    .te_show_in_notebook(page, task$task_id)
    if (mode == "select") {
      message("Click faces, video tracks or people below, then Copy R code. Paste the line ",
              "into a new cell and run it: read_face_selection() returns the chosen faces.")
    }
    return(invisible(normalizePath(page)))
  }

  if (is.null(max_side)) max_side <- 1280
  api <- if (isTRUE(live) && mode == "select") c("selection", "cancel") else character()
  task <- .te_write_face_page(faces, dir, mode, title, max_side, selected, api)

  if (!isTRUE(live)) {
    index <- normalizePath(file.path(dir, "index.html"))
    if (isTRUE(open)) utils::browseURL(index)
    if (mode == "select") {
      message("Select faces in the browser and click Export selection. Then read the ",
              "downloaded file with read_face_selection(\"<path>/selection-", task$task_id,
              ".json\", faces).")
    }
    return(invisible(index))
  }

  if (mode == "view") {
    server <- .te_serve_page(dir, port = port)
    .te_open_page(server, open, paste(
      "The faces are at %s. The page stays up until R ends;",
      "to stop it sooner, keep what view_faces() returns and call its $stop()."
    ))
    return(invisible(structure(list(url = server$url, dir = normalizePath(dir), stop = server$stop),
                               class = "te_face_page")))
  }

  state <- new.env(parent = emptyenv())
  server <- .te_serve_page(dir, port = port, handlers = list(
    selection = function(body) {
      if (!identical(body$task_id, task$task_id)) stop("This page belongs to another view_faces() call.")
      state$result <- .te_selection_rows(body, faces)
      list(ok = TRUE, n_faces = nrow(state$result))
    },
    cancel = function(body) {
      state$cancelled <- TRUE
      list(ok = TRUE)
    }
  ))
  on.exit(server$stop(), add = TRUE)
  .te_open_page(server, open, paste(
    "Choose faces at %s and click Send to R.",
    "R waits until then; press Esc (Ctrl+C in a terminal) to stop waiting."
  ))
  .te_wait_for(function() !is.null(state$result) || isTRUE(state$cancelled))
  if (is.null(state$result)) {
    message("Selection cancelled.")
    return(invisible(NULL))
  }
  message(nrow(state$result), " face(s) selected.")
  state$result
}

#' @export
print.te_face_page <- function(x, ...) {
  cat("Faces page at ", x$url, "\nFolder: ", x$dir, "\nStop it with $stop().\n", sep = "")
  invisible(x)
}

#' @noRd
# Writes the page into dir: the app's files, data.js and the display copies.
# api names the server's handlers the page may call (data.js sets
# window.TE_API); none for a page opened from disk. notebook, the R name of
# the faces table, writes a page for a notebook: the images go into data.js,
# and window.TE_NOTEBOOK tells the page to give a line of code. Returns the task.
.te_write_face_page <- function(faces, dir, mode, title, max_side, selected = NULL,
                                api = character(), notebook = NULL) {
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
  # A notebook shows the page from its own output, with no folder beside it
  if (!is.null(notebook)) shown$file <- .te_data_uri(file.path(dir, shown$file))

  task <- .te_face_task(faces, shown, is_video, mode, title,
                        selected_ids = if (is.null(selected)) character() else .te_face_ids(selected),
                        live = length(api) > 0, notebook = !is.null(notebook))
  json <- jsonlite::toJSON(task, auto_unbox = TRUE, null = "null", na = "null", digits = 4)
  js <- paste0("window.TE_TASK = ", json, ";")
  if (length(api)) {
    # Relative to the page, so the server's random prefix is not repeated here
    endpoints <- stats::setNames(as.list(paste0("api/", api)), api)
    js <- c(js, paste0("window.TE_API = ", jsonlite::toJSON(endpoints, auto_unbox = TRUE), ";"))
  }
  if (!is.null(notebook)) {
    js <- c(js, paste0("window.TE_NOTEBOOK = ",
                       jsonlite::toJSON(list(faces = notebook), auto_unbox = TRUE), ";"))
  }
  writeLines(js, file.path(dir, "data.js"), useBytes = TRUE)
  app <- system.file("www", "face-annotator", .te_face_app_files, package = "transforEmotion",
                     mustWork = TRUE)
  file.copy(app, dir, overwrite = TRUE)
  task
}

#' Read faces chosen in the browser
#'
#' Reads the file that \strong{Export selection} in
#' \code{\link{view_faces}}\code{(select = TRUE)} downloads, or the code in
#' the line that \strong{Copy R code} gives in a notebook, and returns the
#' chosen faces.
#'
#' @param file Path of the exported \code{selection-<task>.json} file, its
#'   text (as copied from the page), or the \code{"te-selection:..."} code in
#'   the line that \strong{Copy R code} gives in a notebook.
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
  # The code may come back wrapped over several lines
  code <- regmatches(file, regexec("^[[:space:]]*te-selection:([A-Za-z0-9+/=[:space:]]+)$", file))[[1]]
  if (length(code)) {
    file <- tryCatch(rawToChar(jsonlite::base64_dec(gsub("[[:space:]]", "", code[2]))),
                     error = function(e) stop("The selection code is cut off or changed; copy it again.",
                                              call. = FALSE))
    Encoding(file) <- "UTF-8"
  }
  is_text <- grepl("^\\s*\\{", file)
  if (!is_text && !file.exists(file)) stop("File does not exist: ", file, call. = FALSE)
  sel <- tryCatch(jsonlite::fromJSON(file, simplifyVector = TRUE),
                  error = function(e) stop("Cannot read the selection: ", conditionMessage(e),
                                           call. = FALSE))
  .te_selection_rows(sel, faces)
}

#' @noRd
# The rows of faces a selection (parsed JSON, from a file or the page) names
.te_selection_rows <- function(sel, faces) {
  if (!is.list(sel) || !identical(sel$schema, "transforEmotion.face_selection")) {
    stop("This is not a face selection exported by view_faces().", call. = FALSE)
  }
  chosen <- as.character(unlist(sel$selected_face_ids))
  if (is.data.frame(sel$media) && all(c("media_id", "label") %in% names(sel$media))) {
    media <- .te_media_ids(faces)
    both <- match(sel$media$media_id, media$media_id)
    other <- !is.na(both) & basename(media$source[both]) != sel$media$label
    # Files with the same names (another participant's folder, say) differ in
    # content; files that cannot be read now are checked by name only
    if ("key" %in% names(sel$media)) {
      saved <- as.character(sel$media$key)
      now <- rep(NA_character_, length(both))
      now[!is.na(both)] <- .te_media_key(media$source[both[!is.na(both)]])
      other <- other | (!is.na(saved) & !is.na(now) & saved != now)
    }
    if (any(other)) {
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
# MD5 of each image or video file, so a selection is read with the same files
# and not with others of the same name; NA for URLs and missing files
.te_media_key <- function(sources) {
  keys <- rep(NA_character_, length(sources))
  local <- file.exists(sources) & !dir.exists(sources)
  keys[local] <- unname(tools::md5sum(sources[local]))
  keys
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
.te_face_task <- function(faces, shown, is_video, mode, title, selected_ids = character(),
                          live = FALSE, notebook = FALSE) {
  face_ids <- .te_face_ids(faces)
  column <- function(name, fill = NA) if (name %in% names(faces)) faces[[name]] else rep(fill, nrow(faces))
  media_id <- shown$media_id[match(faces$source, shown$source)]
  if (is_video) {
    size <- shown[1, c("img_w", "img_h")]
    media <- list(list(
      media_id = shown$media_id[1], media_type = "video", label = basename(shown$source[1]),
      key = .te_media_key(shown$source[1]),
      img_w = size$img_w, img_h = size$img_h,
      frames = data.frame(frame = shown$frame, frame_index = shown$frame_index,
                          timestamp_s = shown$time, src = shown$file,
                          stringsAsFactors = FALSE)
    ))
    img_w <- rep(size$img_w, nrow(faces))
    img_h <- rep(size$img_h, nrow(faces))
  } else {
    keys <- .te_media_key(shown$source)
    media <- lapply(seq_len(nrow(shown)), function(i) list(
      media_id = shown$media_id[i], media_type = "image", label = basename(shown$source[i]),
      key = keys[i],
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
  instructions <- if (mode == "select" && live) {
    "Click faces, video tracks or people to select them, then Send to R."
  } else if (mode == "select" && notebook) {
    paste("Click faces, video tracks or people to select them, then Copy R code,",
          "and run the copied line in a new cell.")
  } else if (mode == "select") {
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
