#' Have people label emotions in images, video clips or texts
#'
#' Opens a page that shows one item at a time (an image, frames of a video
#' clip, a text, or a clip with its text) and asks the person coding it a few
#' questions, such as which emotion they see. Use it to collect human labels
#' for a sample, or to review a model's errors: with the reference
#' \code{label} and the model's \code{prediction} in \code{items}, the page
#' shows them after the coder has given their own answer, and asks what
#' explains a disagreement. The answers come back as a data frame that
#' \code{\link{evaluate_emotions}} compares with the label or the model.
#'
#' The page works like \code{\link{view_faces}}. By default R serves it on
#' your own computer (\code{127.0.0.1}) and sends every answer to R as the
#' coder goes, saving them to \code{save}; \strong{Finish} returns them from
#' \code{annotate_emotions()}. In a Jupyter notebook, such as Google Colab,
#' the page is shown below the cell, and \strong{Copy R code} gives a line to
#' run in a new cell. With \code{live = FALSE} the page is a folder to open
#' or send to a coder, and \strong{Export answers} downloads a file. Both are
#' read with \code{read_annotations()}.
#'
#' @param items A data frame with one row per item to code, and these
#'   columns (all but one of \code{media} and \code{text} optional):
#'   \describe{
#'     \item{\code{id}}{A unique id per item. Default: the row number.}
#'     \item{\code{media}}{Path of an image or video file.}
#'     \item{\code{text}}{Text shown with the item, or coded on its own.}
#'     \item{\code{label}}{The reference label, such as the dataset's label.}
#'     \item{\code{prediction}}{The model's top class.}
#'     \item{one column per class}{The model's scores, shown as bars.}
#'   }
#' @param classes The emotion classes the coder chooses from.
#' @param questions The questions to ask, as returned by
#'   \code{annotation_questions()}.
#' @param coder The coder's name or id, stored with the answers. The coder
#'   can also type it on the page.
#' @param blind Logical. Show \code{label}, \code{prediction} and the scores
#'   only after the coder has answered the questions asked before them.
#'   Default \code{TRUE}, so the coder's own label is not swayed by them.
#' @param frames Number of frames shown per video clip, spread evenly over the
#'   clip as \code{video_scores(uniform = TRUE)} samples them. Default 4.
#' @param faces Logical. Find the faces in the images and frames and draw
#'   their boxes; the largest face, the one \code{\link{video_scores}} and
#'   \code{\link{image_scores}} score by default, has a solid box. Default
#'   \code{TRUE}.
#' @param play Logical. Also put the video clips in the page, with their
#'   sound, when the browser can play them (MP4, WebM, MOV). Default
#'   \code{TRUE}. Not in a notebook, where the clips would have to be stored
#'   in the notebook itself.
#' @param live Logical. Serve the page from R and get the answers directly
#'   (the default). \code{FALSE} only writes the folder.
#' @param inline Logical. Show the page inside the notebook, below the cell.
#'   Default \code{TRUE} in a Jupyter notebook and \code{FALSE} elsewhere.
#' @param save Path of a JSON file that keeps the answers sent to R, after
#'   every answer, so none are lost if R stops waiting. Default
#'   \code{answers.json} in \code{dir}. Read it with \code{read_annotations()}.
#' @param open Logical. Open the page in the web browser. Default is
#'   \code{TRUE} in an interactive session.
#' @param dir A new or empty folder for the page. Default is a new folder in
#'   the temporary directory.
#' @param title Title shown at the top of the page.
#' @param max_side Longest side, in pixels, of the image and frame copies.
#'   Default 960, or 480 in a notebook.
#' @param port Port of the local server. Default is a free port chosen at
#'   random.
#'
#' @return With \code{live = TRUE}: the answers, as returned by
#'   \code{read_annotations()}, after \strong{Finish}, or \code{NULL} after
#'   \strong{Cancel}. Otherwise, invisibly, the path of the page
#'   (\code{index.html}, or \code{annotate.html} in a notebook).
#'
#' @details
#' Text-only items need nothing but R. Images and video clips need the
#' package's Python environment, as \code{\link{detect_faces}} does.
#'
#' The page shows the images, the texts and the file names, so share it only
#' with people who may see them.
#'
#' @seealso \code{\link{evaluate_emotions}} to compare the answers with the
#'   label or the model, \code{\link{view_faces}} to look at faces.
#'
#' @examples
#' \dontrun{
#' emotions <- c("angry", "fearful", "happy", "sad")
#' # The held-out clips a model got wrong, with their descriptions
#' items <- data.frame(id = errors$clip, media = errors$path, text = errors$description,
#'                     label = errors$label, prediction = errors$predicted,
#'                     errors[, emotions])
#' answers <- annotate_emotions(items, emotions, coder = "AT")
#' table(answers$why)
#'
#' # The coder against the dataset's label, and the model against the coder
#' evaluate_emotions(answers, truth_col = "label", pred_col = "emotion")
#' evaluate_emotions(answers, truth_col = "emotion", pred_col = "prediction")
#'
#' # In Colab: the page appears below the cell; Copy R code gives a line like
#' answers <- read_annotations("te-annotations:eyJzY2hlbWEiOi...", items)
#' }
#'
#' @export
annotate_emotions <- function(items, classes, questions = annotation_questions(classes, items),
                              coder = NULL, blind = TRUE, frames = 4, faces = TRUE,
                              play = TRUE, live = TRUE, inline = .te_in_notebook(),
                              save = NULL, open = interactive(), dir = NULL, title = NULL,
                              max_side = NULL, port = NULL) {
  items_name <- paste(deparse(substitute(items)), collapse = " ")
  if (!identical(make.names(items_name), items_name)) items_name <- "items"
  items <- .te_check_items(items, classes)
  questions <- .te_check_questions(questions)
  if (is.null(dir)) dir <- tempfile("te-annotate-")
  if (dir.exists(dir) && length(list.files(dir, all.files = TRUE, no.. = TRUE))) {
    stop("dir must be a new or empty folder: ", dir, call. = FALSE)
  }
  dir.create(file.path(dir, "media"), recursive = TRUE, showWarnings = FALSE)
  if (is.null(max_side)) max_side <- if (isTRUE(inline)) 480 else 960
  # A notebook page is one file: the clips would have to be stored in it
  if (isTRUE(inline)) play <- FALSE
  serve <- isTRUE(live) && !isTRUE(inline)

  task <- .te_annotation_task(items, classes, questions, coder, blind, title,
                              how = if (isTRUE(inline)) "notebook" else if (serve) "live" else "file")
  task$items <- .te_item_media(items, dir, score_cols = intersect(classes, names(items)),
                               frames = frames, faces = faces,
                               play = isTRUE(play), max_side = max_side,
                               data_uri = isTRUE(inline))
  js <- paste0("window.TE_TASK = ", jsonlite::toJSON(task, auto_unbox = TRUE, null = "null",
                                                     na = "null", digits = 4), ";")
  if (serve) {
    js <- c(js, paste0("window.TE_API = ", jsonlite::toJSON(
      list(answers = "api/answers", finish = "api/finish", cancel = "api/cancel"), auto_unbox = TRUE), ";"))
  }
  if (isTRUE(inline)) {
    js <- c(js, paste0("window.TE_NOTEBOOK = ",
                       jsonlite::toJSON(list(items = items_name), auto_unbox = TRUE), ";"))
  }
  writeLines(js, file.path(dir, "data.js"), useBytes = TRUE)
  app <- system.file("www", "emotion-annotator", .te_annotator_files, package = "transforEmotion",
                     mustWork = TRUE)
  file.copy(app, dir, overwrite = TRUE)

  if (isTRUE(inline)) {
    page <- file.path(dir, "annotate.html")
    writeLines(.te_single_file_page(dir), page, useBytes = TRUE)
    .te_show_in_notebook(page, task$task_id)
    message("Code the items below, then Copy R code. Paste the line into a new cell and run it: ",
            "read_annotations() returns the answers.")
    return(invisible(normalizePath(page)))
  }
  if (!serve) {
    index <- normalizePath(file.path(dir, "index.html"))
    if (isTRUE(open)) utils::browseURL(index)
    message("Code the items in the browser and click Export answers. Then read the downloaded ",
            "file with read_annotations(\"<path>/answers-", task$task_id, ".json\", items).")
    return(invisible(index))
  }

  if (is.null(save)) save <- file.path(dir, "answers.json")
  state <- new.env(parent = emptyenv())
  keep <- function(body) {
    if (!identical(body$task_id, task$task_id)) stop("This page belongs to another annotate_emotions() call.")
    writeLines(as.character(jsonlite::toJSON(body, auto_unbox = TRUE, null = "null", na = "null")),
               save, useBytes = TRUE)
    state$doc <- body
    list(ok = TRUE, saved = length(body$answers$id))
  }
  server <- .te_serve_page(dir, port = port, handlers = list(
    answers = keep,
    finish = function(body) {
      out <- keep(body)
      state$finished <- TRUE
      out
    },
    cancel = function(body) {
      state$cancelled <- TRUE
      list(ok = TRUE)
    }
  ))
  on.exit(server$stop(), add = TRUE)
  .te_open_page(server, open, paste(
    "Code the items at %s and click Finish.",
    "R saves every answer to", save, "as you go and waits until then;",
    "press Esc (Ctrl+C in a terminal) to stop waiting, and read the answers with read_annotations()."
  ))
  .te_wait_for(function() isTRUE(state$finished) || isTRUE(state$cancelled))
  if (!isTRUE(state$finished)) {
    message("Cancelled. The answers given so far are in ", save, ".")
    return(invisible(NULL))
  }
  out <- .te_annotation_rows(state$doc, items)
  message(sum(.te_answered(out, questions)), " of ", nrow(items), " items coded.")
  out
}

#' Questions for annotate_emotions()
#'
#' The questions \code{\link{annotate_emotions}} asks about each item. By
#' default: which emotion the coder sees and, when the items have a reference
#' \code{label} and a model \code{prediction}, what explains a disagreement
#' between them, plus a free note.
#'
#' @param classes The emotion classes.
#' @param items Optional: the items, to ask \code{why} only when they have both
#'   \code{label} and \code{prediction}.
#' @param emotion,why,note Logical. Ask which emotion the coder sees, why the
#'   model and the label disagree, and for a note.
#' @param other The option for none of the classes; \code{NULL} for none.
#' @param why_options The answers to the \code{why} question: a named
#'   character vector, names stored in the answers and values shown.
#'
#' @return A list of questions, each a list with \code{id}, \code{prompt},
#'   \code{options} (a named character vector, or \code{NULL} for free text)
#'   and \code{reveal} (\code{TRUE} for questions asked after the label,
#'   prediction and scores are shown).
#'
#' @examples
#' annotation_questions(c("angry", "happy", "sad"))
#'
#' # Only the emotion, with no "none of these"
#' annotation_questions(c("angry", "happy", "sad"), why = FALSE, note = FALSE, other = NULL)
#'
#' @export
annotation_questions <- function(classes, items = NULL, emotion = TRUE, why = NULL, note = TRUE,
                                 other = "none of these",
                                 why_options = c(model = "the model is wrong",
                                                 label = "the label is debatable",
                                                 face = "the wrong face was scored",
                                                 no_face = "no face is visible",
                                                 agree = "the model and the label agree",
                                                 other = "something else")) {
  if (!is.character(classes) || length(classes) < 2) stop("classes must name at least two emotions.", call. = FALSE)
  if (is.null(why)) why <- is.null(items) || all(c("label", "prediction") %in% names(items))
  out <- list()
  if (isTRUE(emotion)) {
    options <- stats::setNames(classes, classes)
    if (!is.null(other)) options <- c(options, other = other)
    out <- c(out, list(list(id = "emotion", prompt = "Which emotion do you see?",
                            options = options, reveal = FALSE)))
  }
  if (isTRUE(why)) {
    out <- c(out, list(list(id = "why",
                            prompt = "Compare with the label and the model. What explains the result?",
                            options = why_options, reveal = TRUE)))
  }
  if (isTRUE(note)) {
    out <- c(out, list(list(id = "note", prompt = "Note (optional)", options = NULL, reveal = TRUE)))
  }
  if (!length(out)) stop("Ask at least one question.", call. = FALSE)
  out
}

#' Read the answers given in annotate_emotions()
#'
#' Reads the answers that \code{\link{annotate_emotions}} saved (its
#' \code{save} file), that \strong{Export answers} downloaded, or that
#' \strong{Copy R code} gave in a notebook.
#'
#' @param x Path of a JSON file of answers, its text, or the
#'   \code{"te-annotations:..."} code.
#' @param items Optional: the items that were coded. Their \code{label},
#'   \code{prediction} and class score columns are added to the answers.
#'
#' @return A data frame with one row per item answered: \code{id},
#'   \code{coder}, one column per question (\code{emotion}, \code{why},
#'   \code{note} by default; \code{NA} if not answered), \code{seconds} spent
#'   on the item and \code{answered_at}, then the item's \code{label},
#'   \code{prediction} and scores, when \code{items} has them.
#'
#' @seealso \code{\link{evaluate_emotions}}: for example
#'   \code{evaluate_emotions(answers, truth_col = "label", pred_col = "emotion")}.
#'
#' @export
read_annotations <- function(x, items = NULL) {
  if (!is.character(x) || length(x) != 1 || is.na(x)) {
    stop("x must be one path, the exported text or the copied code.", call. = FALSE)
  }
  code <- regmatches(x, regexec("^[[:space:]]*te-annotations:([A-Za-z0-9+/=[:space:]]+)$", x))[[1]]
  if (length(code)) {
    x <- tryCatch(rawToChar(jsonlite::base64_dec(gsub("[[:space:]]", "", code[2]))),
                  error = function(e) stop("The answers code is cut off or changed; copy it again.",
                                           call. = FALSE))
    Encoding(x) <- "UTF-8"
  }
  if (!grepl("^[[:space:]]*\\{", x) && !file.exists(x)) stop("File does not exist: ", x, call. = FALSE)
  doc <- tryCatch(jsonlite::fromJSON(x, simplifyVector = TRUE),
                  error = function(e) stop("Cannot read the answers: ", conditionMessage(e), call. = FALSE))
  if (!is.null(items)) items <- .te_check_items(items, NULL, check_files = FALSE)
  .te_annotation_rows(doc, items)
}

# ---------------------------------------------------------------------------

# The page's files in inst/www/emotion-annotator, besides data.js and media/
.te_annotator_files <- c("index.html", "style.css", "logic.js", "app.js")
.te_video_ext <- "\\.(mp4|m4v|mov|webm|mkv|avi|flv|mpg|mpeg|wmv)$"
# Formats a browser plays in a <video> element
.te_playable_ext <- "\\.(mp4|m4v|webm|mov)$"

#' @noRd
# check_files: FALSE when reading answers, which needs no media
.te_check_items <- function(items, classes, check_files = TRUE) {
  if (!is.data.frame(items) || !nrow(items)) stop("items must be a data frame with rows.", call. = FALSE)
  items <- as.data.frame(items, stringsAsFactors = FALSE)
  if (!any(c("media", "text") %in% names(items))) {
    stop("items needs a media column (image or video paths) or a text column.", call. = FALSE)
  }
  if (!"id" %in% names(items)) items$id <- seq_len(nrow(items))
  items$id <- as.character(items$id)
  if (anyNA(items$id) || anyDuplicated(items$id)) stop("items$id must be unique and not NA.", call. = FALSE)
  if ("media" %in% names(items)) items$media <- as.character(items$media)
  if ("media" %in% names(items) && check_files) {
    given <- !is.na(items$media) & nzchar(items$media)
    missing <- items$media[given & !grepl("^https?://", items$media) & !file.exists(items$media)]
    if (length(missing)) {
      stop("Media file(s) not found: ", paste(utils::head(missing, 3), collapse = ", "),
           if (length(missing) > 3) ", ...", call. = FALSE)
    }
  }
  for (col in intersect(c("text", "label", "prediction"), names(items))) items[[col]] <- as.character(items[[col]])
  if (!is.null(classes)) {
    for (col in intersect(c("label", "prediction"), names(items))) {
      other <- setdiff(stats::na.omit(unique(items[[col]])), classes)
      if (length(other)) {
        warning("items$", col, " has values that are not in classes: ",
                paste(utils::head(other, 5), collapse = ", "), call. = FALSE)
      }
    }
  }
  items
}

#' @noRd
.te_check_questions <- function(questions) {
  if (!is.list(questions) || !length(questions)) stop("questions must be a list of questions.", call. = FALSE)
  ids <- vapply(questions, function(q) as.character(q$id), character(1))
  if (anyDuplicated(ids) || any(!grepl("^[A-Za-z][A-Za-z0-9_]*$", ids))) {
    stop("Question ids must be unique names such as emotion or why.", call. = FALSE)
  }
  if (any(ids %in% c("id", "coder", "seconds", "answered_at", "label", "prediction"))) {
    stop("Question ids cannot be id, coder, seconds, answered_at, label or prediction.", call. = FALSE)
  }
  lapply(questions, function(q) {
    opts <- q$options
    if (!is.null(opts)) {
      if (is.null(names(opts))) names(opts) <- opts
      names(opts)[!nzchar(names(opts))] <- opts[!nzchar(names(opts))]
    }
    list(id = as.character(q$id), prompt = as.character(q$prompt),
         options = if (is.null(opts)) NULL else lapply(seq_along(opts), function(i)
           list(value = names(opts)[i], text = unname(opts[i]))),
         reveal = isTRUE(q$reveal))
  })
}

#' @noRd
# The task the page reads from data.js, without the items' media
.te_annotation_task <- function(items, classes, questions, coder, blind, title, how) {
  score_cols <- intersect(classes, names(items))
  instructions <- switch(how,
    live = "Answer the questions for each item. Every answer is sent to R as you go; click Finish when you are done.",
    notebook = "Answer the questions for each item, then Copy R code and run the copied line in a new cell.",
    file = "Answer the questions for each item, then Export answers and read the file in R with read_annotations().")
  list(
    schema = "transforEmotion.annotation_task", schema_version = "1.0.0",
    task_id = format(Sys.time(), "annotate-%Y%m%d-%H%M%S"),
    title = if (is.null(title)) "Annotate emotions" else as.character(title),
    instructions = instructions, coder = if (is.null(coder)) "" else as.character(coder),
    blind = isTRUE(blind), classes = I(classes), score_classes = I(score_cols),
    questions = questions,
    has = list(label = "label" %in% names(items), prediction = "prediction" %in% names(items),
               scores = length(score_cols) > 0)
  )
}

#' @noRd
# One entry per item for the page: its text, label, prediction and scores,
# and its frames or image with face boxes (display copies in dir/media, or
# data: URIs), and the clip itself when it can play in the page
.te_item_media <- function(items, dir, score_cols, frames, faces, play, max_side, data_uri) {
  score_cols <- score_cols[vapply(items[score_cols], is.numeric, logical(1))]
  has_media <- "media" %in% names(items) && any(!is.na(items$media) & nzchar(items$media))
  te_faces <- NULL
  if (has_media) {
    ensure_te_py_env()
    te_faces <- .te_faces_py()
  }
  lapply(seq_len(nrow(items)), function(i) {
    it <- items[i, , drop = FALSE]
    out <- list(id = it$id,
                text = if ("text" %in% names(it)) it$text else NA_character_,
                label = if ("label" %in% names(it)) it$label else NA_character_,
                prediction = if ("prediction" %in% names(it)) it$prediction else NA_character_,
                scores = if (length(score_cols)) as.list(stats::setNames(as.numeric(it[1, score_cols]), score_cols)) else NULL,
                media_type = NA_character_, label_file = NA_character_, frames = list(), video = NA_character_)
    path <- if ("media" %in% names(it)) it$media else NA_character_
    if (is.na(path) || !nzchar(path)) return(out)
    key <- sprintf("i%04d", i)
    is_video <- grepl(.te_video_ext, path, ignore.case = TRUE)
    out$media_type <- if (is_video) "video" else "image"
    out$label_file <- basename(path)
    tmp <- tempfile("te-item-")
    dir.create(tmp)
    on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
    if (is_video) {
      fr <- te_faces$extract_frames(path, as.integer(frames), uniform = TRUE, frame_dir = tmp,
                                    video_name = key)
      paths <- vapply(fr, function(f) f$path, character(1))
      times <- vapply(fr, function(f) as.numeric(f$time), numeric(1))
    } else {
      paths <- path
      times <- NA_real_
    }
    files <- file.path("media", sprintf("%s-%02d.jpg", key, seq_along(paths)))
    sizes <- te_faces$display_copies(as.list(paths), as.list(file.path(dir, files)),
                                     max_side = as.integer(max_side))
    boxes <- if (isTRUE(faces)) .te_item_faces(te_faces, paths, if (is_video) fr else NULL) else NULL
    srcs <- if (isTRUE(data_uri)) .te_data_uri(file.path(dir, files)) else files
    out$frames <- lapply(seq_along(paths), function(k) {
      b <- if (is.null(boxes)) NULL else boxes[boxes$source == paths[k], , drop = FALSE]
      list(src = srcs[k], w = as.integer(sizes[[k]][[1]]), h = as.integer(sizes[[k]][[2]]),
           time = times[k],
           faces = if (is.null(b) || !nrow(b)) list() else lapply(seq_len(nrow(b)), function(j)
             list(x = b$x[j], y = b$y[j], w = b$width[j], h = b$height[j], scored = b$scored[j])))
    })
    if (is_video && isTRUE(play) && grepl(.te_playable_ext, path, ignore.case = TRUE) && file.exists(path)) {
      clip <- file.path("media", paste0(key, ".", tolower(tools::file_ext(path))))
      file.copy(path, file.path(dir, clip))
      out$video <- clip
    }
    out
  })
}

#' @noRd
# Face boxes in the images or frames, and which one is scored by default
# (the largest)
.te_item_faces <- function(te_faces, paths, frames) {
  res <- te_faces$faces_table(as.list(paths), detector = "yunet", min_confidence = 0.6,
                              yunet_model = .te_yunet_model(), frames = frames)
  b <- .te_faces_frame(res, paths, "yunet", list(type = "images"))
  if (!nrow(b)) return(NULL)
  area <- b$width * b$height
  b$scored <- stats::ave(area, b$source, FUN = function(a) seq_along(a) == which.max(a)) == 1
  b
}

#' @noRd
# The answers document (from the page or a file) -> a data frame
.te_annotation_rows <- function(doc, items = NULL) {
  if (!is.list(doc) || !identical(doc$schema, "transforEmotion.annotations")) {
    stop("This is not a set of answers from annotate_emotions().", call. = FALSE)
  }
  ans <- doc$answers
  qids <- as.character(unlist(doc$question_ids))
  if (is.null(ans) || !length(ans) || (is.data.frame(ans) && !nrow(ans))) {
    out <- data.frame(id = character(), coder = character(), stringsAsFactors = FALSE)
    for (q in qids) out[[q]] <- character()
    out$seconds <- numeric()
    out$answered_at <- character()
  } else {
    ans <- as.data.frame(ans, stringsAsFactors = FALSE)
    out <- data.frame(id = as.character(ans$id), stringsAsFactors = FALSE)
    out$coder <- if (!is.null(doc$coder) && nzchar(doc$coder)) as.character(doc$coder) else NA_character_
    for (q in qids) {
      v <- if (q %in% names(ans)) as.character(ans[[q]]) else rep(NA_character_, nrow(ans))
      v[!is.na(v) & !nzchar(v)] <- NA_character_
      out[[q]] <- v
    }
    out$seconds <- if ("seconds" %in% names(ans)) as.numeric(ans$seconds) else NA_real_
    out$answered_at <- if ("answered_at" %in% names(ans)) as.character(ans$answered_at) else NA_character_
  }
  if (!is.null(items)) {
    unknown <- setdiff(out$id, items$id)
    if (length(unknown)) {
      warning(length(unknown), " answer(s) are for items not in items (",
              paste(utils::head(unknown, 5), collapse = ", "), "); were they coded on another set?",
              call. = FALSE)
    }
    extra <- intersect(names(items), c("label", "prediction"))
    scores <- names(items)[vapply(items, is.numeric, logical(1))]
    scores <- setdiff(scores, c("id", names(out)))
    keep <- unique(c(extra, scores))
    if (length(keep)) {
      m <- match(out$id, items$id)
      for (col in keep) out[[col]] <- items[[col]][m]
    }
  }
  attr(out, "task_id") <- doc$task_id
  out
}

#' @noRd
# Items with every choice question answered
.te_answered <- function(answers, questions) {
  ids <- vapply(questions, function(q) if (!is.null(q$options)) q$id else NA_character_, character(1))
  ids <- intersect(stats::na.omit(ids), names(answers))
  if (!length(ids)) return(rep(TRUE, nrow(answers)))
  stats::complete.cases(answers[, ids, drop = FALSE])
}
