#' Questions for a face annotation task
#'
#' Ready-made questions for \code{\link{annotation_task}}. Each question is a
#' list, so you can change them or add your own (see Details).
#'
#' @param which Names of the ready-made questions, in the order they are
#'   asked:
#'   \describe{
#'     \item{\code{"face_check"}}{Does the outline contain a face? Yes, not a
#'       face, a different person, or can't tell. The other questions are
#'       asked only after "Yes", so detector mistakes never get an emotion.}
#'     \item{\code{"emotion"}}{Which emotion the face shows, one of the
#'       task's \code{classes}.}
#'     \item{\code{"intensity"}}{How intense it is, 1 to 7.}
#'     \item{\code{"valence"}}{How negative or positive the face is, -3 to 3.}
#'     \item{\code{"arousal"}}{How calm or excited the face is, 0 to 6.}
#'     \item{\code{"why"}}{For items with a reference label and a model
#'       prediction (\code{annotation_task(items = ...)}): what explains the
#'       result, compared with the label and the model. The model is wrong, the
#'       label is debatable, the wrong face was scored, no face is visible, the
#'       model and the label agree, or something else. Asked after the label,
#'       the prediction and the scores are shown (\code{reveal = TRUE}).}
#'   }
#'
#' @details
#' A question is a list with
#' \describe{
#'   \item{\code{id}}{A name in lower case: letters, digits and underscores.}
#'   \item{\code{type}}{\code{"single"} (one of a few options) or
#'     \code{"likert"} (a whole number from \code{min} to \code{max}).}
#'   \item{\code{prompt}}{The question as coders read it.}
#'   \item{\code{options}}{For \code{"single"}: the answers, or
#'     \code{options_from = "classes"} for the task's classes. Optional
#'     \code{labels}: a named list of the text shown for each answer.}
#'   \item{\code{min}, \code{max}}{For \code{"likert"}, with optional
#'     \code{anchors = list(min = "...", max = "...")}.}
#'   \item{\code{show_if}}{Optional: ask only after another question got one
#'     of these answers, as in \code{list(face_check = "ok")}.}
#'   \item{\code{reveal}}{Optional, for items with a reference label and a
#'     model prediction: \code{TRUE} asks the question only once they are
#'     shown, which with \code{blind = TRUE} is after the other questions are
#'     answered.}
#'   \item{\code{required}}{\code{FALSE} lets coders leave it unanswered.}
#' }
#'
#' @return A list of questions.
#'
#' @seealso \code{\link{annotation_task}}
#'
#' @examples
#' annotation_questions(c("face_check", "emotion"))
#'
#' # Add a question of your own
#' gaze <- list(id = "gaze", type = "single", prompt = "Where does the person look?",
#'              options = c("camera", "away", "closed"), show_if = list(face_check = "ok"))
#' questions <- c(annotation_questions(), list(gaze))
#'
#' @export
annotation_questions <- function(which = c("face_check", "emotion", "intensity")) {
  known <- .te_preset_questions()
  bad <- setdiff(which, names(known))
  if (length(bad)) {
    stop("Unknown questions: ", paste(bad, collapse = ", "), ". Choose from ",
         paste(names(known), collapse = ", "), ".", call. = FALSE)
  }
  out <- unname(known[which])
  # Without face_check, the others are always asked
  lapply(out, function(q) {
    if (!is.null(q$show_if) && !all(names(q$show_if) %in% which)) q$show_if <- NULL
    q
  })
}

#' Set up a face annotation task
#'
#' Turns detected faces into a coding task: each face is an item that coders
#' see outlined in its picture, with a close-up, and answer questions about,
#' such as whether it is a face at all and which emotion it shows. Start the
#' coding page with \code{\link{annotate_faces}}, then read the answers with
#' \code{\link{read_annotations}}.
#'
#' With \code{items} instead of \code{faces}, each item is a video clip, an
#' image or a text, for example the clips a model got wrong. Coders see
#' sampled frames of the clip with its face boxes (the largest face, the one
#' \code{\link{video_scores}} scores by default, drawn solid), the clip itself
#' with its sound, and the item's text. When the items have a reference
#' \code{label} and a model \code{prediction}, the page shows them with the
#' model's scores once the coder has answered (\code{blind = TRUE}), then
#' asks what explains the result (\code{annotation_questions("why")}). Unlike
#' a face task, such a page holds the texts, labels and scores.
#'
#' The task is a folder:
#' \describe{
#'   \item{\code{page/}}{The coding page: small copies of the images or video
#'     frames, under names that say nothing about the files, and the
#'     questions. It holds no file names, scores, or other coders' answers.}
#'   \item{\code{items.csv}}{Which face each item is: its file, frame, face
#'     number, track and person. For you, not for coders.}
#'   \item{\code{task.json}}{The questions, items and each coder's list.}
#'   \item{\code{results/}}{The answers, one file per coder.}
#' }
#'
#' @param faces A data frame returned by \code{\link{detect_faces}} or
#'   \code{\link{select_faces}}: one item per row. Or \code{NULL}, with
#'   \code{items}.
#' @param classes The emotions coders choose from, for example the classes
#'   passed to \code{\link{face_scores}}. Needed by the \code{"emotion"}
#'   question.
#' @param questions The questions, from \code{\link{annotation_questions}}.
#'   Default: \code{annotation_questions()} for faces;
#'   \code{annotation_questions(c("emotion", "why"))} for items with a
#'   \code{label} and a \code{prediction}, else \code{"emotion"}.
#' @param coders Coder ids: letters, digits, dots, dashes or underscores.
#' @param overlap Share of the items that every coder codes, for agreement.
#'   The rest are split between the coders. Default is 1: every coder codes
#'   every item.
#' @param dir A new or empty folder for the task.
#' @param seed Seed for the order of the items. Each coder gets their own
#'   order. Your own random numbers are left as they were.
#' @param title Title shown on the page.
#' @param instructions Text shown to coders before the first item.
#' @param max_side Longest side, in pixels, of the picture copies.
#' @param task_id An id for the task. Coders' progress in the browser is
#'   kept under it. Default is made from the time.
#' @param items Instead of \code{faces}: a data frame with one row per item
#'   and the columns \code{media} (path of a video or image file) and/or
#'   \code{text}, and optionally \code{id}, \code{label} (the reference
#'   label, returned as \code{reference}), \code{prediction} (the model's top
#'   class) and one column of model scores per class.
#' @param frames For items: the number of frames shown per video clip, spread
#'   over the clip as \code{video_scores(uniform = TRUE)} samples them.
#' @param play For items: put the clips themselves in the page, with their
#'   sound, when a browser can play them (MP4, WebM, MOV).
#' @param blind For items: show the label, prediction and scores only after
#'   the questions without \code{reveal} are answered.
#'
#' @return The task, invisibly: a list with its folder (\code{dir}),
#'   \code{task_id}, \code{questions}, \code{coders} and \code{items} (the
#'   contents of \code{items.csv}).
#'
#' @seealso \code{\link{annotate_faces}}, \code{\link{read_annotations}},
#'   \code{\link{annotation_agreement}}, \code{\link{consensus_labels}}
#'
#' @examples
#' \dontrun{
#' faces <- detect_faces(system.file("extdata", "two-people.mp4", package = "transforEmotion"),
#'                       nframes = 20)
#' task <- annotation_task(faces, classes = c("happy", "sad", "angry", "neutral"),
#'                         coders = c("C01", "C02"), dir = "annotation-round-1")
#' annotate_faces(task)
#'
#' # A model's errors on video clips, with their descriptions
#' emotions <- c("angry", "fearful", "happy", "sad")
#' items <- data.frame(id = errors$clip, media = errors$path, text = errors$description,
#'                     label = errors$label, prediction = errors$predicted, errors[, emotions])
#' review <- annotation_task(items = items, classes = emotions, dir = "error-review")
#' annotate_faces(review)
#' answers <- read_annotations(review)
#' table(answers$value[answers$question == "why"])
#' }
#'
#' @export
annotation_task <- function(faces = NULL, classes = NULL, questions = NULL, coders = "C01",
                            overlap = 1, dir, seed = 1, title = NULL, instructions = NULL,
                            max_side = 1280, task_id = NULL, items = NULL, frames = 4,
                            play = TRUE, blind = TRUE) {
  if (!is.null(items)) {
    if (!is.null(faces)) stop("Give faces or items, not both.", call. = FALSE)
    items <- .te_check_review_items(items, classes)
  } else {
    if (is.null(faces)) stop("Give faces from detect_faces(), or items.", call. = FALSE)
    .te_check_faces_table(faces)
    if (!nrow(faces)) stop("faces has no rows to code.", call. = FALSE)
  }
  if (is.null(questions)) {
    questions <- if (is.null(items)) annotation_questions() else .te_review_questions(items)
  }
  if (missing(dir) || !is.character(dir) || length(dir) != 1) {
    stop("dir must be the folder for the task.", call. = FALSE)
  }
  if (dir.exists(dir) && length(list.files(dir, all.files = TRUE, no.. = TRUE))) {
    stop("dir must be a new or empty folder: ", dir, call. = FALSE)
  }
  questions <- .te_check_questions(questions, classes)
  if (!is.character(coders) || !length(coders) || anyDuplicated(coders) ||
      !all(grepl("^[A-Za-z0-9_.-]{1,40}$", coders))) {
    stop("coders must be different ids made of letters, digits, dots, dashes or underscores.",
         call. = FALSE)
  }
  if (!is.numeric(overlap) || length(overlap) != 1 || is.na(overlap) || overlap <= 0 || overlap > 1) {
    stop("overlap must be a share above 0 and up to 1.", call. = FALSE)
  }
  if (is.null(task_id)) task_id <- format(Sys.time(), "annotation-%Y%m%d-%H%M%S")
  if (!grepl("^[A-Za-z0-9_.-]{1,80}$", task_id)) {
    stop("task_id must be letters, digits, dots, dashes or underscores.", call. = FALSE)
  }
  if (!is.null(items)) {
    return(.te_item_annotation_task(items, classes, questions, coders, overlap, dir, seed, title,
                                    instructions, max_side, task_id, frames, play, blind))
  }

  settings <- attr(faces, "settings")
  is_video <- .te_is_video_faces(faces, settings)
  rows <- as.data.frame(unclass(faces), stringsAsFactors = FALSE, check.names = FALSE)
  column <- function(name, fill = NA) if (name %in% names(rows)) rows[[name]] else rep(fill, nrow(rows))

  plan <- .te_with_seed(seed, {
    order <- sample.int(nrow(rows))
    list(order = order, assignments = .te_assign_items(length(order), coders, overlap))
  })
  rows <- rows[plan$order, , drop = FALSE]
  n <- nrow(rows)
  item_ids <- sprintf("i%0*d", max(4, nchar(n)), seq_len(n))

  # One picture per image or frame, numbered in the items' (random) order
  frame_index <- as.integer(column("frame_index"))
  picture_of <- paste(rows$source, ifelse(is.na(frame_index), "", frame_index))
  pictures <- unique(picture_of)
  picture_file <- sprintf("media/p%0*d.jpg", max(3, nchar(length(pictures))), seq_along(pictures))
  first <- match(pictures, picture_of)

  page <- file.path(dir, "page")
  dir.create(file.path(page, "media"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(dir, "results"), showWarnings = FALSE)
  ensure_te_py_env()
  te_faces <- .te_faces_py()
  if (is_video) {
    frame_table <- data.frame(source = rows$source[first], frame_index = frame_index[first],
                              stringsAsFactors = FALSE)
    attr(frame_table, "media") <- attr(faces, "media")
    images <- .te_face_frame_paths(frame_table, settings)
    cleanup <- attr(images, "cleanup")
    on.exit(cleanup(), add = TRUE)
  } else {
    images <- rows$source[first]
  }
  sizes <- te_faces$display_copies(as.list(as.character(images)),
                                   as.list(file.path(page, picture_file)),
                                   max_side = as.integer(max_side))
  width <- vapply(sizes, function(s) as.integer(s[[1]]), integer(1))[match(picture_of, pictures)]
  height <- vapply(sizes, function(s) as.integer(s[[2]]), integer(1))[match(picture_of, pictures)]

  items <- data.frame(
    item_id = item_ids, source = rows$source, frame = as.integer(column("frame")),
    frame_index = frame_index, time = as.numeric(column("time")),
    face = as.integer(column("face")), track_id = as.integer(column("track_id")),
    person_id = as.character(column("person_id")),
    x = rows$x, y = rows$y, width = rows$width, height = rows$height,
    picture = picture_file[match(picture_of, pictures)],
    stringsAsFactors = FALSE
  )
  assignments <- lapply(seq_along(coders), function(k) {
    list(coder_id = coders[k], items = I(item_ids[plan$assignments[[k]]]))
  })
  manifest <- list(
    schema = "transforEmotion.annotation_task", schema_version = "1.0.0",
    task_id = task_id, mode = "annotate", unit = "face",
    title = if (is.null(title)) "Face annotation" else as.character(title),
    instructions = if (is.null(instructions)) .te_default_instructions(questions) else as.character(instructions),
    classes = I(as.character(classes)),
    questions = questions,
    settings = list(allow_back = TRUE, allow_skip = TRUE, show_progress = TRUE, context_padding = 0.25),
    items = lapply(seq_len(n), function(i) list(
      item_id = item_ids[i], media = items$picture[i],
      media_type = if (is_video) "frame" else "image",
      frame_index = if (is_video) frame_index[i] else NULL,
      timestamp_s = if (is_video) items$time[i] else NULL,
      width = width[i], height = height[i],
      box = list(x = items$x[i], y = items$y[i], w = items$width[i], h = items$height[i])
    )),
    assignments = assignments
  )
  json <- jsonlite::toJSON(manifest, auto_unbox = TRUE, null = "null", na = "null",
                           digits = 4, pretty = TRUE)
  writeLines(json, file.path(dir, "task.json"), useBytes = TRUE)
  writeLines(paste0("window.TE_TASK = ", json, ";"), file.path(page, "data.js"), useBytes = TRUE)
  file.copy(system.file("www", "face-annotator", .te_face_app_files, package = "transforEmotion",
                        mustWork = TRUE), page, overwrite = TRUE)
  utils::write.csv(items, file.path(dir, "items.csv"), row.names = FALSE, na = "")

  task <- .te_annotation_task(normalizePath(dir), manifest, items)
  message("The task is in ", task$dir, ": ", n, " items for ", length(coders), " coder(s). ",
          "Start coding with annotate_faces().")
  invisible(task)
}

#' Code faces in the browser
#'
#' Starts the coding page of an \code{\link{annotation_task}} from R and
#' opens it in the web browser. Each answer is saved in the task's
#' \code{results} folder as soon as the coder gives it, so closing the
#' browser or R loses nothing, and a coder who comes back continues where
#' they stopped.
#'
#' The page is served by R (with \pkg{httpuv}) on your own computer only
#' (\code{127.0.0.1}), at an address with a random part, and keeps running
#' while you use R. Coders press the number keys to answer, Enter for the
#' next item, and Backspace to undo.
#'
#' Coders on other computers can open \code{page/index.html} of the task
#' from a copy of the folder instead. Their page keeps answers in that
#' browser and ends with \strong{Export annotations}; put the exported files
#' in the task's \code{results} folder.
#'
#' In a Jupyter notebook, including Google Colab, the coding page is shown
#' below the cell instead (\code{inline = TRUE}, the default there), because
#' a page served from R cannot be reached from a notebook running on another
#' computer. Its \strong{Copy R code} button gives a line such as
#' \code{answers <- read_annotations(task, "te-annotations:...")} to run in
#' a new cell. The line holds every answer given so far, so copy it again
#' after more coding; keep the last one, as the answers are not saved in the
#' task's \code{results} folder.
#'
#' @param task An \code{\link{annotation_task}}, or its folder.
#' @param coder The coder id to start with. Default: the page asks.
#' @param open Logical. Open the page in the web browser. Default is
#'   \code{TRUE} in an interactive session.
#' @param port Port of the local server. Default is a free port chosen at
#'   random.
#' @param inline Logical. Show the coding page inside the notebook, below the
#'   cell. Default is \code{TRUE} in a Jupyter notebook (Google Colab,
#'   JupyterLab, VS Code) and \code{FALSE} elsewhere.
#' @param max_side Longest side, in pixels, of the pictures in a notebook
#'   page, which stores them in the notebook. Default 640.
#'
#' @return Invisibly, the page: a list with its \code{url} and
#'   \code{stop()}, which stops the server. With \code{inline = TRUE},
#'   invisibly, the path of the page as one self-contained file.
#'
#' @seealso \code{\link{read_annotations}}, \code{\link{annotation_progress}}
#'
#' @examples
#' \dontrun{
#' coding <- annotate_faces("annotation-round-1", coder = "C01")
#' annotation_progress("annotation-round-1")
#' coding$stop()
#' }
#'
#' @export
annotate_faces <- function(task, coder = NULL, open = interactive(), port = NULL,
                           inline = .te_in_notebook(), max_side = 640) {
  task_code <- .te_code_name(substitute(task), "task")
  task <- .te_load_annotation_task(task)
  if (!is.null(coder) && !coder %in% task$coders) {
    stop("coder must be one of the task's coders: ", paste(task$coders, collapse = ", "), call. = FALSE)
  }
  if (isTRUE(inline)) return(.te_annotate_in_notebook(task, coder, task_code, max_side))
  results <- file.path(task$dir, "results")
  dir.create(results, showWarnings = FALSE)
  assigned <- stats::setNames(lapply(task$assignments, function(a) as.character(unlist(a$items))),
                              vapply(task$assignments, function(a) a$coder_id, character(1)))
  check_coder <- function(id) {
    if (!is.character(id) || length(id) != 1 || !id %in% names(assigned)) {
      stop("Unknown coder.")
    }
    id
  }
  handlers <- list(
    # One or more records; each is appended to results/<coder>.jsonl
    save = structure(function(body) {
      records <- body$records
      if (!is.list(records) || !length(records)) stop("No records.")
      lines <- vapply(records, function(r) {
        if (!is.list(r) || !identical(r$task_id, task$task_id)) stop("The record belongs to another task.")
        id <- check_coder(r$coder_id)
        if (!is.character(r$item_id) || !isTRUE(r$item_id %in% assigned[[id]])) {
          stop("The item is not on this coder's list.")
        }
        if (!isTRUE(r$status %in% c("answered", "skipped"))) stop("A record needs status answered or skipped.")
        as.character(jsonlite::toJSON(.te_flat_record(r), auto_unbox = TRUE, null = "null", na = "null"))
      }, character(1))
      coders <- vapply(records, function(r) r$coder_id, character(1))
      for (id in unique(coders)) {
        cat(paste0(lines[coders == id], "\n"), file = file.path(results, paste0(id, ".jsonl")),
            sep = "", append = TRUE)
      }
      list(ok = TRUE, saved = length(lines))
    }, simplify = FALSE),
    # What a coder saved before, so they continue where they stopped
    load = function(body) {
      id <- check_coder(body$coder_id)
      file <- file.path(results, paste0(id, ".jsonl"))
      lines <- if (file.exists(file)) readLines(file, warn = FALSE, encoding = "UTF-8") else character()
      lines <- lines[nzchar(trimws(lines))]
      list(ok = TRUE, records = structure(paste0("[", paste(lines, collapse = ","), "]"), class = "json"))
    }
  )
  data_js <- tempfile("te-data-", fileext = ".js")
  writeLines(c(readLines(file.path(task$dir, "page", "data.js"), warn = FALSE, encoding = "UTF-8"),
               'window.TE_API = {"save":"api/save","load":"api/load"};'),
             data_js, useBytes = TRUE)
  server <- .te_serve_page(file.path(task$dir, "page"), handlers = handlers, port = port,
                           files = list("data.js" = data_js))
  stop_server <- function() {
    server$stop()
    unlink(data_js)
    invisible(NULL)
  }
  page <- server
  if (!is.null(coder)) page$url <- paste0(server$url, "?coder=", utils::URLencode(coder, reserved = TRUE))
  .te_open_page(page, open, paste0(
    "Coding page at %s. Answers are saved in ", results, " as coders go. ",
    "It runs until R ends; to stop it sooner, keep what annotate_faces() returns and call its $stop()."
  ))
  invisible(structure(list(url = page$url, dir = task$dir, stop = stop_server),
                      class = "te_face_page"))
}

#' Read the answers of a face annotation task
#'
#' Reads what coders answered in \code{\link{annotate_faces}}: the files in
#' the task's \code{results} folder, and any exported files you add.
#'
#' @param task An \code{\link{annotation_task}}, or its folder.
#' @param files Answer files to read, or \code{"te-annotations:..."} codes
#'   from the \strong{Copy R code} button of a coding page in a notebook.
#'   Default: every \code{.jsonl} and \code{.json} file in the task's
#'   \code{results} folder.
#' @param revisions \code{"last"} (default) keeps each coder's last answer
#'   to each item; \code{"all"} keeps every change, for checking.
#'
#' @return A data frame with one row per item, coder and question:
#'   \code{item_id}, \code{coder_id}, \code{question}, \code{value} (the
#'   answer as text, \code{NA} when the question was not asked or the item
#'   was skipped), \code{value_num} (the answer as a number, for Likert
#'   questions), \code{status} (\code{"answered"} or \code{"skipped"}),
#'   \code{revision}, \code{rt_ms} (time on the item), \code{shown_at},
#'   \code{submitted_at} and \code{session_id}, followed by the item's face:
#'   \code{source}, \code{frame}, \code{frame_index}, \code{time},
#'   \code{face}, \code{track_id} and \code{person_id}. For a task made from
#'   \code{items}: the item's \code{id}, \code{kind}, \code{media},
#'   \code{reference} (its \code{label}), \code{prediction} and score
#'   columns.
#'
#' @seealso \code{\link{annotation_agreement}}, \code{\link{consensus_labels}}
#'
#' @export
read_annotations <- function(task, files = NULL, revisions = c("last", "all")) {
  task <- .te_load_annotation_task(task)
  revisions <- match.arg(revisions)
  if (is.null(files)) {
    files <- list.files(file.path(task$dir, "results"), pattern = "\\.(jsonl|json)$",
                        full.names = TRUE)
  }
  is_code <- grepl("^[[:space:]]*te-annotations:", files)
  missing_files <- files[!is_code & !file.exists(files)]
  if (length(missing_files)) stop("File does not exist: ", missing_files[1], call. = FALSE)
  records <- unlist(lapply(seq_along(files), function(i) {
    if (is_code[i]) .te_read_record_code(files[i]) else .te_read_record_file(files[i])
  }), recursive = FALSE)
  qids <- vapply(task$questions, function(q) q$id, character(1))
  if (length(records)) {
    other <- vapply(records, function(r) !identical(r$task_id, task$task_id), logical(1))
    if (any(other)) {
      warning(sum(other), " answer(s) from another task were left out.", call. = FALSE)
      records <- records[!other]
    }
  }
  records <- Filter(function(r) !is.null(r$item_id) && r$item_id %in% task$items$item_id, records)
  out <- .te_records_table(records, qids)
  if (nrow(out)) {
    # The same record can be in a coder's .jsonl and in an export
    key <- paste(out$coder_id, out$session_id, out$seq, sep = "\r")
    out <- out[!duplicated(key), , drop = FALSE]
    if (revisions == "last") {
      out <- out[order(out$coder_id, out$item_id, out$submitted_at, out$revision, out$seq), , drop = FALSE]
      last <- !duplicated(paste(out$coder_id, out$item_id, sep = "\r"), fromLast = TRUE)
      out <- out[last, , drop = FALSE]
    }
  }
  long <- .te_long_answers(out, task$questions)
  item_cols <- .te_item_columns(task)
  long <- cbind(long, task$items[match(long$item_id, task$items$item_id), item_cols, drop = FALSE])
  rownames(long) <- NULL
  attr(long, "questions") <- task$questions
  attr(long, "task_id") <- task$task_id
  long
}

#' Progress of a face annotation task
#'
#' @param task An \code{\link{annotation_task}}, or its folder.
#'
#' @return A data frame with one row per coder: items assigned, answered,
#'   skipped and left, and the median seconds per answered item.
#'
#' @seealso \code{\link{annotate_faces}}
#'
#' @export
annotation_progress <- function(task) {
  task <- .te_load_annotation_task(task)
  ann <- read_annotations(task)
  first_q <- task$questions[[1]]$id
  ann <- ann[ann$question == first_q, , drop = FALSE]
  rows <- lapply(task$assignments, function(a) {
    items <- as.character(unlist(a$items))
    mine <- ann[ann$coder_id == a$coder_id & ann$item_id %in% items, , drop = FALSE]
    answered <- sum(mine$status == "answered")
    skipped <- sum(mine$status == "skipped")
    data.frame(coder_id = a$coder_id, assigned = length(items), answered = answered,
               skipped = skipped, left = length(items) - answered - skipped,
               median_seconds = if (answered) stats::median(mine$rt_ms[mine$status == "answered"]) / 1000 else NA_real_,
               stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
}

#' Agreement between coders
#'
#' Krippendorff's alpha, Fleiss' kappa and percent agreement for each
#' question of a face annotation task.
#'
#' @param annotations The result of \code{\link{read_annotations}}.
#' @param questions Questions to compute agreement for. Default: all.
#' @param level Level of measurement for alpha, as a named character vector
#'   (for example \code{c(intensity = "ordinal")}): \code{"nominal"},
#'   \code{"ordinal"}, \code{"interval"} or \code{"ratio"}. Default:
#'   \code{"nominal"} for single-choice questions and \code{"interval"} for
#'   Likert questions.
#'
#' @details
#' Only items with answers from at least two coders count; a question that
#' was not asked (after "Not a face", say) is missing, not an answer.
#'
#' \describe{
#'   \item{\code{alpha}}{Krippendorff's alpha, from the coincidence matrix.
#'     It allows for missing answers and for items coded by different
#'     numbers of coders.}
#'   \item{\code{fleiss_kappa}}{Fleiss' kappa, when every counted item has
#'     answers from the same number of coders; \code{NA} otherwise. Treats
#'     answers as categories.}
#'   \item{\code{percent_agreement}}{The share of pairs of coders who gave
#'     the same answer to an item, averaged over items.}
#' }
#'
#' @return A data frame with one row per question: \code{question},
#'   \code{level}, \code{n_items}, \code{n_coders}, \code{n_answers},
#'   \code{percent_agreement}, \code{alpha} and \code{fleiss_kappa}.
#'
#' @references
#' Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability.
#' \url{https://repository.upenn.edu/items/034a6030-c584-4d14-9d3d-7b7e8d16df20}
#'
#' Fleiss, J. L. (1971). Measuring nominal scale agreement among many
#' raters. \emph{Psychological Bulletin, 76}(5), 378-382.
#'
#' @seealso \code{\link{consensus_labels}}
#'
#' @export
annotation_agreement <- function(annotations, questions = NULL, level = NULL) {
  .te_check_annotations(annotations)
  specs <- attr(annotations, "questions")
  if (is.null(questions)) questions <- unique(annotations$question)
  rows <- lapply(questions, function(qid) {
    spec <- Filter(function(q) identical(q$id, qid), specs)
    lvl <- if (!is.null(level) && qid %in% names(level)) {
      level[[qid]]
    } else if (length(spec) && identical(spec[[1]]$type, "likert")) {
      "interval"
    } else {
      "nominal"
    }
    lvl <- match.arg(lvl, c("nominal", "ordinal", "interval", "ratio"))
    a <- annotations[annotations$question == qid & annotations$status == "answered" &
                       !is.na(annotations$value), , drop = FALSE]
    numeric <- lvl != "nominal"
    if (numeric && anyNA(a$value_num)) stop("Question ", qid, " has answers that are not numbers.", call. = FALSE)
    values <- if (numeric) a$value_num else a$value
    units <- split(values, a$item_id)
    units <- units[lengths(units) >= 2]
    data.frame(
      question = qid, level = lvl, n_items = length(units),
      n_coders = length(unique(a$coder_id[a$item_id %in% names(units)])),
      n_answers = sum(lengths(units)),
      percent_agreement = .te_percent_agreement(units),
      alpha = .te_kripp_alpha(units, lvl),
      fleiss_kappa = .te_fleiss_kappa(units),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

#' One label per item from several coders
#'
#' Combines the coders' answers to one question into one value per item: the
#' most common answer, or the mean or median of Likert answers. Use it as
#' the truth for \code{\link{evaluate_emotions}}.
#'
#' @param annotations The result of \code{\link{read_annotations}}.
#' @param question The question to combine. Default is \code{"emotion"}.
#' @param method \code{"majority"} (the most common answer), \code{"mean"} or
#'   \code{"median"}.
#' @param min_coders Items with fewer answers than this get \code{NA}.
#'
#' @return A data frame with one row per item: \code{item_id}, \code{label}
#'   (the combined answer, \code{NA} for a tie or too few answers),
#'   \code{n_coders}, \code{agreement} (the share of coders who gave the
#'   majority answer; \code{NA} for the mean or median), \code{tie}, and the
#'   item's face (\code{source}, \code{frame}, \code{frame_index},
#'   \code{time}, \code{face}, \code{track_id}, \code{person_id}), or, for
#'   a task made from \code{items}, its \code{id}, \code{reference},
#'   \code{prediction} and scores: compare the coders with the reference by
#'   \code{evaluate_emotions(consensus, truth_col = "reference", pred_col = "label")}.
#'
#' @seealso \code{\link{annotation_agreement}}
#'
#' @export
consensus_labels <- function(annotations, question = "emotion",
                             method = c("majority", "mean", "median"), min_coders = 1) {
  .te_check_annotations(annotations)
  method <- match.arg(method)
  if (!question %in% annotations$question) {
    stop("The annotations have no question ", question, ".", call. = FALSE)
  }
  a <- annotations[annotations$question == question, , drop = FALSE]
  items <- unique(a$item_id)
  given <- a[a$status == "answered" & !is.na(a$value), , drop = FALSE]
  rows <- lapply(items, function(id) {
    v <- given[given$item_id == id, , drop = FALSE]
    n <- nrow(v)
    label <- NA
    agreement <- NA_real_
    tie <- FALSE
    if (n >= max(1, min_coders)) {
      if (method == "majority") {
        counts <- sort(table(v$value), decreasing = TRUE)
        tie <- length(counts) > 1 && counts[[1]] == counts[[2]]
        if (!tie) label <- names(counts)[1]
        agreement <- counts[[1]] / n
      } else {
        if (anyNA(v$value_num)) stop("Question ", question, " has answers that are not numbers.", call. = FALSE)
        label <- if (method == "mean") mean(v$value_num) else stats::median(v$value_num)
      }
    }
    data.frame(item_id = id, label = label, n_coders = n, agreement = agreement, tie = tie,
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, rows)
  item_cols <- setdiff(names(a), c("item_id", "coder_id", "question", "value", "value_num", "status",
                                   "revision", "rt_ms", "shown_at", "submitted_at", "session_id"))
  cbind(out, a[match(out$item_id, a$item_id), item_cols, drop = FALSE], row.names = NULL)
}

#' @export
print.te_annotation_task <- function(x, ...) {
  cat(if (identical(x$unit, "item")) "Annotation task " else "Face annotation task ", x$task_id, "\n",
      "Folder: ", x$dir, "\n",
      nrow(x$items), " items; coders: ", paste(x$coders, collapse = ", "), "\n",
      "Questions: ", paste(vapply(x$questions, function(q) q$id, character(1)), collapse = ", "), "\n",
      sep = "")
  invisible(x)
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#' @noRd
.te_preset_questions <- function() {
  shown <- list(face_check = "ok")
  list(
    face_check = list(
      id = "face_check", type = "single", prompt = "Does the outline contain a face?",
      options = c("ok", "not_face", "wrong_person", "unclear"),
      labels = list(ok = "Yes, a face", not_face = "Not a face",
                    wrong_person = "A different person", unclear = "Can't tell"),
      required = TRUE
    ),
    emotion = list(id = "emotion", type = "single", prompt = "Which emotion does the face show?",
                   options_from = "classes", show_if = shown, required = TRUE),
    intensity = list(id = "intensity", type = "likert", prompt = "How intense is it?",
                     min = 1L, max = 7L, anchors = list(min = "very weak", max = "very strong"),
                     show_if = shown, required = TRUE),
    valence = list(id = "valence", type = "likert", prompt = "How negative or positive is the face?",
                   min = -3L, max = 3L, anchors = list(min = "very negative", max = "very positive"),
                   show_if = shown, required = TRUE),
    arousal = list(id = "arousal", type = "likert", prompt = "How calm or excited is the face?",
                   min = 0L, max = 6L, anchors = list(min = "very calm", max = "very excited"),
                   show_if = shown, required = TRUE),
    why = list(id = "why", type = "single",
               prompt = "Compare with the label and the model. What explains the result?",
               options = c("model", "label", "face", "no_face", "agree", "other"),
               labels = list(model = "The model is wrong", label = "The label is debatable",
                             face = "The wrong face was scored", no_face = "No face is visible",
                             agree = "The model and the label agree", other = "Something else"),
               reveal = TRUE, required = TRUE)
  )
}

#' @noRd
# Checks questions and returns them ready for JSON (options stay arrays)
.te_check_questions <- function(questions, classes) {
  if (!is.list(questions) || !length(questions)) stop("questions must be a list of questions.", call. = FALSE)
  if (!is.null(questions$id)) questions <- list(questions) # one question on its own
  seen <- character()
  out <- lapply(questions, function(q) {
    if (!is.list(q) || !is.character(q$id) || length(q$id) != 1 || !grepl("^[a-z][a-z0-9_]{0,39}$", q$id)) {
      stop("Each question needs an id in lower case (letters, digits, underscores).", call. = FALSE)
    }
    if (q$id %in% seen) stop("Two questions have the id ", q$id, ".", call. = FALSE)
    if (is.null(q$type) || !q$type %in% c("single", "likert")) {
      stop("Question ", q$id, " needs type \"single\" or \"likert\".", call. = FALSE)
    }
    if (is.null(q$prompt)) q$prompt <- gsub("_", " ", q$id)
    if (q$type == "single") {
      if (identical(q$options_from, "classes")) {
        if (!is.character(classes) || length(classes) < 2 || anyDuplicated(classes)) {
          stop("classes must be at least two different emotions.", call. = FALSE)
        }
        options <- classes
      } else {
        options <- q$options
        if (!is.character(options) || !length(options) || anyDuplicated(options)) {
          stop("Question ", q$id, " needs options: different answers as text.", call. = FALSE)
        }
        q$options <- I(options)
      }
      if (!is.null(q$labels)) q$labels <- as.list(q$labels)
    } else {
      if (!is.numeric(q$min) || !is.numeric(q$max) || q$min != round(q$min) ||
          q$max != round(q$max) || q$min >= q$max || q$max - q$min > 10) {
        stop("Question ", q$id, " needs whole numbers min < max, at most 10 apart.", call. = FALSE)
      }
      q$min <- as.integer(q$min)
      q$max <- as.integer(q$max)
      options <- as.character(seq(q$min, q$max))
    }
    if (!is.null(q$show_if)) {
      if (!is.list(q$show_if) || is.null(names(q$show_if)) || !all(names(q$show_if) %in% seen)) {
        stop("show_if of question ", q$id, " must name questions asked before it.", call. = FALSE)
      }
      q$show_if <- lapply(q$show_if, function(v) if (length(v) > 1) I(v) else v)
    }
    q$required <- !isFALSE(q$required)
    q$reveal <- if (isTRUE(q$reveal)) TRUE else NULL
    attr(q, "answers") <- options
    seen <<- c(seen, q$id)
    q
  })
  lapply(out, function(q) {
    for (k in names(q$show_if)) {
      before <- out[[match(k, seen)]]
      if (!all(as.character(unlist(q$show_if[[k]])) %in% attr(before, "answers"))) {
        stop("show_if of question ", q$id, " uses an answer question ", k, " does not have.", call. = FALSE)
      }
    }
    attr(q, "answers") <- NULL
    q
  })
}

#' @noRd
.te_default_instructions <- function(questions) {
  ids <- vapply(questions, function(q) q$id, character(1))
  paste0(
    "Each screen shows one face, outlined in its picture, with a close-up. ",
    if ("face_check" %in% ids) "First say whether the outline really contains a face. " else "",
    "Answer with the mouse or the number keys; Enter goes to the next face and Backspace undoes the last answer."
  )
}

#' @noRd
# Which items (positions 1..n) each coder codes: a shared part for agreement,
# the rest split between coders. Uses the current random numbers.
.te_assign_items <- function(n, coders, overlap) {
  k <- length(coders)
  if (k == 1 || overlap >= 1) return(lapply(seq_len(k), function(i) sample.int(n)))
  shared <- seq_len(ceiling(overlap * n))
  rest <- setdiff(seq_len(n), shared)
  lapply(seq_len(k), function(i) {
    mine <- c(shared, rest[(seq_along(rest) - 1) %% k + 1 == i])
    mine[sample.int(length(mine))]
  })
}

#' @noRd
# Evaluates expr after set.seed(seed), then restores the user's random numbers
.te_with_seed <- function(seed, expr) {
  .te_without_rng({
    set.seed(seed)
    expr
  })
}

#' @noRd
.te_annotation_task <- function(dir, manifest, items) {
  structure(list(
    dir = dir, task_id = manifest$task_id, title = manifest$title,
    unit = if (is.null(manifest$unit)) "face" else manifest$unit,
    questions = lapply(manifest$questions, function(q) {
      q$options <- if (is.null(q$options)) NULL else as.character(unlist(q$options))
      q
    }),
    classes = as.character(unlist(manifest$classes)),
    coders = vapply(manifest$assignments, function(a) a$coder_id, character(1)),
    assignments = manifest$assignments, items = items
  ), class = "te_annotation_task")
}

#' @noRd
# An annotation_task() result, or its folder
.te_load_annotation_task <- function(task) {
  if (inherits(task, "te_annotation_task")) return(task)
  if (!is.character(task) || length(task) != 1 || !file.exists(file.path(task, "task.json"))) {
    stop("task must be an annotation_task() result or its folder.", call. = FALSE)
  }
  manifest <- jsonlite::read_json(file.path(task, "task.json"), simplifyVector = FALSE)
  if (!identical(manifest$schema, "transforEmotion.annotation_task")) {
    stop("task.json in ", task, " is not an annotation task.", call. = FALSE)
  }
  file <- file.path(task, "items.csv")
  text_cols <- c("item_id", "source", "person_id", "picture", "id", "kind", "media", "reference",
                 "prediction", "text", "video")
  header <- names(utils::read.csv(file, nrows = 0, check.names = FALSE))
  classes <- stats::setNames(rep("character", length(intersect(text_cols, header))),
                             intersect(text_cols, header))
  items <- utils::read.csv(file, stringsAsFactors = FALSE, na.strings = "", colClasses = classes,
                           check.names = FALSE)
  .te_annotation_task(normalizePath(task), manifest, items)
}

#' @noRd
# One record with only the fields the files keep
.te_flat_record <- function(r) {
  keep <- c("task_id", "item_id", "coder_id", "session_id", "seq", "revision", "status",
            "responses", "shown_at", "submitted_at", "rt_ms")
  r <- r[intersect(keep, names(r))]
  if (is.data.frame(r$responses)) r$responses <- as.list(r$responses)
  r$responses <- lapply(r$responses, function(v) if (is.null(v) || length(v) == 0) NA else v)
  r
}

#' @noRd
# Records from a .jsonl file (one per line) or an exported .json file
.te_read_record_file <- function(file) {
  if (grepl("\\.jsonl$", file)) {
    lines <- readLines(file, warn = FALSE, encoding = "UTF-8")
    lines <- lines[nzchar(trimws(lines))]
    return(lapply(lines, function(l) {
      tryCatch(jsonlite::parse_json(l, simplifyVector = FALSE), error = function(e) {
        warning("Skipped a line that is not JSON in ", basename(file), ".", call. = FALSE)
        NULL
      })
    }))
  }
  doc <- tryCatch(jsonlite::read_json(file, simplifyVector = FALSE), error = function(e) NULL)
  if (is.null(doc) || !identical(doc$schema, "transforEmotion.annotations")) {
    warning(basename(file), " is not an annotations file; left out.", call. = FALSE)
    return(list())
  }
  doc$records
}

#' @noRd
# Records from the code a coding page in a notebook gives: the exported file,
# base64-encoded
.te_read_record_code <- function(code) {
  b64 <- sub("^[[:space:]]*te-annotations:", "", code)
  if (!grepl("^[A-Za-z0-9+/=[:space:]]+$", b64)) {
    stop("The annotations code is cut off or changed; copy it again.", call. = FALSE)
  }
  text <- tryCatch(rawToChar(jsonlite::base64_dec(gsub("[[:space:]]", "", b64))),
                   error = function(e) stop("The annotations code is cut off or changed; copy it again.",
                                            call. = FALSE))
  Encoding(text) <- "UTF-8"
  doc <- tryCatch(jsonlite::parse_json(text, simplifyVector = FALSE),
                  error = function(e) stop("The annotations code is cut off or changed; copy it again.",
                                           call. = FALSE))
  if (!identical(doc$schema, "transforEmotion.annotations")) {
    stop("This code does not hold annotations.", call. = FALSE)
  }
  doc$records
}

#' @noRd
# How the copied line of code names an argument: its name, a quoted folder,
# or the default
.te_code_name <- function(expr, default) {
  if (is.name(expr)) {
    name <- as.character(expr)
    if (identical(make.names(name), name)) return(name)
  }
  if (is.character(expr) && length(expr) == 1 && !is.na(expr)) return(deparse(expr))
  default
}

#' @noRd
# annotate_faces() in a notebook: the task's page as one self-contained file,
# with smaller copies of the pictures inside it, shown below the cell
.te_annotate_in_notebook <- function(task, coder, task_code, max_side) {
  page <- file.path(task$dir, "page")
  tmp <- tempfile("te-annotate-nb-")
  dir.create(file.path(tmp, "media"), recursive = TRUE)
  file.copy(file.path(page, .te_face_app_files), tmp)
  manifest <- jsonlite::read_json(file.path(task$dir, "task.json"), simplifyVector = FALSE)
  pictures <- unique(unlist(lapply(manifest$items, function(it) c(it$media, unlist(lapply(it$frames, `[[`, "src"))))))
  pictures <- pictures[!is.na(pictures) & nzchar(pictures)]
  if (length(pictures)) {
    ensure_te_py_env()
    small <- file.path(tmp, pictures)
    .te_faces_py()$display_copies(as.list(file.path(page, pictures)), as.list(small),
                                  max_side = as.integer(max_side))
    uri <- stats::setNames(.te_data_uri(small), pictures)
    manifest$items <- lapply(manifest$items, function(it) {
      if (!is.null(it$media) && it$media %in% names(uri)) it$media <- unname(uri[it$media])
      if (!is.null(it$frames)) {
        it$frames <- lapply(it$frames, function(fr) {
          fr$src <- unname(uri[fr$src])
          fr
        })
      }
      it$video <- NULL # a clip would have to be stored in the notebook
      it
    })
  }
  json <- jsonlite::toJSON(manifest, auto_unbox = TRUE, null = "null", na = "null", digits = 4)
  notebook <- list(task = task_code)
  if (!is.null(coder)) notebook$coder <- coder
  writeLines(c(paste0("window.TE_TASK = ", json, ";"),
               paste0("window.TE_NOTEBOOK = ", jsonlite::toJSON(notebook, auto_unbox = TRUE), ";")),
             file.path(tmp, "data.js"), useBytes = TRUE)
  out <- file.path(tmp, "annotate.html")
  writeLines(.te_single_file_page(tmp), out, useBytes = TRUE)
  .te_show_in_notebook(out, task$task_id)
  message("Code below, then Copy R code. Paste the line into a new cell and run it: ",
          "read_annotations() returns the answers.")
  invisible(normalizePath(out))
}

#' @noRd
# Records (lists) as a data frame, one row each, responses in columns q_<id>
.te_records_table <- function(records, qids) {
  records <- Filter(Negate(is.null), records)
  val <- function(r, k, type) {
    v <- r[[k]]
    if (is.null(v) || length(v) == 0) return(type)
    if (is.character(type)) as.character(v[[1]]) else as.numeric(v[[1]])
  }
  out <- data.frame(
    item_id = vapply(records, val, character(1), "item_id", NA_character_),
    coder_id = vapply(records, val, character(1), "coder_id", NA_character_),
    session_id = vapply(records, val, character(1), "session_id", NA_character_),
    seq = vapply(records, val, numeric(1), "seq", NA_real_),
    revision = vapply(records, val, numeric(1), "revision", NA_real_),
    status = vapply(records, val, character(1), "status", NA_character_),
    rt_ms = vapply(records, val, numeric(1), "rt_ms", NA_real_),
    shown_at = vapply(records, val, character(1), "shown_at", NA_character_),
    submitted_at = vapply(records, val, character(1), "submitted_at", NA_character_),
    stringsAsFactors = FALSE
  )
  for (q in qids) {
    out[[paste0("q_", q)]] <- vapply(records, function(r) {
      v <- r$responses[[q]]
      if (is.null(v) || length(v) == 0 || (length(v) == 1 && is.na(v))) NA_character_ else as.character(v[[1]])
    }, character(1))
  }
  out
}

#' @noRd
# Wide records to one row per record and question
.te_long_answers <- function(wide, questions) {
  cols <- c("item_id", "coder_id", "question", "value", "value_num", "status", "revision",
            "rt_ms", "shown_at", "submitted_at", "session_id")
  if (!nrow(wide)) {
    empty <- data.frame(item_id = character(), coder_id = character(), question = character(),
                        value = character(), value_num = numeric(), status = character(),
                        revision = numeric(), rt_ms = numeric(), shown_at = character(),
                        submitted_at = character(), session_id = character(), stringsAsFactors = FALSE)
    return(empty)
  }
  parts <- lapply(questions, function(q) {
    value <- wide[[paste0("q_", q$id)]]
    data.frame(item_id = wide$item_id, coder_id = wide$coder_id, question = q$id, value = value,
               value_num = if (identical(q$type, "likert")) suppressWarnings(as.numeric(value)) else NA_real_,
               status = wide$status, revision = wide$revision, rt_ms = wide$rt_ms,
               shown_at = wide$shown_at, submitted_at = wide$submitted_at,
               session_id = wide$session_id, stringsAsFactors = FALSE)[cols]
  })
  out <- do.call(rbind, parts)
  out[order(out$item_id, out$coder_id, match(out$question, vapply(questions, function(q) q$id, ""))), , drop = FALSE]
}

#' @noRd
.te_check_annotations <- function(annotations) {
  needed <- c("item_id", "coder_id", "question", "value", "value_num", "status")
  if (!is.data.frame(annotations) || !all(needed %in% names(annotations))) {
    stop("annotations must come from read_annotations().", call. = FALSE)
  }
  invisible(TRUE)
}

#' @noRd
# Mean over items of the share of coder pairs who agree
.te_percent_agreement <- function(units) {
  if (!length(units)) return(NA_real_)
  mean(vapply(units, function(v) {
    m <- length(v)
    counts <- table(as.character(v))
    sum(counts * (counts - 1)) / (m * (m - 1))
  }, numeric(1)))
}

#' @noRd
# Krippendorff's alpha from the coincidence matrix (Krippendorff, 2011).
# units: list of the values each item got, at least two per item.
.te_kripp_alpha <- function(units, level = "nominal") {
  units <- units[lengths(units) >= 2]
  if (!length(units)) return(NA_real_)
  values <- sort(unique(unlist(units)))
  k <- length(values)
  if (k < 2) return(NA_real_) # everyone gave one answer: no variation to agree on
  o <- matrix(0, k, k)
  for (v in units) {
    idx <- match(v, values)
    m <- length(idx)
    counts <- tabulate(idx, k)
    pairs <- outer(counts, counts)
    diag(pairs) <- counts * (counts - 1)
    o <- o + pairs / (m - 1)
  }
  n_c <- rowSums(o)
  n <- sum(n_c)
  delta <- switch(level,
    nominal = 1 - diag(k),
    interval = outer(values, values, "-")^2,
    ratio = (outer(values, values, "-") / outer(values, values, "+"))^2,
    ordinal = {
      cum <- cumsum(n_c)
      d <- matrix(0, k, k)
      for (c in seq_len(k)) for (e in seq_len(k)) {
        lo <- min(c, e)
        hi <- max(c, e)
        d[c, e] <- (sum(n_c[lo:hi]) - (n_c[c] + n_c[e]) / 2)^2
      }
      d
    }
  )
  delta[is.nan(delta)] <- 0
  expected <- sum(outer(n_c, n_c) * delta)
  if (expected == 0) return(NA_real_)
  1 - (n - 1) * sum(o * delta) / expected
}

#' @noRd
# Fleiss' kappa; NA unless every item has the same number of answers
.te_fleiss_kappa <- function(units) {
  units <- units[lengths(units) >= 2]
  if (!length(units) || length(unique(lengths(units))) != 1) return(NA_real_)
  m <- lengths(units)[1]
  cats <- sort(unique(as.character(unlist(units))))
  counts <- t(vapply(units, function(v) tabulate(match(as.character(v), cats), length(cats)),
                     numeric(length(cats))))
  if (length(cats) == 1) counts <- t(counts)
  p_i <- (rowSums(counts^2) - m) / (m * (m - 1))
  p_j <- colSums(counts) / (nrow(counts) * m)
  p_e <- sum(p_j^2)
  if (p_e == 1) return(NA_real_)
  (mean(p_i) - p_e) / (1 - p_e)
}
