# annotation_task(items = ...): coding tasks whose items are video clips,
# images or texts rather than single faces, for example a model's errors to
# review. Same folder, coders, answers and agreement as face tasks; see
# R/annotation.R.

# Video files: frames are sampled from them; browsers play the second set
.te_video_ext <- "\\.(mp4|m4v|mov|webm|mkv|avi|flv|mpg|mpeg|wmv)$"
.te_playable_ext <- "\\.(mp4|m4v|webm|mov)$"

#' @noRd
# Checks the items of an item task: media and/or text, unique ids, and the
# label and prediction against the classes
.te_check_review_items <- function(items, classes) {
  if (!is.data.frame(items) || !nrow(items)) stop("items must be a data frame with rows.", call. = FALSE)
  items <- as.data.frame(items, stringsAsFactors = FALSE)
  if (!any(c("media", "text") %in% names(items))) {
    stop("items needs a media column (video or image paths) or a text column.", call. = FALSE)
  }
  if (!"id" %in% names(items)) items$id <- seq_len(nrow(items))
  items$id <- as.character(items$id)
  if (anyNA(items$id) || anyDuplicated(items$id)) stop("items$id must be unique and not NA.", call. = FALSE)
  if ("reference" %in% names(items) && !"label" %in% names(items)) items$label <- items$reference
  for (col in intersect(c("media", "text", "label", "prediction"), names(items))) {
    items[[col]] <- as.character(items[[col]])
  }
  if ("media" %in% names(items)) {
    given <- !is.na(items$media) & nzchar(items$media)
    missing <- items$media[given & !file.exists(items$media)]
    if (length(missing)) {
      stop("Media file(s) not found: ", paste(utils::head(missing, 3), collapse = ", "),
           if (length(missing) > 3) ", ...", call. = FALSE)
    }
  }
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
# Default questions for items: the emotion, and why, when there is a label and
# a prediction to compare with
.te_review_questions <- function(items) {
  annotation_questions(if (all(c("label", "prediction") %in% names(items))) c("emotion", "why") else "emotion")
}

#' @noRd
# The items' columns that read_annotations() adds to every answer
.te_item_columns <- function(task) {
  if (identical(task$unit, "item")) {
    return(setdiff(names(task$items), c("item_id", "text", "n_frames", "video")))
  }
  c("source", "frame", "frame_index", "time", "face", "track_id", "person_id")
}

#' @noRd
# annotation_task() for items: writes page/, task.json, items.csv and
# results/ as for faces, with frames, face boxes, clips, texts, labels and
# scores in the page
.te_item_annotation_task <- function(items, classes, questions, coders, overlap, dir, seed, title,
                                     instructions, max_side, task_id, frames, play, blind) {
  if (!is.numeric(frames) || length(frames) != 1 || frames < 1) stop("frames must be at least 1.", call. = FALSE)
  score_cols <- intersect(as.character(classes), names(items))
  score_cols <- score_cols[vapply(items[score_cols], is.numeric, logical(1))]
  plan <- .te_with_seed(seed, {
    order <- sample.int(nrow(items))
    list(order = order, assignments = .te_assign_items(length(order), coders, overlap))
  })
  items <- items[plan$order, , drop = FALSE]
  n <- nrow(items)
  item_ids <- sprintf("i%0*d", max(4, nchar(n)), seq_len(n))

  page <- file.path(dir, "page")
  dir.create(file.path(page, "media"), recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(dir, "results"), showWarnings = FALSE)
  has_media <- "media" %in% names(items) && any(!is.na(items$media) & nzchar(items$media))
  te_faces <- NULL
  if (has_media) {
    ensure_te_py_env()
    te_faces <- .te_faces_py()
  }
  col <- function(name, i) if (name %in% names(items) && !is.na(items[[name]][i])) items[[name]][i] else NULL

  entries <- lapply(seq_len(n), function(i) {
    path <- col("media", i)
    entry <- list(item_id = item_ids[i], kind = "text", text = col("text", i),
                  reference = col("label", i), prediction = col("prediction", i),
                  scores = if (length(score_cols)) as.list(stats::setNames(as.numeric(items[i, score_cols]), score_cols)) else NULL,
                  frames = list(), video = NULL)
    if (is.null(path)) return(entry)
    key <- sprintf("c%0*d", max(4, nchar(n)), i)
    is_video <- grepl(.te_video_ext, path, ignore.case = TRUE)
    entry$kind <- if (is_video) "clip" else "image"
    tmp <- tempfile("te-item-")
    dir.create(tmp)
    on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
    if (is_video) {
      fr <- te_faces$extract_frames(path, as.integer(frames), uniform = TRUE, frame_dir = tmp, video_name = key)
      paths <- vapply(fr, function(f) f$path, character(1))
      times <- vapply(fr, function(f) as.numeric(f$time), numeric(1))
    } else {
      fr <- NULL
      paths <- path
      times <- NA_real_
    }
    files <- file.path("media", sprintf("%s-%02d.jpg", key, seq_along(paths)))
    sizes <- te_faces$display_copies(as.list(paths), as.list(file.path(page, files)),
                                     max_side = as.integer(max_side))
    boxes <- .te_item_faces(te_faces, paths, fr)
    entry$frames <- lapply(seq_along(paths), function(k) {
      b <- if (is.null(boxes)) NULL else boxes[boxes$source == paths[k], , drop = FALSE]
      list(src = files[k], width = as.integer(sizes[[k]][[1]]), height = as.integer(sizes[[k]][[2]]),
           time = if (is.na(times[k])) NULL else times[k],
           faces = if (is.null(b) || !nrow(b)) list() else lapply(seq_len(nrow(b)), function(j)
             list(x = b$x[j], y = b$y[j], w = b$width[j], h = b$height[j], scored = isTRUE(b$scored[j]))))
    })
    if (is_video && isTRUE(play) && grepl(.te_playable_ext, path, ignore.case = TRUE)) {
      clip <- file.path("media", paste0(key, ".", tolower(tools::file_ext(path))))
      file.copy(path, file.path(page, clip))
      entry$video <- clip
    }
    entry
  })

  key_table <- data.frame(item_id = item_ids, id = items$id,
                          kind = vapply(entries, `[[`, "", "kind"),
                          media = if ("media" %in% names(items)) items$media else NA_character_,
                          reference = if ("label" %in% names(items)) items$label else NA_character_,
                          prediction = if ("prediction" %in% names(items)) items$prediction else NA_character_,
                          stringsAsFactors = FALSE)
  for (cl in score_cols) key_table[[cl]] <- items[[cl]]
  key_table$n_frames <- vapply(entries, function(e) length(e$frames), integer(1))
  key_table$video <- vapply(entries, function(e) if (is.null(e$video)) NA_character_ else e$video, "")
  key_table$text <- if ("text" %in% names(items)) items$text else NA_character_

  has_reference <- any(!is.na(key_table$reference)) || any(!is.na(key_table$prediction)) || length(score_cols) > 0
  assignments <- lapply(seq_along(coders), function(k) {
    list(coder_id = coders[k], items = I(item_ids[plan$assignments[[k]]]))
  })
  manifest <- list(
    schema = "transforEmotion.annotation_task", schema_version = "1.0.0",
    task_id = task_id, mode = "annotate", unit = "item",
    title = if (is.null(title)) "Annotation" else as.character(title),
    instructions = if (is.null(instructions)) .te_item_instructions(questions, blind, has_reference) else as.character(instructions),
    classes = I(as.character(classes)), score_classes = I(score_cols),
    questions = questions,
    settings = list(allow_back = TRUE, allow_skip = TRUE, show_progress = TRUE, blind = isTRUE(blind)),
    items = entries,
    assignments = assignments
  )
  json <- jsonlite::toJSON(manifest, auto_unbox = TRUE, null = "null", na = "null",
                           digits = 4, pretty = TRUE)
  writeLines(json, file.path(dir, "task.json"), useBytes = TRUE)
  writeLines(paste0("window.TE_TASK = ", json, ";"), file.path(page, "data.js"), useBytes = TRUE)
  file.copy(system.file("www", "face-annotator", .te_face_app_files, package = "transforEmotion",
                        mustWork = TRUE), page, overwrite = TRUE)
  utils::write.csv(key_table, file.path(dir, "items.csv"), row.names = FALSE, na = "")

  task <- .te_annotation_task(normalizePath(dir), manifest, key_table)
  message("The task is in ", task$dir, ": ", n, " items for ", length(coders), " coder(s). ",
          "Start coding with annotate_faces().")
  invisible(task)
}

#' @noRd
.te_item_instructions <- function(questions, blind, has_reference) {
  ids <- vapply(questions, function(q) q$id, character(1))
  paste0(
    "Each screen shows one item: frames of a clip with the faces found in them (the solid box is the face ",
    "the model scores), the clip itself when it can play here, and its text. ",
    if (has_reference && isTRUE(blind)) "Give your own answer first; then the label and the model's answer are shown. " else "",
    if ("why" %in% ids) "Then say what explains the result. " else "",
    "Answer with the mouse or the number keys; Enter goes to the next item and Backspace undoes the last answer."
  )
}

#' @noRd
# Face boxes in the images or frames, and which one is scored by default
# (the largest); NULL when there are none
.te_item_faces <- function(te_faces, paths, frames) {
  res <- te_faces$faces_table(as.list(paths), detector = "yunet", min_confidence = 0.6,
                              yunet_model = .te_yunet_model(), frames = frames)
  b <- .te_faces_frame(res, paths, "yunet", list(type = "images"))
  if (!nrow(b)) return(NULL)
  area <- b$width * b$height
  b$scored <- stats::ave(area, b$source, FUN = function(a) seq_along(a) == which.max(a)) == 1
  b
}
