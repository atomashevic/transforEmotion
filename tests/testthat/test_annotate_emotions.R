# annotate_emotions(), annotation_questions() and read_annotations(). Python is
# mocked: text items need none, media items get made-up frames and faces. The
# page itself is tested by tests/js (Node).

emotions <- c("angry", "happy", "sad")

text_items <- function() {
  data.frame(id = c("c1", "c2", "c3"), text = c("He shouts.", "She laughs.", "He cries."),
             label = c("angry", "happy", "sad"), prediction = c("sad", "happy", "sad"),
             angry = c(0.2, 0.1, 0.1), happy = c(0.1, 0.8, 0.1), sad = c(0.7, 0.1, 0.8),
             stringsAsFactors = FALSE)
}

# The task annotate_emotions() wrote to data.js
read_annotation_task <- function(dir) {
  js <- grep("^window\\.TE_TASK = ", readLines(file.path(dir, "data.js"), warn = FALSE), value = TRUE)
  jsonlite::fromJSON(sub(";\\s*$", "", sub("^window\\.TE_TASK = ", "", js)), simplifyVector = FALSE)
}

# Answers as the page sends them
answers_doc <- function(task_id = "t", answers = NULL, coder = "AT") {
  if (is.null(answers)) {
    answers <- data.frame(id = c("c1", "c3"), emotion = c("angry", "sad"), why = c("model", NA),
                          note = c("loud", NA), seconds = c(12.5, 4), answered_at = "2026-10-20T10:00:00Z",
                          stringsAsFactors = FALSE)
  }
  list(schema = "transforEmotion.annotations", schema_version = "1.0.0", task_id = task_id,
       coder = coder, question_ids = c("emotion", "why", "note"), answers = answers)
}

no_python <- function(env = parent.frame()) {
  local_mocked_bindings(ensure_te_py_env = function() stop("Python is not needed for text items"),
                        .env = env)
}

test_that("annotation_questions asks why only when there is a label and a prediction", {
  q <- annotation_questions(emotions)
  expect_equal(vapply(q, `[[`, "", "id"), c("emotion", "why", "note"))
  expect_equal(unname(q[[1]]$options), c(emotions, "none of these"))
  expect_equal(names(q[[1]]$options), c(emotions, "other"))
  expect_false(q[[1]]$reveal)
  expect_true(q[[2]]$reveal)
  expect_equal(names(q[[2]]$options), c("model", "label", "face", "no_face", "agree", "other"))
  expect_null(q[[3]]$options)

  only_text <- data.frame(id = 1, text = "x")
  expect_equal(vapply(annotation_questions(emotions, only_text), `[[`, "", "id"), c("emotion", "note"))
  expect_equal(vapply(annotation_questions(emotions, why = FALSE, note = FALSE, other = NULL), `[[`, "", "id"), "emotion")
  expect_length(annotation_questions(emotions, why = FALSE, note = FALSE, other = NULL)[[1]]$options, 3)
  expect_error(annotation_questions("angry"), "at least two")
})

test_that("annotate_emotions writes a page for text items without Python", {
  no_python()
  dir <- file.path(withr::local_tempdir(), "page")
  expect_message(index <- annotate_emotions(text_items(), emotions, live = FALSE, inline = FALSE,
                                            open = FALSE, dir = dir, coder = "AT"),
                 "read_annotations")
  expect_equal(basename(index), "index.html")
  expect_setequal(list.files(dir), c("index.html", "style.css", "logic.js", "app.js", "data.js", "media"))
  task <- read_annotation_task(dir)
  expect_equal(task$schema, "transforEmotion.annotation_task")
  expect_true(task$blind)
  expect_equal(task$coder, "AT")
  expect_equal(unlist(task$classes), emotions)
  expect_equal(unlist(task$score_classes), emotions)
  expect_equal(vapply(task$questions, `[[`, "", "id"), c("emotion", "why", "note"))
  expect_equal(task$questions[[1]]$options[[4]], list(value = "other", text = "none of these"))
  expect_equal(vapply(task$items, `[[`, "", "id"), c("c1", "c2", "c3"))
  expect_equal(task$items[[1]]$text, "He shouts.")
  expect_equal(task$items[[1]]$prediction, "sad")
  expect_equal(task$items[[1]]$scores, list(angry = 0.2, happy = 0.1, sad = 0.7))
  expect_length(task$items[[1]]$frames, 0)
  expect_true(task$has$label && task$has$prediction && task$has$scores)
  expect_match(task$instructions, "Export answers")
  expect_false(any(grepl("TE_API|TE_NOTEBOOK", readLines(file.path(dir, "data.js")))))
})

test_that("annotate_emotions checks its items", {
  no_python()
  expect_error(annotate_emotions(data.frame(id = 1:2, label = "a"), emotions), "media column")
  expect_error(annotate_emotions(data.frame(id = c(1, 1), text = "x"), emotions), "unique")
  expect_error(annotate_emotions(data.frame(media = "no-such-file.mp4"), emotions), "not found")
  items <- text_items()
  items$label[1] <- "furious"
  expect_warning(annotate_emotions(items, emotions, live = FALSE, inline = FALSE, open = FALSE,
                                   dir = file.path(withr::local_tempdir(), "p")) |> suppressMessages(),
                 "not in classes: furious")
  # Items without ids are numbered
  dir <- file.path(withr::local_tempdir(), "p2")
  suppressMessages(annotate_emotions(data.frame(text = c("a", "b")), emotions, live = FALSE,
                                     inline = FALSE, open = FALSE, dir = dir))
  expect_equal(vapply(read_annotation_task(dir)$items, `[[`, "", "id"), c("1", "2"))
})

test_that("read_annotations reads files, text and the notebook code", {
  items <- text_items()
  doc <- answers_doc()
  json <- as.character(jsonlite::toJSON(doc, auto_unbox = TRUE, na = "null"))
  out <- read_annotations(json, items)
  expect_equal(out$id, c("c1", "c3"))
  expect_equal(out$coder, c("AT", "AT"))
  expect_equal(out$emotion, c("angry", "sad"))
  expect_equal(out$why, c("model", NA))
  expect_equal(out$seconds, c(12.5, 4))
  expect_equal(out$label, c("angry", "sad"))
  expect_equal(out$prediction, c("sad", "sad"))
  expect_equal(out$sad, c(0.7, 0.8))

  file <- withr::local_tempfile(fileext = ".json")
  writeLines(json, file)
  expect_equal(read_annotations(file)$emotion, c("angry", "sad"))
  expect_false("label" %in% names(read_annotations(file)))

  code <- paste0("te-annotations:", jsonlite::base64_enc(charToRaw(json)))
  expect_equal(read_annotations(code, items)$why, c("model", NA))
  expect_error(read_annotations(substr(code, 1, 30)), "cut off|answers")
  expect_error(read_annotations('{"schema": "other"}'), "not a set of answers")
  expect_error(read_annotations(file.path(tempdir(), "no-such.json")), "does not exist")
  expect_warning(read_annotations(json, items[1, ]), "1 answer\\(s\\) are for items not in items \\(c3\\)")
  # The items' media need not exist where the answers are read
  gone <- items
  gone$media <- file.path(tempdir(), "moved", paste0(gone$id, ".mp4"))
  expect_equal(read_annotations(json, gone)$label, c("angry", "sad"))

  # No answers yet
  none <- read_annotations(as.character(jsonlite::toJSON(answers_doc(answers = list()), auto_unbox = TRUE)))
  expect_equal(nrow(none), 0)
  expect_true(all(c("id", "coder", "emotion", "why", "note", "seconds") %in% names(none)))
})

test_that("the answers go into evaluate_emotions()", {
  out <- read_annotations(as.character(jsonlite::toJSON(answers_doc(), auto_unbox = TRUE, na = "null")),
                          text_items())
  coder_vs_label <- evaluate_emotions(out, truth_col = "label", pred_col = "emotion", classes = emotions,
                                      metrics = c("accuracy", "krippendorff"))
  expect_equal(coder_vs_label$accuracy, 1)
  model_vs_coder <- evaluate_emotions(out, truth_col = "emotion", pred_col = "prediction", classes = emotions,
                                      metrics = "accuracy")
  expect_equal(model_vs_coder$accuracy, 0.5)
})

test_that("annotate_emotions gets the answers from the page as the coder goes", {
  no_python()
  served <- new.env()
  local_mocked_bindings(
    .te_serve_page = function(dir, handlers = list(), port = NULL) {
      served$dir <- dir
      served$handlers <- handlers
      list(url = "http://127.0.0.1:4321/tok/", port = 4321L, stop = function() served$stopped <- TRUE)
    },
    # The browser's part: one answer, then Finish
    .te_service = function() {
      task <- read_annotation_task(served$dir)
      if (is.null(served$step)) {
        served$step <- 1
        doc <- answers_doc(task$task_id, answers = answers_doc()$answers[1, ])
        served$first <- served$handlers$answers(doc)
      } else if (served$step == 1) {
        served$step <- 2
        served$finish <- served$handlers$finish(answers_doc(task$task_id))
      }
    }
  )
  save <- withr::local_tempfile(fileext = ".json")
  expect_message(expect_message(out <- annotate_emotions(text_items(), emotions, inline = FALSE, open = FALSE,
                                                         save = save),
                                "Finish"), "1 of 3 items coded") # c3 has no answer to why
  expect_equal(served$first$saved, 1L)
  expect_equal(served$finish$saved, 2L)
  expect_true(served$stopped)
  expect_equal(out$emotion, c("angry", "sad"))
  expect_equal(out$label, c("angry", "sad"))
  # The file kept every answer, and reads back the same
  expect_equal(read_annotations(save, text_items())$emotion, out$emotion)
  js <- readLines(file.path(served$dir, "data.js"))
  expect_true(any(grepl('window.TE_API = {"answers":"api/answers","finish":"api/finish","cancel":"api/cancel"};',
                        js, fixed = TRUE)))
  expect_match(read_annotation_task(served$dir)$instructions, "Finish")

  # Answers from another call are refused; Cancel returns NULL
  served$step <- NULL
  local_mocked_bindings(.te_service = function() {
    expect_error(served$handlers$answers(answers_doc("old")), "another annotate_emotions")
    served$handlers$cancel(list())
  })
  expect_message(expect_message(none <- annotate_emotions(text_items(), emotions, inline = FALSE, open = FALSE),
                                "Finish"), "Cancelled")
  expect_null(none)
})

test_that("annotate_emotions shows frames, faces and the clip of video items", {
  root <- withr::local_tempdir()
  clip <- file.path(root, "clip.mp4")
  picture <- file.path(root, "face.jpg")
  writeLines("not really a video", clip)
  writeLines("not really an image", picture)
  calls <- new.env()
  te_faces <- list(
    extract_frames = function(path, nframes, uniform, frame_dir, video_name) {
      calls$extract <- list(nframes = nframes, uniform = uniform)
      lapply(seq_len(nframes), function(k) {
        p <- file.path(frame_dir, sprintf("%s-%d.jpg", video_name, k))
        file.create(p)
        list(path = p, time = k - 0.5, frame_index = k * 10L)
      })
    },
    display_copies = function(images, out_paths, max_side) {
      calls$max_side <- max_side
      file.create(unlist(out_paths))
      lapply(images, function(i) list(640L, 360L))
    }
  )
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    .te_faces_py = function() te_faces,
    .te_item_faces = function(te_faces, paths, frames) {
      data.frame(source = paths[c(1, 1, 2)], x = c(10, 300, 20), y = 5, width = c(50, 120, 60),
                 height = c(50, 120, 60), scored = c(FALSE, TRUE, TRUE))
    }
  )
  items <- data.frame(id = c("v", "p"), media = c(clip, picture), text = c("A clip.", NA),
                      stringsAsFactors = FALSE)
  dir <- file.path(root, "page")
  suppressMessages(annotate_emotions(items, emotions, frames = 3, live = FALSE, inline = FALSE,
                                     open = FALSE, dir = dir))
  task <- read_annotation_task(dir)
  v <- task$items[[1]]
  expect_equal(v$media_type, "video")
  expect_equal(v$label_file, "clip.mp4")
  expect_length(v$frames, 3)
  expect_equal(calls$extract, list(nframes = 3L, uniform = TRUE))
  expect_equal(v$frames[[1]]$src, "media/i0001-01.jpg")
  expect_equal(v$frames[[1]]$time, 0.5)
  expect_equal(c(v$frames[[1]]$w, v$frames[[1]]$h), c(640, 360))
  expect_equal(vapply(v$frames[[1]]$faces, `[[`, TRUE, "scored"), c(FALSE, TRUE))
  expect_equal(v$video, "media/i0001.mp4")
  expect_true(file.exists(file.path(dir, "media", "i0001.mp4")))
  expect_equal(calls$max_side, 960L)
  p <- task$items[[2]]
  expect_equal(p$media_type, "image")
  expect_length(p$frames, 1)
  expect_null(p$video)
  expect_false(task$has$label)
  # Without label or prediction there is no "why" question
  expect_equal(vapply(task$questions, `[[`, "", "id"), c("emotion", "note"))

  # In a notebook: one file, images inside it, no clip
  shown <- new.env()
  local_mocked_bindings(.te_display_html = function(html, page) shown$html <- html)
  dir2 <- file.path(root, "page2")
  page <- suppressMessages(annotate_emotions(items, emotions, inline = TRUE, dir = dir2))
  expect_equal(basename(page), "annotate.html")
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  expect_false(grepl("<script src=|src=\"media/|\"media/i0001.mp4\"", html))
  expect_match(html, "data:image/jpeg;base64,", fixed = TRUE)
  expect_match(html, "window.TE_NOTEBOOK = {\"items\":\"items\"};", fixed = TRUE)
  expect_equal(calls$max_side, 480L)
  expect_match(shown$html, "<iframe id=\"te-annotate-", fixed = TRUE)
})
