# annotation_task(), annotate_faces(), read_annotations(), annotation_progress(),
# annotation_agreement() and consensus_labels(). Python is mocked (see
# helper-faces.R for the made-up faces); the coding page's logic is tested by
# tests/js.

# Mocks the Python display copies: writes empty files and reports a size
mock_annotation_python <- function(env = parent.frame()) {
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    .te_faces_py = function() list(display_copies = function(images, out_paths, max_side) {
      file.create(unlist(out_paths))
      lapply(images, function(i) list(400L, 300L))
    }),
    .env = env
  )
}

make_task <- function(dir = file.path(withr::local_tempdir(.local_envir = parent.frame()), "task"), ...) {
  suppressMessages(annotation_task(fake_faces(), classes = c("happy", "sad", "neutral"),
                                   dir = dir, task_id = "t1", ...))
}

# A record as the coding page sends it
rec <- function(item, coder = "C01", emotion = "happy", status = "answered", seq = 1,
                revision = 1, session = "s1", at = "2026-10-11T10:00:00.000Z", task = "t1",
                face_check = if (is.na(emotion)) "not_face" else "ok", intensity = NA) {
  list(task_id = task, item_id = item, coder_id = coder, session_id = session, seq = seq,
       revision = revision, status = status,
       responses = list(face_check = if (status == "answered") face_check else NULL,
                        emotion = if (is.na(emotion) || status != "answered") NULL else emotion,
                        intensity = if (is.na(intensity)) NULL else intensity),
       shown_at = at, submitted_at = at, rt_ms = 1500)
}

write_jsonl <- function(records, file) {
  lines <- vapply(records, function(r) {
    as.character(jsonlite::toJSON(r, auto_unbox = TRUE, null = "null", na = "null"))
  }, character(1))
  cat(paste0(lines, "\n"), file = file, sep = "", append = TRUE)
}

test_that("ready-made questions ask the rest only for faces", {
  qs <- annotation_questions()
  expect_equal(vapply(qs, `[[`, "", "id"), c("face_check", "emotion", "intensity"))
  expect_equal(qs[[2]]$show_if, list(face_check = "ok"))
  # Without face_check, every question is asked
  expect_null(annotation_questions(c("emotion", "valence"))[[2]]$show_if)
  expect_error(annotation_questions("mood"), "Unknown questions: mood")
})

test_that("questions are checked before a task is made", {
  check <- function(q, classes = c("a", "b")) .te_check_questions(q, classes)
  expect_error(check(list(list(id = "Bad id", type = "single", options = "x"))), "lower case")
  expect_error(check(list(list(id = "q", type = "slider"))), "type")
  expect_error(check(list(list(id = "q", type = "single"))), "needs options")
  expect_error(check(list(list(id = "q", type = "likert", min = 1, max = 20))), "at most 10 apart")
  expect_error(check(list(list(id = "q", type = "single", options = "x", show_if = list(z = "x")))),
               "asked before")
  expect_error(check(annotation_questions(), classes = "happy"), "at least two")
  expect_error(check(c(annotation_questions("face_check"),
                       list(list(id = "e", type = "single", options = "x", show_if = list(face_check = "yes"))))),
               "does not have")
  one <- check(list(list(id = "gaze", type = "single", options = "camera")))
  expect_s3_class(one[[1]]$options, "AsIs") # stays a JSON array
  expect_true(one[[1]]$required)
  expect_equal(one[[1]]$prompt, "gaze")
})

test_that("annotation_task writes a blinded page, the item key and the plan", {
  mock_annotation_python()
  set.seed(7)
  before <- stats::runif(1)
  set.seed(7)
  task <- make_task(coders = c("C01", "C02"))
  expect_equal(stats::runif(1), before) # the user's random numbers are untouched
  expect_s3_class(task, "te_annotation_task")
  expect_output(print(task), "3 items; coders: C01, C02")
  files <- list.files(task$dir, recursive = TRUE)
  expect_true(all(c("task.json", "items.csv", "page/index.html", "page/app.js", "page/data.js") %in% files))
  expect_true(dir.exists(file.path(task$dir, "results")))
  expect_equal(length(grep("^page/media/p\\d+\\.jpg$", files)), 2) # one per picture

  # The page knows nothing about the files
  data_js <- paste(readLines(file.path(task$dir, "page", "data.js")), collapse = "\n")
  expect_false(grepl("a\\.jpg|b\\.jpg|person_id|track_id", data_js))
  expect_false(grepl("TE_API", data_js))
  manifest <- jsonlite::read_json(file.path(task$dir, "task.json"), simplifyVector = TRUE)
  expect_equal(manifest$schema, "transforEmotion.annotation_task")
  expect_equal(manifest$task_id, "t1")
  expect_equal(manifest$questions$id, c("face_check", "emotion", "intensity"))
  expect_equal(unlist(manifest$classes), c("happy", "sad", "neutral"))
  expect_equal(manifest$items$item_id, c("i0001", "i0002", "i0003"))
  # Everyone codes everything, in their own order
  orders <- manifest$assignments$items
  expect_setequal(orders[[1]], manifest$items$item_id)
  expect_setequal(orders[[2]], manifest$items$item_id)

  # items.csv maps items back to faces
  items <- utils::read.csv(file.path(task$dir, "items.csv"), stringsAsFactors = FALSE)
  key <- paste(items$source, items$face)
  expect_setequal(key, c("a.jpg 1", "a.jpg 2", "b.jpg 1"))
  boxes <- manifest$items$box[match(items$item_id, manifest$items$item_id), ]
  expect_equal(boxes$w, items$width)

  # The same seed gives the same plan
  again <- make_task(coders = c("C01", "C02"))
  expect_identical(again$items, task$items)
  expect_identical(again$assignments, task$assignments)
  expect_error(make_task(dir = task$dir), "new or empty folder")
  expect_error(suppressMessages(annotation_task(fake_faces(), classes = c("a", "b"))), "folder for the task")
  expect_error(make_task(coders = c("C01", "C01")), "different ids")
  expect_error(make_task(overlap = 0), "share above 0")
})

test_that("overlap gives every coder the shared items and splits the rest", {
  set.seed(1)
  plan <- .te_assign_items(10, c("A", "B"), overlap = 0.2)
  expect_true(all(1:2 %in% plan[[1]]) && all(1:2 %in% plan[[2]]))
  expect_equal(sort(c(plan[[1]], plan[[2]])), sort(c(1:2, 1:10)))
  expect_length(intersect(plan[[1]], plan[[2]]), 2)
})

test_that("annotate_faces saves each coder's answers in R", {
  mock_annotation_python()
  task <- make_task(coders = c("C01", "C02"))
  served <- new.env()
  local_mocked_bindings(.te_serve_page = function(dir, handlers = list(), port = NULL, files = list()) {
    served$dir <- dir
    served$handlers <- handlers
    served$files <- files
    list(url = "http://127.0.0.1:4321/tok/", port = 4321L, stop = function() served$stopped <- TRUE)
  })
  expect_message(page <- annotate_faces(task, coder = "C02", open = FALSE), "Answers are saved")
  expect_equal(page$url, "http://127.0.0.1:4321/tok/?coder=C02")
  expect_match(paste(readLines(served$files[["data.js"]]), collapse = "\n"),
               'window.TE_API = {"save":"api/save","load":"api/load"};', fixed = TRUE)
  expect_false(attr(served$handlers$save, "simplify")) # gets the records as lists

  items <- task$items$item_id
  save <- served$handlers$save
  expect_equal(save(list(records = list(rec(items[1]), rec(items[2], seq = 2))))$saved, 2L)
  expect_equal(save(list(records = list(rec(items[1], coder = "C02", emotion = "sad"))))$saved, 1L)
  expect_true(all(file.exists(file.path(task$dir, "results", c("C01.jsonl", "C02.jsonl")))))
  expect_error(save(list(records = list(rec(items[1], coder = "../x")))), "Unknown coder")
  expect_error(save(list(records = list(rec(items[1], task = "other")))), "another task")
  expect_error(save(list(records = list(rec("i9999")))), "not on this coder's list")
  expect_error(save(list(records = list(rec(items[1], status = "maybe")))), "status")
  expect_error(save(list()), "No records")

  loaded <- served$handlers$load(list(coder_id = "C01"))
  back <- jsonlite::fromJSON(unclass(loaded$records), simplifyVector = FALSE)
  expect_length(back, 2)
  expect_equal(back[[1]]$responses$emotion, "happy")
  expect_null(back[[1]]$responses$intensity) # not answered: null
  expect_equal(served$handlers$load(list(coder_id = "C02"))$ok, TRUE)
  expect_error(served$handlers$load(list(coder_id = "nobody")), "Unknown coder")

  page$stop()
  expect_true(served$stopped)
  expect_false(file.exists(served$files[["data.js"]]))
  expect_error(annotate_faces(task, coder = "C09", open = FALSE), "one of the task's coders")
})

test_that("the task and the saved answers follow the schema file", {
  mock_annotation_python()
  task <- make_task()
  schema <- jsonlite::read_json(system.file("schema", "face-task-v1.json", package = "transforEmotion"))
  has_required <- function(x, def) {
    expect_true(all(unlist(schema$`$defs`[[def]]$required) %in% names(x)), label = def)
  }
  manifest <- jsonlite::read_json(file.path(task$dir, "task.json"))
  has_required(manifest, "annotation_task")
  expect_equal(manifest$schema, schema$`$defs`$annotation_task$properties$schema$const)
  item_def <- schema$`$defs`$face_item
  expect_length(item_def$required, 6)
  expect_true(all(unlist(item_def$required) %in% names(manifest$items[[1]])))
  for (q in manifest$questions) has_required(q, "question")

  served <- new.env()
  local_mocked_bindings(.te_serve_page = function(dir, handlers = list(), port = NULL, files = list()) {
    served$handlers <- handlers
    list(url = "http://127.0.0.1:4321/tok/", port = 4321L, stop = function() NULL)
  })
  suppressMessages(annotate_faces(task, open = FALSE))
  served$handlers$save(list(records = list(rec(task$items$item_id[1]))))
  line <- jsonlite::parse_json(readLines(file.path(task$dir, "results", "C01.jsonl"))[1])
  has_required(line, "record")
  expect_named(line$responses, vapply(manifest$questions, `[[`, "", "id"), ignore.order = TRUE)
})

test_that("annotate_faces serves the coding page from R", {
  skip_on_cran() # opens a local port
  mock_annotation_python()
  task <- make_task()
  page <- suppressMessages(annotate_faces(task$dir, coder = "C01", open = FALSE))
  on.exit(page$stop(), add = TRUE)
  get <- function(path) httr::GET(paste0(sub("\\?.*$", "", page$url), path), httr::timeout(5))
  expect_equal(httr::status_code(httr::GET(page$url, httr::timeout(5))), 200L) # with ?coder=
  data <- httr::content(get("data.js"), as = "text", encoding = "UTF-8")
  expect_match(data, "TE_API")
  # The task's own data.js, for copies of the folder, has no server in it
  expect_false(any(grepl("TE_API", readLines(file.path(task$dir, "page", "data.js")))))
  expect_equal(httr::status_code(get("media/p001.jpg")), 200L)
})

test_that("read_annotations keeps each coder's last answer", {
  mock_annotation_python()
  task <- make_task(coders = c("C01", "C02"))
  items <- task$items$item_id
  results <- file.path(task$dir, "results")
  write_jsonl(list(
    rec(items[1], emotion = "sad", seq = 1, at = "2026-10-11T10:00:01Z"),
    rec(items[2], emotion = NA, seq = 2, at = "2026-10-11T10:00:02Z"), # not a face
    rec(items[3], status = "skipped", seq = 3, at = "2026-10-11T10:00:03Z"),
    rec(items[1], emotion = "happy", seq = 4, revision = 2, at = "2026-10-11T10:00:04Z", intensity = 5)
  ), file.path(results, "C01.jsonl"))
  # An export from the other coder, with a record from another task
  export <- list(schema = "transforEmotion.annotations", schema_version = "1.0.0", task_id = "t1",
                 coder_id = "C02", records = list(rec(items[1], coder = "C02", emotion = "happy"),
                                                  rec(items[2], coder = "C02", task = "t0")))
  jsonlite::write_json(export, file.path(results, "annotations-t1-C02.json"), auto_unbox = TRUE,
                       null = "null", na = "null")
  # The same record twice (in the .jsonl and an export) counts once
  write_jsonl(list(rec(items[1], coder = "C02", emotion = "happy")), file.path(results, "C02.jsonl"))

  expect_warning(ann <- read_annotations(task$dir), "another task")
  expect_equal(nrow(ann), 4 * 3) # C01: three items, C02: one; three questions each
  expect_equal(names(ann)[1:6], c("item_id", "coder_id", "question", "value", "value_num", "status"))
  mine <- ann[ann$coder_id == "C01" & ann$item_id == items[1], ]
  expect_equal(mine$value, c("ok", "happy", "5")) # the second revision
  expect_equal(mine$value_num, c(NA, NA, 5))
  expect_equal(mine$revision, c(2, 2, 2))
  not_face <- ann[ann$coder_id == "C01" & ann$item_id == items[2], ]
  expect_equal(not_face$value, c("not_face", NA, NA))
  skipped <- ann[ann$coder_id == "C01" & ann$item_id == items[3], ]
  expect_true(all(skipped$status == "skipped") && all(is.na(skipped$value)))
  # The item's face comes along
  expect_equal(unique(mine$source), task$items$source[1])
  expect_true(all(c("frame_index", "face", "person_id") %in% names(ann)))

  all_revisions <- suppressWarnings(read_annotations(task, revisions = "all"))
  expect_equal(nrow(all_revisions), 5 * 3)

  progress <- suppressWarnings(annotation_progress(task))
  expect_equal(progress$coder_id, c("C01", "C02"))
  expect_equal(progress$answered, c(2L, 1L))
  expect_equal(progress$skipped, c(1L, 0L))
  expect_equal(progress$left, c(0L, 2L))
  expect_equal(progress$median_seconds, c(1.5, 1.5))

  expect_error(read_annotations(withr::local_tempdir()), "annotation_task")
  expect_error(read_annotations(task, files = "nope.jsonl"), "does not exist")
})

test_that("Krippendorff's alpha matches the published example", {
  # Krippendorff (2011), 4 coders x 12 units; unit 12 has one value only
  m <- matrix(c(1, 1, NA, 1, 2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 2, 3, 4, 4, 4, 4, 4,
                1, 1, 2, 1, 2, 2, 2, 2, NA, 5, 5, 5, NA, NA, 1, 1, NA, NA, 3, NA), nrow = 4)
  units <- lapply(seq_len(ncol(m)), function(j) stats::na.omit(m[, j]))
  expect_equal(round(.te_kripp_alpha(units, "nominal"), 3), 0.743)
  expect_equal(round(.te_kripp_alpha(units, "ordinal"), 3), 0.815)
  expect_equal(round(.te_kripp_alpha(units, "interval"), 3), 0.849)
  expect_equal(round(.te_kripp_alpha(units, "ratio"), 3), 0.797)
  expect_equal(.te_kripp_alpha(list(c("a", "a"), c("b", "b")), "nominal"), 1)
  expect_true(is.na(.te_kripp_alpha(list(c("a", "a")), "nominal"))) # nothing varies
})

test_that("Fleiss' kappa matches the textbook example", {
  # Fleiss (1971) as reworked on Wikipedia: 10 subjects, 14 raters, 5 categories
  counts <- rbind(c(0, 0, 0, 0, 14), c(0, 2, 6, 4, 2), c(0, 0, 3, 5, 6), c(0, 3, 9, 2, 0),
                  c(2, 2, 8, 1, 1), c(7, 7, 0, 0, 0), c(3, 2, 6, 3, 0), c(2, 5, 3, 2, 2),
                  c(6, 5, 2, 1, 0), c(0, 2, 2, 3, 7))
  units <- lapply(seq_len(nrow(counts)), function(i) rep(letters[1:5], counts[i, ]))
  expect_equal(round(.te_fleiss_kappa(units), 3), 0.210)
  expect_true(is.na(.te_fleiss_kappa(list(c("a", "b"), c("a", "a", "b")))))
  expect_equal(.te_percent_agreement(list(c("a", "a", "b"), c("a", "a"))), (1 / 3 + 1) / 2)
})

# A made-up read_annotations() result: 4 items, 3 coders, emotion and intensity
fake_annotations <- function() {
  emotion <- c("happy", "happy", "happy",   # i1: all agree
               "sad", "sad", "happy",       # i2: two of three
               "sad", "happy", NA,          # i3: a tie (the third saw no face)
               "neutral", "neutral", "neutral")
  intensity <- c(5, 6, 7, 2, 2, 3, 4, 4, NA, 1, 1, 2)
  base <- data.frame(item_id = rep(sprintf("i%d", 1:4), each = 3), coder_id = rep(c("A", "B", "C"), 4),
                     status = "answered", stringsAsFactors = FALSE)
  out <- rbind(
    cbind(base, question = "emotion", value = emotion, value_num = NA_real_),
    cbind(base, question = "intensity", value = as.character(intensity), value_num = intensity)
  )
  out$source <- paste0(out$item_id, ".jpg")
  attr(out, "questions") <- list(list(id = "emotion", type = "single"),
                                 list(id = "intensity", type = "likert"))
  out
}

test_that("annotation_agreement reports alpha, kappa and percent agreement", {
  ann <- fake_annotations()
  agree <- annotation_agreement(ann)
  expect_equal(agree$question, c("emotion", "intensity"))
  expect_equal(agree$level, c("nominal", "interval"))
  expect_equal(agree$n_items, c(4, 4))
  expect_equal(agree$n_answers, c(11, 11))
  units <- split(ann$value[ann$question == "emotion" & !is.na(ann$value)],
                 ann$item_id[ann$question == "emotion" & !is.na(ann$value)])
  expect_equal(agree$alpha[1], .te_kripp_alpha(units, "nominal"))
  expect_true(is.na(agree$fleiss_kappa[1])) # item 3 has two answers, the others three
  expect_equal(agree$percent_agreement[1], (1 + 1 / 3 + 0 + 1) / 4)
  expect_equal(annotation_agreement(ann, "intensity", level = c(intensity = "ordinal"))$level, "ordinal")
  expect_error(annotation_agreement(data.frame(x = 1)), "read_annotations")
  # Fully crossed: kappa is there
  full <- ann[ann$item_id != "i3", ]
  expect_false(is.na(annotation_agreement(full, "emotion")$fleiss_kappa))
})

test_that("consensus_labels gives one label per item", {
  ann <- fake_annotations()
  truth <- consensus_labels(ann)
  expect_equal(truth$item_id, sprintf("i%d", 1:4))
  expect_equal(truth$label, c("happy", "sad", NA, "neutral"))
  expect_equal(truth$tie, c(FALSE, FALSE, TRUE, FALSE))
  expect_equal(truth$n_coders, c(3, 3, 2, 3))
  expect_equal(truth$agreement, c(1, 2 / 3, 1 / 2, 1))
  expect_equal(truth$source, sprintf("i%d.jpg", 1:4))
  expect_equal(consensus_labels(ann, min_coders = 3)$label, c("happy", "sad", NA, "neutral"))
  expect_equal(consensus_labels(ann, "intensity", method = "mean")$label, c(6, 7 / 3, 4, 4 / 3))
  expect_equal(consensus_labels(ann, "intensity", method = "median")$label, c(6, 2, 4, 1))
  expect_error(consensus_labels(ann, "valence"), "no question valence")
})

test_that("annotate_faces in a notebook shows the coding page below the cell", {
  mock_annotation_python()
  task <- make_task(coders = c("C01", "C02"))
  shown <- new.env()
  local_mocked_bindings(.te_display_html = function(html, page) {
    shown$html <- html
    invisible(TRUE)
  })
  expect_message(page <- annotate_faces(task, coder = "C02", inline = TRUE), "Copy R code")
  expect_equal(basename(page), "annotate.html")
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  # One file: no scripts, styles or pictures beside it, and no server
  expect_false(grepl("<script src=|<link rel=\"stylesheet\"|\"media/p0", html))
  expect_match(html, "data:image/jpeg;base64,", fixed = TRUE)
  expect_match(html, "window.TE_NOTEBOOK = {\"task\":\"task\",\"coder\":\"C02\"};", fixed = TRUE)
  expect_false(grepl("window.TE_API = ", html, fixed = TRUE))
  expect_match(shown$html, "<iframe id=\"te-t1-", fixed = TRUE)
  # The task's own page is left as it was
  expect_match(paste(readLines(file.path(task$dir, "page", "data.js")), collapse = ""), "media/p0")

  # A task given as its folder, written out, is named by the folder in the copied line
  page <- suppressMessages(eval(bquote(annotate_faces(.(task$dir), inline = TRUE))))
  expect_match(paste(readLines(page, warn = FALSE), collapse = "\n"),
               paste0("window.TE_NOTEBOOK = ", jsonlite::toJSON(list(task = deparse(task$dir)), auto_unbox = TRUE), ";"),
               fixed = TRUE)
  expect_equal(.te_code_name(quote(round1), "task"), "round1")
  expect_equal(.te_code_name("round-1", "task"), "\"round-1\"")
  expect_equal(.te_code_name(quote(tasks[[1]]), "task"), "task")
})

test_that("read_annotations reads the code a notebook coding page gives", {
  mock_annotation_python()
  task <- make_task()
  doc <- list(schema = "transforEmotion.annotations", schema_version = "1.0.0", task_id = "t1",
              coder_id = "C01", records = list(rec(task$items$item_id[1]),
                                               rec(task$items$item_id[2], emotion = "sad", seq = 2)))
  json <- as.character(jsonlite::toJSON(doc, auto_unbox = TRUE, null = "null", na = "null"))
  code <- paste0("te-annotations:", jsonlite::base64_enc(charToRaw(json)))
  ann <- read_annotations(task, code)
  emotion <- ann[ann$question == "emotion", ]
  expect_equal(emotion$value[match(task$items$item_id[1:2], emotion$item_id)], c("happy", "sad"))
  expect_error(read_annotations(task, substr(code, 1, 40)), "cut off")
  other <- paste0("te-annotations:", jsonlite::base64_enc(charToRaw('{"schema": "x"}')))
  expect_error(read_annotations(task, other), "does not hold annotations")
})


# ---------------------------------------------------------------------------
# Item tasks: clips, images and texts, for example a model's errors
# ---------------------------------------------------------------------------

review_classes <- c("angry", "happy", "sad")
text_items <- function() {
  data.frame(id = c("c1", "c2", "c3"), text = c("He shouts.", "She laughs.", "He cries."),
             label = c("angry", "happy", "sad"), prediction = c("sad", "happy", "sad"),
             angry = c(0.2, 0.1, 0.1), happy = c(0.1, 0.8, 0.1), sad = c(0.7, 0.1, 0.8),
             stringsAsFactors = FALSE)
}
make_review <- function(items = text_items(), ...) {
  dir <- file.path(withr::local_tempdir(.local_envir = parent.frame()), "review")
  suppressMessages(annotation_task(items = items, classes = review_classes, dir = dir, task_id = "r1", ...))
}

test_that("annotation_task(items = ...) writes an item task without Python for texts", {
  local_mocked_bindings(ensure_te_py_env = function() stop("no Python needed for texts"))
  task <- make_review()
  expect_equal(task$unit, "item")
  manifest <- jsonlite::read_json(file.path(task$dir, "task.json"))
  expect_equal(manifest$unit, "item")
  expect_true(manifest$settings$blind)
  expect_equal(unlist(manifest$score_classes), review_classes)
  expect_equal(vapply(manifest$questions, `[[`, "", "id"), c("emotion", "why"))
  expect_true(manifest$questions[[2]]$reveal)
  expect_null(manifest$questions[[1]]$reveal)
  it <- manifest$items[[match("c1", task$items$id[match(vapply(manifest$items, `[[`, "", "item_id"), task$items$item_id)])]]
  expect_equal(it$kind, "text")
  expect_equal(it$text, "He shouts.")
  expect_equal(it$reference, "angry")
  expect_equal(it$prediction, "sad")
  expect_equal(it$scores, list(angry = 0.2, happy = 0.1, sad = 0.7))
  # The key: the user's id, label as reference, prediction and scores
  expect_setequal(task$items$id, c("c1", "c2", "c3"))
  expect_true(all(c("item_id", "id", "kind", "reference", "prediction", review_classes, "text") %in% names(task$items)))
  # It follows the schema, and loads back from its folder
  schema <- jsonlite::read_json(system.file("schema", "face-task-v1.json", package = "transforEmotion"))
  expect_true(all(unlist(schema$`$defs`$review_item$required) %in% names(it)))
  again <- .te_load_annotation_task(task$dir)
  expect_equal(again$unit, "item")
  expect_type(again$items$id, "character")
  expect_equal(again$items$reference, task$items$reference)
  # Without a label and prediction, only the emotion is asked
  plain <- make_review(data.frame(text = c("a", "b")))
  expect_equal(vapply(plain$questions, `[[`, "", "id"), "emotion")
  expect_error(annotation_task(fake_faces(), items = text_items(), dir = tempfile()), "not both")
  expect_error(annotation_task(dir = tempfile()), "faces from detect_faces\\(\\), or items")
  expect_error(annotation_task(items = data.frame(media = "no-such.mp4"), dir = tempfile()), "not found")
  expect_warning(make_review(transform(text_items(), label = c("furious", "happy", "sad"))), "furious")
})

test_that("item tasks show clip frames with face boxes, and copy the clip", {
  root <- withr::local_tempdir()
  clip <- file.path(root, "clip.mp4")
  writeLines("not really a video", clip)
  calls <- new.env()
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    .te_faces_py = function() list(
      extract_frames = function(path, nframes, uniform, frame_dir, video_name) {
        calls$extract <- c(nframes = nframes, uniform = uniform)
        lapply(seq_len(nframes), function(k) {
          p <- file.path(frame_dir, sprintf("%s-%d.jpg", video_name, k))
          file.create(p)
          list(path = p, time = k - 0.5, frame_index = k * 10L)
        })
      },
      display_copies = function(images, out_paths, max_side) {
        file.create(unlist(out_paths))
        lapply(images, function(i) list(640L, 360L))
      }),
    .te_item_faces = function(te_faces, paths, frames) {
      data.frame(source = paths[c(1, 1, 2)], x = c(10, 300, 20), y = 5, width = c(50, 120, 60),
                 height = c(50, 120, 60), scored = c(FALSE, TRUE, TRUE))
    }
  )
  items <- data.frame(id = "v1", media = clip, text = "A clip.", label = "sad", prediction = "angry")
  task <- make_review(items, frames = 3)
  it <- jsonlite::read_json(file.path(task$dir, "task.json"))$items[[1]]
  expect_equal(it$kind, "clip")
  expect_length(it$frames, 3)
  expect_equal(unname(calls$extract), c(3, 1))
  expect_equal(it$frames[[1]]$src, "media/c0001-01.jpg")
  expect_equal(c(it$frames[[1]]$width, it$frames[[1]]$height), c(640, 360))
  expect_equal(vapply(it$frames[[1]]$faces, `[[`, TRUE, "scored"), c(FALSE, TRUE))
  expect_equal(it$frames[[2]]$time, 1.5)
  expect_equal(it$video, "media/c0001.mp4")
  expect_true(file.exists(file.path(task$dir, "page", "media", "c0001.mp4")))
  # The page may play its own clips
  expect_match(paste(readLines(file.path(task$dir, "page", "index.html")), collapse = ""), "media-src 'self'", fixed = TRUE)
  expect_equal(task$items$n_frames, 3)
  # Without play, no clip in the page
  quiet <- make_review(items, play = FALSE)
  expect_null(jsonlite::read_json(file.path(quiet$dir, "task.json"))$items[[1]]$video)

  # In a notebook: frames inside the page, no clip
  local_mocked_bindings(.te_display_html = function(html, page) invisible(TRUE))
  page <- suppressMessages(annotate_faces(task, inline = TRUE))
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  expect_false(grepl("media/c0001", html, fixed = TRUE))
  expect_match(html, "data:image/jpeg;base64,", fixed = TRUE)
})

test_that("answers to an item task carry the item's reference and the model's answer", {
  local_mocked_bindings(ensure_te_py_env = function() stop("no Python needed for texts"))
  task <- make_review(coders = c("C01", "C02"))
  id_of <- stats::setNames(task$items$item_id, task$items$id)
  item_rec <- function(id, coder, emotion, why) {
    r <- rec(id_of[[id]], coder = coder, task = "r1", seq = match(id, c("c1", "c2", "c3")))
    r$responses <- list(emotion = emotion, why = why)
    r
  }
  dir.create(file.path(task$dir, "results"), showWarnings = FALSE)
  write_jsonl(list(item_rec("c1", "C01", "angry", "model"), item_rec("c2", "C01", "happy", "agree"),
                   item_rec("c3", "C01", "sad", "agree")), file.path(task$dir, "results", "C01.jsonl"))
  write_jsonl(list(item_rec("c1", "C02", "angry", "model"), item_rec("c2", "C02", "happy", "agree"),
                   item_rec("c3", "C02", "angry", "label")), file.path(task$dir, "results", "C02.jsonl"))
  ann <- read_annotations(task)
  expect_true(all(c("id", "reference", "prediction", review_classes) %in% names(ann)))
  why <- ann[ann$question == "why" & ann$coder_id == "C01", ]
  expect_equal(why$value[match(c("c1", "c2", "c3"), why$id)], c("model", "agree", "agree"))
  cons <- consensus_labels(ann)
  cons <- cons[match(c("c1", "c2", "c3"), cons$id), ]
  expect_equal(cons$label, c("angry", "happy", NA))
  expect_true(cons$tie[3])
  expect_equal(cons$reference, c("angry", "happy", "sad"))
  ok <- cons[!is.na(cons$label), ]
  coders_vs_label <- evaluate_emotions(ok, truth_col = "reference", pred_col = "label",
                                       classes = review_classes, metrics = "accuracy")
  expect_equal(coders_vs_label$accuracy, 1)
  model_vs_coders <- evaluate_emotions(ok, truth_col = "label", pred_col = "prediction",
                                       classes = review_classes, metrics = "accuracy")
  expect_equal(model_vs_coders$accuracy, 0.5)
  agree <- annotation_agreement(ann)
  expect_equal(agree$question, c("emotion", "why"))
  expect_output(print(task), "Annotation task r1")
})
