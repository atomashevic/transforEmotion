# view_faces() and read_face_selection(). The tests at the top mock Python;
# the last one runs it (see helper-faces.R). The page itself is tested by
# tests/js (Node) and by hand in a browser; its local server (face_server.R)
# is tested here with made-up requests and, in view mode, real ones.

# Mocks the Python display copies: writes empty files and reports a size
mock_display_copies <- function(env = parent.frame()) {
  calls <- new.env()
  display_copies <- function(images, out_paths, max_side) {
    calls$images <- unlist(images)
    calls$max_side <- max_side
    file.create(unlist(out_paths))
    lapply(images, function(i) list(400L, 300L))
  }
  local_mocked_bindings(ensure_te_py_env = function() invisible(TRUE),
                        .te_faces_py = function() list(display_copies = display_copies),
                        .env = env)
  calls
}

# The task view_faces() wrote to data.js
read_task <- function(dir) {
  js <- grep("^window\\.TE_TASK = ", readLines(file.path(dir, "data.js"), warn = FALSE), value = TRUE)
  jsonlite::fromJSON(sub(";\\s*$", "", sub("^window\\.TE_TASK = ", "", js)))
}

# A video faces table: track 1 in frames 0 and 15, track 2 in frame 15
fake_video_faces <- function() {
  out <- fake_faces(c("v.mp4", "v.mp4"))
  out$frame <- c(0L, 1L, 1L)
  out$frame_index <- c(0L, 15L, 15L)
  out$time <- c(0, 0.6, 0.6)
  out$face <- c(1L, 1L, 2L)
  out$track_id <- c(1L, 1L, 2L)
  attr(out, "media") <- data.frame(source = "v.mp4", frame = 0:2, frame_index = c(0L, 15L, 30L),
                                   time = c(0, 0.6, 1.2), n_faces = c(1L, 2L, 0L))
  attr(out, "settings") <- list(type = "video", face_detector = "yunet", nframes = 3)
  out
}

test_that("faces get stable ids from their image or frame and number", {
  faces <- fake_faces()
  expect_equal(.te_face_ids(faces), c("m01#0#1", "m01#0#2", "m02#0#1"))
  # A subset keeps the ids of the full table
  expect_equal(.te_face_ids(faces[3, , drop = FALSE]), "m02#0#1")
  expect_equal(.te_face_ids(fake_video_faces()), c("m01#0#1", "m01#15#1", "m01#15#2"))

  twice <- faces
  twice$face <- 1L
  expect_error(.te_face_ids(twice), "two rows for the same face")
})

test_that("images without faces are shown only for a complete table", {
  faces <- fake_faces(c("a.jpg", "b.jpg"))
  attr(faces, "media") <- data.frame(source = c("a.jpg", "b.jpg", "empty.jpg"), n_faces = c(2L, 1L, 0L))
  expect_equal(.te_shown_media(faces, FALSE)$source, c("a.jpg", "b.jpg", "empty.jpg"))
  expect_equal(.te_shown_media(faces[3, , drop = FALSE], FALSE)$media_id, "m02")
  # Every sampled frame of a video
  expect_equal(.te_shown_media(fake_video_faces(), TRUE)$frame_index, c(0L, 15L, 30L))
})

test_that("view_faces writes a page with the images and faces", {
  dir <- file.path(withr::local_tempdir(), "page")
  calls <- mock_display_copies()
  faces <- fake_faces()
  index <- view_faces(faces, live = FALSE, dir = dir, open = FALSE, title = "Test")
  expect_equal(basename(index), "index.html")
  expect_setequal(list.files(dir, recursive = TRUE),
                  c("index.html", "style.css", "logic.js", "app.js", "data.js",
                    "media/m01.jpg", "media/m02.jpg"))
  expect_equal(calls$images, c("a.jpg", "b.jpg"))

  task <- read_task(dir)
  expect_equal(task$schema, "transforEmotion.face_task")
  expect_equal(task$mode, "view")
  expect_equal(task$title, "Test")
  expect_equal(task$media$label, c("a.jpg", "b.jpg"))
  expect_true(all(is.na(task$media$key))) # the made-up files do not exist
  expect_equal(task$media$img_w, c(400, 400))
  expect_equal(task$faces$face_id, c("m01#0#1", "m01#0#2", "m02#0#1"))
  expect_equal(task$faces$w, c(50, 80, 60))
  expect_equal(task$faces$person_id, c("p1", "p2", "p1"))
  expect_true(all(is.na(task$faces$track_id)))
  expect_false(any(task$faces$selected))

  expect_error(view_faces(faces, live = FALSE, dir = dir, open = FALSE), "new or empty folder")
  expect_error(view_faces(faces, selected = faces, open = FALSE), "needs select = TRUE")
  expect_error(view_faces(faces[0, ], open = FALSE), "no rows")
})

test_that("view_faces in select mode marks the faces to start with", {
  dir <- withr::local_tempdir()
  mock_display_copies()
  faces <- fake_faces()
  expect_message(view_faces(faces, select = TRUE, selected = select_faces(faces, "largest"),
                            live = FALSE, dir = dir, open = FALSE), "read_face_selection")
  task <- read_task(dir)
  expect_equal(task$mode, "select")
  expect_equal(task$faces$selected, c(FALSE, TRUE, TRUE))
  # Opened from disk, the page has no R to send to
  expect_false(any(grepl("TE_API", readLines(file.path(dir, "data.js")))))
})

test_that("view_faces shows every sampled frame of a video", {
  dir <- withr::local_tempdir()
  calls <- mock_display_copies()
  cleaned <- FALSE
  local_mocked_bindings(.te_face_frame_paths = function(faces, settings) {
    structure(paste0("frame-", faces$frame_index, ".jpg"), cleanup = function() cleaned <<- TRUE)
  })
  view_faces(fake_video_faces(), live = FALSE, dir = dir, open = FALSE)
  expect_equal(calls$images, c("frame-0.jpg", "frame-15.jpg", "frame-30.jpg"))
  expect_true(cleaned)

  task <- read_task(dir)
  expect_equal(nrow(task$media), 1)
  expect_equal(task$media$media_type, "video")
  frames <- task$media$frames[[1]]
  expect_equal(frames$frame_index, c(0, 15, 30))
  expect_equal(frames$src, sprintf("media/m01-f%06d.jpg", c(0, 15, 30)))
  expect_true(all(file.exists(file.path(dir, frames$src))))
  expect_equal(task$faces$track_id, c(1, 1, 2))
  expect_equal(task$faces$timestamp_s, c(0, 0.6, 0.6))
})

test_that("read_face_selection returns the chosen faces", {
  faces <- fake_faces()
  selection <- function(ids, labels = c("a.jpg", "b.jpg")) {
    jsonlite::toJSON(list(
      schema = "transforEmotion.face_selection", schema_version = "1.0.0", task_id = "t",
      media = data.frame(media_id = c("m01", "m02"), label = labels),
      selected_face_ids = I(ids)
    ), auto_unbox = TRUE)
  }
  chosen <- read_face_selection(as.character(selection(c("m02#0#1", "m01#0#2"))), faces)
  expect_s3_class(chosen, "te_faces")
  expect_equal(chosen$x, c(200, 50))
  expect_identical(attr(chosen, "media"), attr(faces, "media"))

  file <- withr::local_tempfile(fileext = ".json")
  writeLines(selection("m01#0#1"), file)
  expect_equal(nrow(read_face_selection(file, faces)), 1)
  # A subset without the chosen face
  expect_warning(none <- read_face_selection(file, select_faces(faces, persons = "p2")), "not in faces")
  expect_equal(nrow(none), 0)

  expect_warning(out <- read_face_selection(as.character(selection(c("m01#0#1", "m07#0#1"))), faces),
                 "1 selected face\\(s\\) are not in faces \\(m07#0#1\\)")
  expect_equal(nrow(out), 1)
  expect_error(read_face_selection(as.character(selection("m01#0#1", c("x.jpg", "b.jpg"))), faces),
               "other images or another video")
  expect_error(read_face_selection('{"schema": "something else"}', faces), "not a face selection")

  # Same file names in another folder: told apart by their content
  dirs <- file.path(withr::local_tempdir(), c("p1", "p2"))
  for (i in 1:2) {
    dir.create(dirs[i])
    writeLines(paste("a in", i), file.path(dirs[i], "a.jpg"))
    writeLines(paste("b in", i), file.path(dirs[i], "b.jpg"))
  }
  ours <- fake_faces(file.path(dirs[1], c("a.jpg", "b.jpg")))
  theirs <- fake_faces(file.path(dirs[2], c("a.jpg", "b.jpg")))
  keyed <- jsonlite::toJSON(list(
    schema = "transforEmotion.face_selection", task_id = "t",
    media = data.frame(media_id = c("m01", "m02"), label = c("a.jpg", "b.jpg"),
                       key = .te_media_key(ours$source[c(1, 3)])),
    selected_face_ids = "m02#0#1"
  ), auto_unbox = TRUE)
  expect_equal(nrow(read_face_selection(as.character(keyed), ours)), 1)
  expect_error(read_face_selection(as.character(keyed), theirs), "other images or another video")
  # Files that cannot be read any more are checked by name only
  unlink(dirs[1], recursive = TRUE)
  expect_equal(nrow(read_face_selection(as.character(keyed), ours)), 1)
  expect_equal(.te_media_key(c(file.path(dirs[2], "b.jpg"), "https://example.org/x.jpg", dirs[2]))[-1],
               c(NA_character_, NA_character_))
  expect_error(read_face_selection(file.path(tempdir(), "no-such.json"), faces), "does not exist")
})

test_that("read_face_selection reads the code a notebook page gives", {
  faces <- fake_faces()
  doc <- jsonlite::toJSON(list(
    schema = "transforEmotion.face_selection", schema_version = "1.0.0", task_id = "t",
    media = data.frame(media_id = c("m01", "m02"), label = c("a.jpg", "b.jpg")),
    selected_face_ids = I(c("m01#0#2", "m02#0#1"))
  ), auto_unbox = TRUE)
  code <- paste0("te-selection:", jsonlite::base64_enc(charToRaw(as.character(doc))))
  chosen <- read_face_selection(code, faces)
  expect_equal(chosen$x, c(200, 50))
  # Surrounding spaces from copying are fine; a cut-off code is not
  expect_equal(nrow(read_face_selection(paste0("  ", code, "\n"), faces)), 2)
  expect_error(read_face_selection(substr(code, 1, 40), faces), "cut off|selection")
})

test_that("view_faces in a notebook shows one self-contained page below the cell", {
  dir <- file.path(withr::local_tempdir(), "page")
  calls <- mock_display_copies()
  shown <- new.env()
  local_mocked_bindings(.te_display_html = function(html, page) {
    shown$html <- html
    shown$page <- page
    invisible(TRUE)
  })
  my_faces <- fake_faces()
  expect_message(page <- view_faces(my_faces, select = TRUE, inline = TRUE, dir = dir,
                                    selected = my_faces[1, ]),
                 "Copy R code")
  expect_equal(basename(page), "faces.html")
  html <- paste(readLines(page, warn = FALSE), collapse = "\n")
  # Nothing outside the page: styles, scripts and images are inside it
  expect_false(grepl("<script src=|<link rel=\"stylesheet\"|src=\"media/", html))
  expect_match(html, "data:image/jpeg;base64,", fixed = TRUE)
  expect_match(html, "window.TE_NOTEBOOK = {\"faces\":\"my_faces\"};", fixed = TRUE)
  expect_match(html, "script-src 'unsafe-inline'", fixed = TRUE)
  expect_false(grepl("window.TE_API = ", html, fixed = TRUE))
  task <- jsonlite::fromJSON(sub(";$", "", regmatches(html, regexpr("(?<=window\\.TE_TASK = ).*?;(?=\n)", html, perl = TRUE))))
  expect_match(task$instructions, "Copy R code")
  expect_equal(task$faces$selected, c(TRUE, FALSE, FALSE))
  expect_match(task$media$src, "^data:image/jpeg;base64,")
  # The notebook gets an iframe holding the page, escaped as an attribute
  expect_match(shown$html, "<iframe id=\"te-faces-", fixed = TRUE)
  expect_match(shown$html, "srcdoc=\"&lt;!doctype html&gt;|srcdoc=\"<!doctype html>")
  expect_false(grepl("srcdoc=\"[^\"]*\"[^>]*window.TE_TASK", shown$html))
  expect_equal(normalizePath(shown$page), normalizePath(page))
  # Smaller copies by default
  expect_equal(calls$max_side, 640L)
})

test_that("view_faces shows the page in the notebook by default in Jupyter", {
  calls <- mock_display_copies()
  shown <- new.env()
  local_mocked_bindings(.te_display_html = function(html, page) {
    shown$n <- c(shown$n, 1)
    invisible(TRUE)
  })
  withr::local_options(jupyter.in_kernel = TRUE)
  page <- view_faces(fake_faces(), dir = file.path(withr::local_tempdir(), "page"))
  expect_equal(length(shown$n), 1)
  expect_equal(basename(page), "faces.html")
  # A table that is not a plain name: the code uses faces
  dir <- file.path(withr::local_tempdir(), "page2")
  page <- suppressMessages(view_faces(fake_faces()[1:2, ], select = TRUE, dir = dir))
  expect_match(paste(readLines(page, warn = FALSE), collapse = "\n"),
               "window.TE_NOTEBOOK = {\"faces\":\"faces\"};", fixed = TRUE)
})

test_that("scripts inlined into the notebook page cannot end early", {
  dir <- withr::local_tempdir()
  file.copy(system.file("www", "face-annotator", .te_face_app_files, package = "transforEmotion"), dir)
  writeLines('window.TE_TASK = {"title": "a </script><b>"};', file.path(dir, "data.js"))
  page <- .te_single_file_page(dir)
  expect_false(grepl("a </script>", page, fixed = TRUE))
  expect_match(page, "a <\\/script><b>", fixed = TRUE)
  expect_equal(lengths(regmatches(page, gregexpr("<script>", page, fixed = TRUE))), 3)
})

test_that("the notebook page keeps UTF-8 text in any locale", {
  dir <- withr::local_tempdir()
  file.copy(system.file("www", "face-annotator", .te_face_app_files, package = "transforEmotion"), dir)
  writeLines(enc2utf8('window.TE_TASK = {"title": "čaša · ☕"};'), file.path(dir, "data.js"),
             useBytes = TRUE)
  withr::local_locale(c(LC_CTYPE = "C"))
  page <- .te_single_file_page(dir)
  bytes <- function(x) paste(as.character(charToRaw(x)), collapse = " ")
  expect_match(bytes(page), bytes(enc2utf8("čaša · ☕")), fixed = TRUE)
  # app.js labels such as "P1 · T2" stay intact
  expect_false(grepl("<U+00B7>", page, fixed = TRUE))
  html <- NULL
  local_mocked_bindings(.te_display_html = function(h, page) html <<- h)
  file <- file.path(dir, "faces.html")
  writeLines(page, file, useBytes = TRUE)
  .te_show_in_notebook(file, "faces-1")
  expect_match(bytes(html), bytes(enc2utf8("·")), fixed = TRUE)
  expect_false(grepl("<U+00B7>", html, fixed = TRUE))
})

# A request as httpuv hands it to the server's call function
fake_request <- function(path, body = NULL, method = "POST", port = 4321L,
                         type = "application/json", host = paste0("127.0.0.1:", port)) {
  raw_body <- charToRaw(if (is.null(body)) "" else body)
  list(REQUEST_METHOD = method, PATH_INFO = path, HTTP_HOST = host, CONTENT_TYPE = type,
       rook.input = list(read = function() raw_body))
}

test_that("the page's server passes only its own JSON posts to R", {
  handlers <- list(echo = function(body) list(got = body$x),
                   fail = function(body) stop("no good"))
  api <- function(...) .te_page_api(fake_request(...), "/tok", 4321L, handlers)
  ok <- api("/tok/api/echo", '{"x": "é"}')
  expect_equal(ok$status, 200L)
  expect_equal(jsonlite::fromJSON(ok$body)$got, "é")
  expect_equal(api("/tok/api/echo", '{"x": 1}', host = "evil.example:4321")$status, 403L)
  expect_equal(api("/tok/api/echo", '{"x": 1}', host = "127.0.0.1:80")$status, 403L)
  expect_equal(api("/tok/api/other", "{}")$status, 404L)
  expect_equal(api("/other/api/echo", "{}")$status, 404L)
  expect_equal(api("/tok/api/echo", method = "GET")$status, 405L)
  expect_equal(api("/tok/api/echo", "{}", type = "text/plain")$status, 415L)
  expect_equal(api("/tok/api/echo", "not json")$status, 400L)
  expect_equal(api("/tok/api/echo", '"/etc/passwd"')$status, 400L)
  failed <- api("/tok/api/fail", "{}")
  expect_equal(failed$status, 400L)
  expect_match(failed$body, "no good")
})

test_that("starting the server leaves the user's random numbers alone", {
  set.seed(1)
  a <- stats::runif(1)
  set.seed(1)
  token <- .te_random_token()
  expect_equal(stats::runif(1), a)
  expect_match(token, "^[0-9a-f]{32}$")
  expect_false(identical(token, .te_random_token()))
})

test_that("view_faces serves the page from R", {
  skip_on_cran() # opens a local port
  mock_display_copies()
  expect_message(page <- view_faces(fake_faces(), open = FALSE), "127\\.0\\.0\\.1")
  on.exit(page$stop(), add = TRUE)
  expect_s3_class(page, "te_face_page")
  expect_output(print(page), "Stop it")
  expect_match(page$url, "^http://127\\.0\\.0\\.1:\\d+/[0-9a-f]{32}/$")
  # Files are served by httpuv's own thread, so these work while R is busy
  get <- function(url) httr::GET(url, httr::timeout(5))
  index <- get(page$url)
  expect_equal(httr::status_code(index), 200L)
  expect_match(httr::content(index, as = "text", encoding = "UTF-8"), "<title>Faces</title>")
  data <- httr::content(get(paste0(page$url, "data.js")), as = "text", encoding = "UTF-8")
  expect_match(data, "window.TE_TASK")
  expect_false(grepl("TE_API", data)) # nothing to send in view mode
  expect_equal(httr::status_code(get(paste0(page$url, "media/m01.jpg"))), 200L)
  # Without the random part of the address there is nothing
  root <- sub("[0-9a-f]{32}/$", "", page$url)
  expect_equal(httr::status_code(get(root)), 404L)
  expect_equal(httr::status_code(get(paste0(root, "index.html"))), 404L)
  expect_equal(httr::status_code(get(paste0(root, "media/m01.jpg"))), 404L)
  page$stop()
  expect_error(httr::GET(page$url, httr::timeout(2)))
})

test_that("view_faces(select = TRUE) returns the faces sent from the page", {
  mock_display_copies()
  faces <- fake_faces()
  served <- new.env()
  local_mocked_bindings(
    .te_serve_page = function(dir, handlers = list(), port = NULL) {
      served$dir <- dir
      served$handlers <- handlers
      list(url = "http://127.0.0.1:4321/tok/", port = 4321L,
           stop = function() served$stopped <- TRUE)
    },
    # The browser's part: the page sends the selection while R waits
    .te_service = function() {
      if (is.null(served$sent)) {
        task <- read_task(served$dir)
        served$sent <- served$handlers$selection(list(
          schema = "transforEmotion.face_selection", task_id = task$task_id,
          media = data.frame(media_id = c("m01", "m02"), label = c("a.jpg", "b.jpg")),
          selected_face_ids = c("m01#0#2", "m02#0#1")))
      }
    }
  )
  expect_message(expect_message(chosen <- view_faces(faces, select = TRUE, open = FALSE),
                                "Send to R"), "2 face\\(s\\) selected")
  expect_equal(served$sent$n_faces, 2L)
  expect_equal(chosen$x, c(200, 50))
  expect_s3_class(chosen, "te_faces")
  expect_true(served$stopped)
  js <- readLines(file.path(served$dir, "data.js"))
  expect_true(any(grepl('window.TE_API = {"selection":"api/selection","cancel":"api/cancel"};', js, fixed = TRUE)))
  expect_equal(read_task(served$dir)$instructions,
               "Click faces, video tracks or people to select them, then Send to R.")

  # A selection from another call, and Cancel
  served$sent <- NULL
  local_mocked_bindings(.te_service = function() {
    expect_error(served$handlers$selection(list(schema = "transforEmotion.face_selection",
                                                task_id = "old", selected_face_ids = "m01#0#1")),
                 "another view_faces")
    served$handlers$cancel(list())
  })
  expect_message(expect_message(none <- view_faces(faces, select = TRUE, open = FALSE),
                                "Send to R"), "cancelled")
  expect_null(none)
})

test_that("view_faces writes display copies of the example picture", {
  skip_without_face_python()
  dir <- withr::local_tempdir()
  faces <- detect_faces(fixture("two-people.jpg"), identify = FALSE)
  view_faces(faces, live = FALSE, dir = dir, open = FALSE, max_side = 300)
  task <- read_task(dir)
  # Boxes stay in the original pixels; the copy is smaller
  expect_equal(task$media$img_w, faces$image_width[1])
  expect_equal(task$media$img_h, faces$image_height[1])
  size <- reticulate::import("PIL.Image")$open(file.path(dir, task$media$src))$size
  expect_equal(max(unlist(size)), 300)
  expect_equal(task$faces$x, round(faces$x, 4))
  expect_equal(task$media$key, unname(tools::md5sum(faces$source[1])))
})
