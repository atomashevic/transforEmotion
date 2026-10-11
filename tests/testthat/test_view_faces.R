# view_faces() and read_face_selection(). The tests at the top mock Python;
# the last one runs it (see helper-faces.R). The page itself is tested by
# tests/js (Node) and by hand in a browser.

# Mocks the Python display copies: writes empty files and reports a size
mock_display_copies <- function(env = parent.frame()) {
  calls <- new.env()
  display_copies <- function(images, out_paths, max_side) {
    calls$images <- unlist(images)
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
  js <- paste(readLines(file.path(dir, "data.js"), warn = FALSE), collapse = "\n")
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
  index <- view_faces(faces, dir = dir, open = FALSE, title = "Test")
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
  expect_equal(task$media$img_w, c(400, 400))
  expect_equal(task$faces$face_id, c("m01#0#1", "m01#0#2", "m02#0#1"))
  expect_equal(task$faces$w, c(50, 80, 60))
  expect_equal(task$faces$person_id, c("p1", "p2", "p1"))
  expect_true(all(is.na(task$faces$track_id)))
  expect_false(any(task$faces$selected))

  expect_error(view_faces(faces, dir = dir, open = FALSE), "new or empty folder")
  expect_error(view_faces(faces, selected = faces, open = FALSE), "needs select = TRUE")
  expect_error(view_faces(faces[0, ], open = FALSE), "no rows")
})

test_that("view_faces in select mode marks the faces to start with", {
  dir <- withr::local_tempdir()
  mock_display_copies()
  faces <- fake_faces()
  expect_message(view_faces(faces, select = TRUE, selected = select_faces(faces, "largest"),
                            dir = dir, open = FALSE), "read_face_selection")
  task <- read_task(dir)
  expect_equal(task$mode, "select")
  expect_equal(task$faces$selected, c(FALSE, TRUE, TRUE))
})

test_that("view_faces shows every sampled frame of a video", {
  dir <- withr::local_tempdir()
  calls <- mock_display_copies()
  cleaned <- FALSE
  local_mocked_bindings(.te_face_frame_paths = function(faces, settings) {
    structure(paste0("frame-", faces$frame_index, ".jpg"), cleanup = function() cleaned <<- TRUE)
  })
  view_faces(fake_video_faces(), dir = dir, open = FALSE)
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
  expect_error(read_face_selection(file.path(tempdir(), "no-such.json"), faces), "does not exist")
})

test_that("view_faces writes display copies of the example picture", {
  skip_without_face_python()
  dir <- withr::local_tempdir()
  faces <- detect_faces(fixture("two-people.jpg"), identify = FALSE)
  view_faces(faces, dir = dir, open = FALSE, max_side = 300)
  task <- read_task(dir)
  # Boxes stay in the original pixels; the copy is smaller
  expect_equal(task$media$img_w, faces$image_width[1])
  expect_equal(task$media$img_h, faces$image_height[1])
  size <- reticulate::import("PIL.Image")$open(file.path(dir, task$media$src))$size
  expect_equal(max(unlist(size)), 300)
  expect_equal(task$faces$x, round(faces$x, 4))
})
