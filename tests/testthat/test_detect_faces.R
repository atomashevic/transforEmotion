# detect_faces(), select_faces(), face_scores() and face_selection. The tests
# at the top use made-up face tables and mock Python; the tests at the bottom
# run the te_faces Python module and need TRANSFOREMOTION_PYTHON to point at a
# Python environment with OpenCV.

# A faces table as detect_faces() returns it: two images, three faces
fake_faces <- function(sources = c("a.jpg", "b.jpg")) {
  out <- data.frame(
    source = sources[c(1, 1, 2)], frame = NA_integer_, frame_index = NA_integer_,
    time = NA_real_, face = c(1L, 2L, 1L), track_id = NA_integer_,
    person_id = c("p1", "p2", "p1"), x = c(10, 200, 50), y = 20,
    width = c(50, 80, 60), height = c(50, 80, 60), confidence = 0.9,
    detector = "yunet", image_width = 400L, image_height = 300L,
    stringsAsFactors = FALSE
  )
  for (lm in .te_landmark_columns) out[[lm]] <- 1
  attr(out, "media") <- data.frame(source = sources, n_faces = c(2L, 1L))
  attr(out, "settings") <- list(type = "images", face_detector = "yunet")
  class(out) <- c("te_faces", "data.frame")
  out
}

# Mocks Python for .te_face_scores(): the scores come back in the order the
# boxes were sent, which is the order of the image paths
mock_face_python <- function(scores) {
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    without_hf_token = function(expr) scores,
    .env = parent.frame()
  )
  local_mocked_bindings(source_python = function(...) invisible(NULL),
                        .package = "reticulate", .env = parent.frame())
}

test_that("face_selection accepts rules, all, face numbers and faces tables", {
  expect_equal(.te_face_selection("largest"), list(mode = "rule", value = "largest"))
  expect_equal(.te_face_selection("none")$mode, "rule")
  expect_equal(.te_face_selection("all"), list(mode = "all", value = "all"))
  expect_equal(.te_face_selection(c(1, 3)), list(mode = "ids", value = c(1L, 3L)))
  expect_equal(.te_face_selection(fake_faces())$mode, "table")

  expect_error(.te_face_selection("middle"), "face numbers")
  expect_error(.te_face_selection(c(1, 3), allow_ids = FALSE),
               "none, all, or a data frame")
  expect_error(.te_face_selection(0), "must be one of")
  expect_error(.te_face_selection(1.5), "must be one of")
  expect_error(.te_face_selection(NA), "must be one of")
  expect_error(.te_face_selection(c("largest", "left")), "must be one of")
  expect_error(.te_face_selection(data.frame(source = "a.jpg")), "detect_faces")
})

test_that("select_faces picks one face per image by rule", {
  faces <- fake_faces()
  expect_equal(select_faces(faces, "largest")$face, c(2L, 1L))
  expect_equal(select_faces(faces, "left")$x, c(10, 50))
  expect_equal(select_faces(faces, "right")$x, c(200, 50))
  expect_equal(nrow(select_faces(faces)), 3)
})

test_that("select_faces keeps faces by number, track and person", {
  faces <- fake_faces()
  expect_equal(select_faces(faces, ids = 1)$source, c("a.jpg", "b.jpg"))
  expect_equal(select_faces(faces, persons = "p2")$x, 200)
  expect_equal(select_faces(faces, "largest", persons = "p1")$x, c(10, 50))
  expect_equal(nrow(select_faces(faces, tracks = 1)), 0)

  faces$track_id <- c(1L, 2L, 2L)
  expect_equal(select_faces(faces, tracks = 2)$x, c(200, 50))

  no_person <- fake_faces()
  no_person$person_id <- NULL
  expect_error(select_faces(no_person, persons = "p1"), "no person_id column")
  expect_error(select_faces(data.frame(a = 1)), "data frame from detect_faces")
})

test_that("selecting and subsetting faces keep the media and settings", {
  faces <- fake_faces()
  for (subset in list(select_faces(faces, persons = "p1"), faces[faces$face == 1, ],
                      faces[1, , drop = FALSE])) {
    expect_s3_class(subset, "te_faces")
    expect_identical(attr(subset, "media"), attr(faces, "media"))
    expect_identical(attr(subset, "settings"), attr(faces, "settings"))
  }
  # A single column is still a vector
  expect_equal(faces[, "x"], c(10, 200, 50))
})

test_that("the Python faces table becomes the detect_faces() data frame", {
  nan <- NaN
  res <- list(
    list(item = list(1L, 1L), frame = list(nan, nan), frame_index = list(nan, nan),
         time = list(nan, nan), face = list(1L, 2L), track_id = list(nan, nan),
         person_id = list("p1", ""), x = list(1, 2), y = list(3, 4),
         width = list(40, 50), height = list(40, 50), confidence = list(0.9, nan),
         image_width = list(100L, 100L), image_height = list(80L, 80L)),
    list(item = list(1L, 2L), frame = list(nan, nan), frame_index = list(nan, nan),
         time = list(nan, nan), image_width = list(100L, 120L),
         image_height = list(80L, 90L), n_faces = list(2L, 0L))
  )
  for (lm in .te_landmark_columns) res[[1]][[lm]] <- list(1, nan)

  out <- .te_faces_frame(res, c("a.jpg", "b.jpg"), "yunet", list(type = "images"))
  expect_s3_class(out, "te_faces")
  expect_equal(out$source, c("a.jpg", "a.jpg"))
  expect_equal(out$person_id, c("p1", NA))
  expect_equal(out$confidence, c(0.9, NA))
  expect_equal(out$right_eye_x, c(1, NA))
  expect_true(all(is.na(out$frame)) && is.integer(out$frame))
  expect_equal(attr(out, "media")$n_faces, c(2L, 0L))
  expect_equal(attr(out, "media")$source, c("a.jpg", "b.jpg"))

  # No faces at all
  empty <- lapply(res[[1]], function(x) list())
  out <- .te_faces_frame(list(empty, res[[2]]), c("a.jpg", "b.jpg"), "yunet",
                         list(type = "images"))
  expect_equal(nrow(out), 0)
  expect_true(all(c("source", "face", "person_id", "x") %in% names(out)))
})

test_that("face_scores returns one row per face, in the order of the table", {
  faces <- fake_faces(sources = c("b.jpg", "a.jpg"))
  # Rows are sent sorted by image path: a.jpg (row 3) first, then b.jpg
  mock_face_python(data.frame(happy = c(0.1, 0.2, 0.3), sad = c(0.9, 0.8, 0.7)))
  local_mocked_bindings(.te_resolve_vision_model = function(...) list(model_id = "m"))

  out <- face_scores(faces, classes = c("happy", "sad"))
  expect_false(inherits(out, "te_faces"))
  expect_equal(out$happy, c(0.2, 0.3, 0.1))
  expect_equal(out$sad, c(0.8, 0.7, 0.9))
  expect_equal(out$person_id, faces$person_id)
  expect_false(any(.te_landmark_columns %in% names(out)))

  none <- face_scores(faces[0, ], classes = c("happy", "sad"))
  expect_equal(nrow(none), 0)
  expect_true(all(c("source", "happy", "sad") %in% names(none)))

  expect_error(face_scores(faces, classes = "happy"), "at least 2")
})

test_that("image_scores scores the faces of a faces table in that image", {
  image <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  faces <- fake_faces(sources = c(image, "other.jpg"))
  mock_face_python(data.frame(a = c(0.4, 0.6), b = c(0.6, 0.4)))

  out <- suppressMessages(image_scores(image, c("a", "b"), face_selection = faces))
  expect_equal(nrow(out), 2)
  expect_equal(out$face, c(1L, 2L))
  expect_equal(out$a, c(0.4, 0.6))

  expect_error(
    suppressMessages(image_scores(image, c("a", "b"),
                                  face_selection = fake_faces(c("x.jpg", "y.jpg")))),
    "no faces in this image"
  )
  expect_error(suppressMessages(image_scores(image, c("a", "b"), face_selection = "middle")),
               "must be one of")
})

test_that("image_scores detects faces for all and face numbers", {
  image <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  mock_face_python(data.frame(a = 0.3, b = 0.7))
  detect_args <- NULL
  local_mocked_bindings(detect_faces = function(x, ...) {
    detect_args <<- list(...)
    fake_faces(sources = c(x, "unused.jpg"))[1:2, ]
  })

  out <- suppressMessages(image_scores(image, c("a", "b"), face_selection = 2,
                                       face_detector = "haar"))
  expect_equal(out$face, 2L)
  expect_equal(out$b, 0.7)
  expect_equal(detect_args$face_detector, "haar")
  expect_false(detect_args$identify)
})

test_that("video_scores refuses a faces table from another video", {
  dir <- withr::local_tempdir()
  video <- file.path(dir, "clip.mp4")
  writeBin(as.raw(1:4), video)
  faces <- fake_faces(sources = c("other.mp4", "other.mp4"))
  local_mocked_bindings(ensure_te_py_env = function() invisible(TRUE))
  local_mocked_bindings(source_python = function(...) invisible(NULL), .package = "reticulate")
  expect_error(
    suppressMessages(video_scores(video, c("a", "b"), face_selection = faces,
                                  save_dir = dir)),
    "faces of another video"
  )
  expect_error(
    suppressMessages(video_scores(video, c("a", "b"), face_selection = 1, save_dir = dir)),
    "must be one of"
  )
})

test_that("video_scores scores a faces table on the frames it kept", {
  dir <- withr::local_tempdir()
  video <- file.path(dir, "clip.mp4")
  writeBin(as.raw(1:4), video)
  frames <- file.path(dir, sprintf("clip-frame-%d.jpg", 0:1))
  for (f in frames) writeBin(as.raw(1:4), f)

  faces <- fake_faces(sources = c(video, video))
  faces$frame <- c(0L, 0L, 1L)
  faces$frame_index <- c(0L, 0L, 15L)
  faces$track_id <- c(1L, 2L, 1L)
  attr(faces, "media") <- data.frame(source = video, frame = 0:1, frame_index = c(0L, 15L),
                                     n_faces = c(2L, 1L), frame_path = frames)
  attr(faces, "settings") <- list(type = "video", nframes = 2, start = 0, end = -1,
                                  uniform = FALSE, ffreq = 15)
  mock_face_python(data.frame(a = c(0.1, 0.2), b = c(0.9, 0.8)))

  out <- suppressMessages(video_scores(video, c("a", "b"),
                                       face_selection = select_faces(faces, tracks = 1),
                                       save_dir = dir))
  expect_equal(nrow(out), 2)
  expect_equal(out$frame, c(0L, 1L))
  expect_equal(out$a, c(0.1, 0.2))
  # The frames belong to the faces table, not to this call
  expect_true(all(file.exists(frames)))

  # Frames extracted again for scoring are removed afterwards
  cleaned <- FALSE
  local_mocked_bindings(
    .te_face_frame_paths = function(faces, settings) {
      structure(frames, cleanup = function() cleaned <<- TRUE)
    },
    .te_resolve_vision_model = function(...) list(model_id = "m")
  )
  out <- face_scores(select_faces(faces, tracks = 1), c("a", "b"))
  expect_equal(out$b, c(0.9, 0.8))
  expect_true(cleaned)
})

test_that("image_scores_dir warns when images share a file name", {
  dir <- withr::local_tempdir()
  dir.create(file.path(dir, "one"))
  dir.create(file.path(dir, "two"))
  image <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  file.copy(image, file.path(dir, "one", "x.png"))
  file.copy(image, file.path(dir, "two", "x.png"))
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    without_hf_token = function(expr) data.frame(image_id = c("x.png", "x.png"),
                                                 a = 0.5, b = 0.5)
  )
  local_mocked_bindings(source_python = function(...) invisible(NULL), .package = "reticulate")
  expect_warning(
    out <- suppressMessages(image_scores_dir(dir, c("a", "b"), recursive = TRUE)),
    "share a file name.*x.png"
  )
  expect_equal(nrow(out), 2)
})

# ---------------------------------------------------------------------------
# te_faces in Python
# ---------------------------------------------------------------------------

skip_without_face_python <- function() {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON")),
          "TRANSFOREMOTION_PYTHON is not set")
  ensure_te_py_env()
  skip_if_not(reticulate::py_module_available("cv2"), "OpenCV is not available")
}

fixture <- function(name) system.file("extdata", name, package = "transforEmotion")

test_that("YuNet and Haar find the face in each fixture", {
  skip_without_face_python()
  images <- fixture(c("boris-1.png", "boris-2.png", "trump1.jpg", "trump2.jpg"))

  faces <- suppressMessages(detect_faces(images, identify = FALSE))
  expect_equal(as.vector(table(factor(faces$source, levels = images))), c(1, 1, 1, 1))
  expect_true(all(faces$confidence > 0.6))
  expect_false(anyNA(faces[, .te_landmark_columns]))
  expect_true(all(faces$person_id %in% NA))
  expect_equal(attr(faces, "media")$n_faces, c(1L, 1L, 1L, 1L))
  # One image (a length-one path must reach Python as a list)
  one <- detect_faces(images[1], identify = FALSE)
  expect_equal(nrow(one), 1)
  expect_equal(one$source, images[1])

  haar <- detect_faces(images, face_detector = "haar", identify = FALSE)
  expect_true(all(table(haar$source) >= 1))
  expect_true(all(is.na(haar$confidence)))
  expect_true(all(is.na(haar$right_eye_x)))
  expect_equal(unique(haar$detector), "haar")
})

test_that("the Haar detector finds the boxes crop_face() has always used", {
  skip_without_face_python()
  te_faces <- .te_faces_py()
  path <- fixture("trump1.jpg")
  # The detection step of crop_face() in image.py
  old <- reticulate::py_run_string(paste0(
    "import cv2, numpy as np\n",
    "from PIL import Image\n",
    "gray = cv2.cvtColor(np.array(Image.open(", encodeString(path, quote = "'"),
    ").convert('RGB')), cv2.COLOR_RGB2GRAY)\n",
    "cascade = cv2.CascadeClassifier(cv2.data.haarcascades + ",
    "'haarcascade_frontalface_default.xml')\n",
    "widths = [float(f[2]) for f in cascade.detectMultiScale(gray, scaleFactor=1.1, ",
    "minNeighbors=5, minSize=(30, 30))]\n"
  ), local = TRUE)$widths
  new <- te_faces$detect(te_faces$load_image(path), "haar")
  expect_gt(length(new), 0)
  expect_setequal(vapply(new, function(f) f$width, numeric(1)), unlist(old))
})

test_that("faces of one person share a person_id across images", {
  skip_without_face_python()
  skip_if(is.null(suppressMessages(.te_sface_model(download = FALSE))),
          "Face recognition model is not downloaded")
  images <- fixture(c("boris-1.png", "trump1.jpg", "boris-2.png", "trump2.jpg"))
  faces <- detect_faces(images)
  expect_equal(faces$person_id, c("p1", "p2", "p1", "p2"))

  # Two people in one image, and one of them again in another image
  te_faces <- .te_faces_py()
  composite <- file.path(withr::local_tempdir(), "composite.png")
  left <- te_faces$load_image(images[1])$resize(reticulate::tuple(400L, 400L))
  right <- te_faces$load_image(images[2])$resize(reticulate::tuple(400L, 400L))
  PIL <- reticulate::import("PIL.Image")
  canvas <- PIL$new("RGB", reticulate::tuple(800L, 400L))
  canvas$paste(left, reticulate::tuple(0L, 0L))
  canvas$paste(right, reticulate::tuple(400L, 0L))
  canvas$save(composite)

  faces <- detect_faces(c(composite, images[3]))
  expect_equal(faces$person_id, c("p1", "p2", "p1"))
  expect_equal(select_faces(faces, persons = "p1")$source, c(composite, images[3]))
})

test_that("assign_person_ids groups units by average cosine similarity", {
  skip_without_face_python()
  te_faces <- .te_faces_py()
  unit <- function(...) list(c(...))
  units <- list(
    unit(1, 0, 0), unit(0, 1, 0), unit(0.9, 0.1, 0), list(NULL),
    list(c(0, 0.9, 0.1), c(0, 1, 0.05))
  )
  ids <- te_faces$assign_person_ids(units, threshold = 0.363)
  expect_equal(ids, list("p1", "p2", "p1", NULL, "p2"))
  # A threshold above every similarity keeps all units apart
  ids <- te_faces$assign_person_ids(units, threshold = 0.9999)
  expect_equal(unlist(ids), c("p1", "p2", "p3", "p4"))
})

test_that("tracks follow faces across frames and are numbered left to right", {
  skip_without_face_python()
  te_faces <- .te_faces_py()
  box <- function(x) list(x = x, y = 10, width = 50, height = 50)
  frames <- list(list(box(300), box(10)), list(box(305), box(12)), list(), list(box(15)))
  tracked <- te_faces$track(frames, iou_threshold = 0.3, max_gap = 5L)
  ids <- lapply(tracked, function(f) vapply(f, function(x) as.integer(x$track_id), 1L))
  expect_equal(ids, list(c(2L, 1L), c(2L, 1L), integer(0), 1L))
  # A gap longer than max_gap starts a new track
  frames <- list(list(box(10)), list(), list(), list(box(10)))
  tracked <- te_faces$track(frames, max_gap = 1L)
  expect_equal(vapply(tracked[c(1, 4)], function(f) as.integer(f[[1]]$track_id), 1L),
               c(1L, 2L))
})

test_that("extract_frames samples the frames video_scores() analyses", {
  skip_without_face_python()
  te_faces <- .te_faces_py()
  cv2 <- reticulate::import("cv2")
  np <- reticulate::import("numpy", convert = FALSE)
  dir <- withr::local_tempdir()
  video <- file.path(dir, "clip.avi")
  writer <- cv2$VideoWriter(video, cv2$VideoWriter_fourcc("M", "J", "P", "G"), 10,
                            reticulate::tuple(64L, 48L))
  for (i in 0:59) writer$write(np$full(c(48L, 64L, 3L), as.integer(i * 4), dtype = "uint8"))
  writer$release()

  index <- function(frames) vapply(frames, function(f) as.integer(f$frame_index), 1L)
  frames <- te_faces$extract_frames(video, 4L, ff = 15L, frame_dir = dir, video_name = "v")
  expect_equal(index(frames), c(0L, 15L, 30L, 45L))
  expect_equal(vapply(frames, function(f) f$time, 1), c(0, 1.5, 3, 4.5))
  expect_true(all(file.exists(file.path(dir, sprintf("v-frame-%d.jpg", 0:3)))))

  # Uniform sampling spreads the frames over the whole video, whatever ff is
  frames <- te_faces$extract_frames(video, 4L, uniform = TRUE, ff = 100L, frame_dir = dir,
                                    video_name = "u")
  expect_equal(index(frames), c(0L, 15L, 30L, 45L))
  # A cut stays inside start and end
  frames <- te_faces$extract_frames(video, 10L, start = 1, end = 3, ff = 5L,
                                    frame_dir = dir, video_name = "c")
  expect_equal(index(frames), c(10L, 15L, 20L, 25L))
  expect_error(te_faces$extract_frames(video, 5L, ff = 15L, frame_dir = dir),
               "too short")
})
