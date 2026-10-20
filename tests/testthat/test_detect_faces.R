# detect_faces(), select_faces(), face_scores(), draw_faces() and
# face_selection. The tests at the top use made-up face tables (helper-faces.R)
# and mock Python; the tests at the bottom run the te_faces Python module and
# need TRANSFOREMOTION_PYTHON to point at a Python environment with OpenCV.

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

test_that("only YouTube URLs count as YouTube inputs to detect_faces", {
  dir <- file.path(withr::local_tempdir(), "youtube")
  dir.create(dir)
  video <- file.path(dir, "clip.mp4")
  file.create(video)
  expect_equal(.te_face_inputs(video, "", FALSE)$type, "video")
  expect_equal(.te_face_inputs("https://youtu.be/abc", "", FALSE)$type, "youtube")
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

# Mocks the Python drawing: records each call and writes an empty file
mock_draw <- function(env = parent.frame()) {
  calls <- new.env()
  calls$list <- list()
  draw <- function(image, boxes, labels, colors, out_path, ...) {
    calls$list[[length(calls$list) + 1]] <- list(
      image = image, boxes = boxes, labels = unlist(labels), colors = colors
    )
    file.create(out_path)
    array(255, c(2, 3, 3))
  }
  local_mocked_bindings(ensure_te_py_env = function() invisible(TRUE),
                        .te_faces_py = function() list(draw = draw), .env = env)
  calls
}

test_that("faces are labelled by number, track and person, and coloured by person", {
  faces <- fake_faces()
  expect_equal(.te_face_labels(faces, FALSE), c("face 1, p1", "face 2, p2", "face 1, p1"))
  colors <- .te_face_colors(faces)
  expect_identical(colors[[1]], colors[[3]])
  expect_false(identical(colors[[1]], colors[[2]]))

  faces$person_id <- c("p1", NA, NA)
  expect_equal(.te_face_labels(faces, FALSE), c("face 1, p1", "face 2", "face 1"))

  # Video: tracks instead of face numbers; without person ids, a colour per track
  faces$track_id <- c(1L, 2L, 1L)
  faces$person_id <- NULL
  expect_equal(.te_face_labels(faces, TRUE), c("track 1", "track 2", "track 1"))
  colors <- .te_face_colors(faces)
  expect_identical(colors[[1]], colors[[3]])
  expect_false(identical(colors[[1]], colors[[2]]))
})

test_that("drawn images are laid out to fit the plot window", {
  img <- function(h, w) array(0, c(h, w, 3))
  grDevices::pdf(NULL, width = 12, height = 4)
  withr::defer(grDevices::dev.off())
  expect_equal(.te_image_grid(rep(list(img(10, 10)), 3)), c(1, 3))
  expect_equal(.te_image_grid(rep(list(img(10, 20)), 4)), c(2, 2))
  # At most six to a page
  expect_equal(.te_image_grid(rep(list(img(10, 10)), 20)), c(2, 3))
})

test_that("draw_faces draws each image once, with its faces", {
  dir <- withr::local_tempdir()
  calls <- mock_draw()
  faces <- fake_faces(c("one/a.jpg", "two/a.jpg"))
  paths <- draw_faces(faces, dir = dir, plot = FALSE)
  # Images with the same file name get different files
  expect_equal(basename(paths), c("a-faces.png", "a-1-faces.png"))
  expect_equal(vapply(calls$list, `[[`, "", "image"), c("one/a.jpg", "two/a.jpg"))
  first <- calls$list[[1]]
  expect_equal(first$labels, c("face 1, p1", "face 2, p2"))
  expect_equal(first$boxes, list(c(10, 20, 50, 50), c(200, 20, 80, 80)))
  expect_equal(calls$list[[2]]$colors, first$colors[1])

  calls$list <- list()
  draw_faces(faces, labels = "", max_images = 1, dir = dir, plot = FALSE)
  expect_length(calls$list, 1)
  expect_equal(calls$list[[1]]$labels, c("", ""))

  grDevices::pdf(NULL)
  withr::defer(grDevices::dev.off())
  expect_no_error(plot(faces, dir = dir))

  expect_error(draw_faces(faces, labels = c("a", "b")), "one entry per row")
  expect_error(draw_faces(data.frame(a = 1)), "data frame from detect_faces")
  expect_message(none <- draw_faces(faces[0, ]), "No faces to draw")
  expect_length(none, 0)
})

test_that("draw_faces draws video frames with track numbers", {
  dir <- withr::local_tempdir()
  calls <- mock_draw()
  cleaned <- FALSE
  local_mocked_bindings(.te_face_frame_paths = function(faces, settings) {
    structure(paste0("frame-", faces$frame_index, ".jpg"),
              cleanup = function() cleaned <<- TRUE)
  })
  faces <- fake_faces(c("v.mp4", "v.mp4"))
  faces$frame <- c(0L, 0L, 1L)
  faces$frame_index <- c(0L, 0L, 25L)
  faces$time <- c(0, 0, 1)
  faces$track_id <- c(1L, 2L, 1L)
  attr(faces, "settings") <- list(type = "video", face_detector = "yunet")

  paths <- draw_faces(faces, dir = dir, plot = FALSE)
  expect_equal(basename(paths), c("v-frame-0-faces.png", "v-frame-1-faces.png"))
  expect_equal(vapply(calls$list, `[[`, "", "image"), c("frame-0.jpg", "frame-25.jpg"))
  expect_equal(calls$list[[1]]$labels, c("track 1, p1", "track 2, p2"))
  # The frames extracted again for drawing are removed
  expect_true(cleaned)
})

# ---------------------------------------------------------------------------
# te_faces in Python
# ---------------------------------------------------------------------------

test_that("the YouTube download picks progressive, else H.264 video-only MP4", {
  skip_without_face_python()
  te_faces <- .te_faces_py()
  # Fake pytubefix streams: only the attributes pick_youtube_stream() reads
  streams <- function(...) {
    rows <- list(...)
    types <- reticulate::import("types")
    lapply(rows, function(r) do.call(types$SimpleNamespace, r))
  }
  stream <- function(res, codec, progressive = FALSE, audio = progressive, subtype = "mp4") {
    list(resolution = res, video_codec = codec, subtype = subtype, is_progressive = progressive,
         includes_video_track = TRUE, includes_audio_track = audio)
  }
  pick <- function(...) {
    s <- te_faces$pick_youtube_stream(streams(...))
    if (is.null(s)) NULL else c(s$resolution, s$video_codec)
  }
  # A progressive stream is preferred, the highest one
  expect_equal(pick(stream("360p", "avc1.42001E", TRUE), stream("720p", "avc1.4D401F"),
                    stream("240p", "avc1.4D4015", TRUE)), c("360p", "avc1.42001E"))
  # Otherwise H.264 video-only up to 720p; AV1, WebM and audio-only are skipped
  adaptive <- list(stream("1080p", "avc1.640028"), stream("720p", "av01.0.05M.08"),
                   stream("720p", "vp9", subtype = "webm"), stream("480p", "avc1.4D401E"),
                   stream("720p", "avc1.4D401F"),
                   list(resolution = NULL, video_codec = NULL, subtype = "mp4",
                        is_progressive = FALSE, includes_video_track = FALSE,
                        includes_audio_track = TRUE))
  expect_equal(do.call(pick, adaptive), c("720p", "avc1.4D401F"))
  # All taller than 720p: the lowest
  expect_equal(pick(stream("2160p", "avc1.640033"), stream("1080p", "avc1.640028")),
               c("1080p", "avc1.640028"))
  expect_null(pick(stream("720p", "av01.0.05M.08")))
})

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

test_that("draw_faces draws the boxes on the example picture and clip", {
  skip_without_face_python()
  dir <- withr::local_tempdir()
  faces <- detect_faces(fixture("two-people.jpg"), identify = FALSE)
  expect_equal(nrow(faces), 2)
  path <- draw_faces(faces, dir = dir, plot = FALSE)
  expect_equal(basename(path), "two-people-faces.png")
  # The middle of each box's left edge has the face's colour (BGR in OpenCV)
  drawn <- reticulate::import("cv2")$imread(path)
  for (k in 1:2) {
    pixel <- drawn[round(faces$y[k] + faces$height[k] / 2) + 1, round(faces$x[k]) + 1, ]
    expect_equal(as.numeric(pixel), rev(.te_face_palette[[k]]))
  }

  # 20 frames, 0.6 s apart, cover the whole clip
  video <- detect_faces(fixture("two-people.mp4"), nframes = 20, identify = FALSE)
  expect_setequal(unique(video$track_id), 1:3)
  before <- list.files(tempdir(), pattern = "^te_frames_")
  paths <- draw_faces(video, max_images = 3, dir = dir, plot = FALSE)
  expect_length(paths, 3)
  expect_true(all(file.exists(paths)))
  expect_equal(list.files(tempdir(), pattern = "^te_frames_"), before)
})
