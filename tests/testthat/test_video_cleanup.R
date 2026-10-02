# Cleanup of the frames video_scores() writes: only files the call creates are
# removed. These tests use generated files and mock the Python call.

touch <- function(path, bytes = as.raw(1:4)) {
  writeBin(bytes, path)
  invisible(path)
}

# Writes the files yt_analyze() would write, then returns or fails
fake_yt_analyze <- function(save_dir, video_name, n, fail = FALSE, mp4 = FALSE) {
  function(expr) {
    for (i in seq_len(n) - 1L) {
      touch(file.path(save_dir, sprintf("%s-frame-%d.jpg", video_name, i)), as.raw(9))
    }
    if (mp4) touch(file.path(save_dir, paste0(video_name, ".mp4")), as.raw(9))
    if (fail) stop("simulated failure during classification")
    data.frame(a = rep(0.5, n), b = rep(0.5, n))
  }
}

run_video_scores <- function(save_dir, video_name = "temp", nframes = 3,
                             save_frames = FALSE, save_video = FALSE,
                             video = NULL, fail = FALSE, mp4 = FALSE) {
  if (is.null(video)) {
    video <- touch(file.path(withr::local_tempdir(.local_envir = parent.frame()), "clip.mp4"))
  }
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    without_hf_token = fake_yt_analyze(save_dir, video_name, nframes, fail, mp4)
  )
  local_mocked_bindings(source_python = function(...) invisible(NULL), .package = "reticulate")
  suppressMessages(video_scores(
    video, classes = c("a", "b"), nframes = nframes, face_selection = "none",
    save_dir = save_dir, video_name = video_name,
    save_frames = save_frames, save_video = save_video
  ))
}

test_that("cleanup keeps frames of videos whose names share a prefix", {
  dir <- withr::local_tempdir()
  other <- touch(file.path(dir, "temp-frame-other-frame-0.jpg"))
  longer <- touch(file.path(dir, "temp2-frame-0.jpg"))
  run_video_scores(dir, video_name = "temp")
  expect_true(file.exists(other))
  expect_true(file.exists(longer))
  expect_false(any(file.exists(file.path(dir, sprintf("temp-frame-%d.jpg", 0:2)))))
})

test_that("cleanup keeps unrelated files", {
  dir <- withr::local_tempdir()
  sentinels <- c("notes.txt", "photo.jpg", "temp-frame-x.jpg", "temp-frame-10.png",
                 "temp-frame-3.jpg.bak")
  for (f in sentinels) touch(file.path(dir, f))
  run_video_scores(dir, video_name = "temp", nframes = 3)
  expect_setequal(list.files(dir), sentinels)
})

test_that("cleanup keeps files that existed under a generated name", {
  dir <- withr::local_tempdir()
  # Frame 3 is beyond this call's frames; frame 1 is overwritten by it
  stale <- touch(file.path(dir, "temp-frame-3.jpg"))
  collision <- touch(file.path(dir, "temp-frame-1.jpg"))
  run_video_scores(dir, video_name = "temp", nframes = 3)
  expect_true(file.exists(stale))
  expect_true(file.exists(collision))
  expect_false(file.exists(file.path(dir, "temp-frame-0.jpg")))
  expect_false(file.exists(file.path(dir, "temp-frame-2.jpg")))
})

test_that("save_frames = TRUE keeps the frames and save_frames = FALSE removes them", {
  kept <- withr::local_tempdir()
  run_video_scores(kept, save_frames = TRUE)
  expect_setequal(list.files(kept), sprintf("temp-frame-%d.jpg", 0:2))

  removed <- withr::local_tempdir()
  run_video_scores(removed, save_frames = FALSE)
  expect_length(list.files(removed), 0)
})

test_that("a downloaded video is removed unless save_video = TRUE", {
  url <- "https://www.youtube.com/watch?v=placeholder"
  dir <- withr::local_tempdir()
  local_mocked_bindings(.te_require = function(...) invisible(TRUE))
  run_video_scores(dir, video = url, mp4 = TRUE)
  expect_false(file.exists(file.path(dir, "temp.mp4")))

  dir2 <- withr::local_tempdir()
  run_video_scores(dir2, video = url, mp4 = TRUE, save_video = TRUE)
  expect_true(file.exists(file.path(dir2, "temp.mp4")))
})

test_that("a local video in save_dir is never removed", {
  dir <- withr::local_tempdir()
  video <- touch(file.path(dir, "temp.mp4"))
  run_video_scores(dir, video = video)
  expect_true(file.exists(video))
})

test_that("cleanup works in paths with spaces and non-ASCII characters", {
  skip_if_not(isTRUE(l10n_info()[["UTF-8"]]), "needs a UTF-8 locale")
  dir <- file.path(withr::local_tempdir(), "dir with spaces ünïcødé")
  dir.create(dir)
  name <- "vidéo clip"
  other <- touch(file.path(dir, paste0(name, "-frame-other-frame-0.jpg")))
  run_video_scores(dir, video_name = name, nframes = 2)
  expect_identical(list.files(dir), basename(other))
})

test_that("frames are removed when processing fails", {
  dir <- withr::local_tempdir()
  other <- touch(file.path(dir, "temp-frame-other-frame-0.jpg"))
  expect_error(run_video_scores(dir, fail = TRUE), "simulated failure")
  expect_identical(list.files(dir), basename(other))
})

test_that("save_frames = TRUE keeps frames written before a failure", {
  dir <- withr::local_tempdir()
  expect_error(run_video_scores(dir, fail = TRUE, save_frames = TRUE), "simulated failure")
  expect_setequal(list.files(dir), sprintf("temp-frame-%d.jpg", 0:2))
})

test_that("the cleanup plan removes only what was created after it was made", {
  dir <- withr::local_tempdir()
  existing <- touch(file.path(dir, "temp-frame-0.jpg"))
  cleanup <- transforEmotion:::.te_video_cleanup_plan(dir, "temp", 2, FALSE, TRUE)
  touch(file.path(dir, "temp-frame-1.jpg"))
  removed <- cleanup()
  expect_identical(basename(removed), "temp-frame-1.jpg")
  expect_true(file.exists(existing))
  # Running it again (for example after an interrupt) is harmless
  expect_length(cleanup(), 0)
})

test_that("an interrupt during processing still runs the cleanup", {
  dir <- withr::local_tempdir()
  process <- function() {
    cleanup <- transforEmotion:::.te_video_cleanup_plan(dir, "temp", 2, FALSE, TRUE)
    on.exit(cleanup(), add = TRUE)
    touch(file.path(dir, "temp-frame-0.jpg"))
    signalCondition(structure(list(message = "", call = NULL),
                              class = c("interrupt", "condition")))
  }
  interrupted <- tryCatch({ process(); FALSE }, interrupt = function(e) TRUE)
  expect_true(interrupted)
  expect_length(list.files(dir), 0)
})
