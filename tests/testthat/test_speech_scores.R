# speech_scores(). The tests at the top check arguments and the result frame
# without Python; the tests at the bottom run the te_audio Python module and
# need TRANSFOREMOTION_PYTHON to point at a Python environment with PyAV.

test_that("speech_scores checks its arguments before starting Python", {
  local_mocked_bindings(.te_require = function(...) stop("Python was started"))
  wav <- withr::local_tempfile(fileext = ".wav")
  file.create(wav)
  classes <- c("angry", "calm")

  expect_error(speech_scores(c(wav, wav), classes), "one file path")
  expect_error(speech_scores("no-such-file.wav", classes), "does not exist")
  expect_error(speech_scores("https://example.com/a.wav", classes), "YouTube video URL")
  expect_error(speech_scores(wav, "angry"), "at least 2")
  expect_error(speech_scores(wav, c("angry", "angry")), "unique")
  expect_error(speech_scores(wav, c("angry", NA)), "empty or NA")
  expect_error(speech_scores(wav, c("angry", "start")), "cannot be named start")
  expect_error(speech_scores(wav, classes, nsegments = 0), "positive whole number")
  expect_error(speech_scores(wav, classes, nsegments = 2.5), "positive whole number")
  expect_error(speech_scores(wav, classes, segment_length = 0), "at most 10")
  expect_error(speech_scores(wav, classes, segment_length = 12), "at most 10")
  expect_error(speech_scores(wav, classes, start = -1), "start must be")
  expect_error(speech_scores(wav, classes, start = 5, end = 3), "greater than start")
  expect_error(speech_scores(wav, classes, local_model_path = tempfile()), "does not exist")
  expect_error(speech_scores(wav, classes, model = ""), "laion-clap")
})

test_that("only http(s) URLs on YouTube hosts count as YouTube videos", {
  yes <- c("https://www.youtube.com/watch?v=abc", "http://youtube.com/watch?v=abc",
           "https://youtu.be/abc", "HTTPS://WWW.YouTube.com/watch?v=abc",
           "https://m.youtube.com/watch?v=abc")
  no <- c("/data/youtube/interview.wav", "youtube.wav", "https://example.com/youtube.wav",
          "https://notyoutube.com/watch?v=abc", "https://youtube.com.example.org/a")
  expect_true(all(.te_is_youtube_url(yes)))
  expect_false(any(.te_is_youtube_url(no)))

  # A local file in a folder named youtube is read, not downloaded
  dir <- file.path(withr::local_tempdir(), "youtube")
  dir.create(dir)
  wav <- file.path(dir, "interview.wav")
  file.create(wav)
  required <- NULL
  local_mocked_bindings(
    .te_require = function(features) required <<- features,
    .te_audio_py = function() list(
      download_youtube_audio = function(...) stop("downloaded"),
      speech_analyze = function(...) list(start = list(0), end = list(5),
                                          scores = matrix(c(0.5, 0.5), nrow = 1))
    )
  )
  expect_equal(nrow(speech_scores(wav, c("a", "b"))), 1)
  expect_equal(required, "audio")
})

test_that("setup_modules downloads the CLAP model only with the audio extra", {
  clap <- "laion/larger_clap_music_and_speech"
  expect_false(clap %in% .te_setup_models(character()))
  expect_false(clap %in% .te_setup_models("rag"))
  expect_true(clap %in% .te_setup_models("audio"))
  expect_true(all(.te_default_models %in% .te_setup_models("audio")))
})

test_that("speech_scores maps the model name and builds one row per segment", {
  expect_equal(.te_speech_model_id("laion-clap"), "laion/larger_clap_music_and_speech")
  expect_equal(.te_speech_model_id("laion/larger_clap_general"), "laion/larger_clap_general")

  wav <- withr::local_tempfile(fileext = ".wav")
  file.create(wav)
  sent <- NULL
  local_mocked_bindings(
    .te_require = function(...) invisible(TRUE),
    .te_audio_py = function() list(speech_analyze = function(...) {
      sent <<- list(...)
      list(start = list(0, 5), end = list(5, 8),
           scores = matrix(c(0.9, 0.2, 0.1, 0.8), nrow = 2))
    })
  )
  res <- speech_scores(wav, c("a person speaking angrily", "calm"), nsegments = 2)
  expect_equal(names(res), c("segment", "start", "end", "a person speaking angrily", "calm"))
  expect_equal(res$segment, 1:2)
  expect_equal(res$end, c(5, 8))
  expect_equal(res$calm, c(0.1, 0.8))
  expect_equal(sent$model_name, "laion/larger_clap_music_and_speech")
  expect_identical(sent$nsegments, 2L)
  expect_identical(sent$end, -1)
  expect_type(sent$labels, "list")
})

test_that("a result without segments is an empty frame with the class columns", {
  res <- .te_speech_frame(list(start = list(), end = list(), scores = matrix(0, 0, 2)),
                          c("a", "b"))
  expect_equal(nrow(res), 0)
  expect_equal(names(res), c("segment", "start", "end", "a", "b"))
})

# ---------------------------------------------------------------------------
# te_audio in Python
# ---------------------------------------------------------------------------

skip_without_audio_python <- function() {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON")),
          "TRANSFOREMOTION_PYTHON is not set")
  ensure_te_py_env()
  skip_if_not(reticulate::py_module_available("av"), "PyAV is not available")
}

# A mono 16-bit WAV file of `seconds` seconds: a 440 Hz tone
write_tone <- function(path, seconds, rate = 16000L) {
  reticulate::py_run_string(sprintf(paste(
    "import wave, numpy as np",
    "t = np.arange(int(%f * %d)) / %d",
    "x = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype('<i2')",
    "with wave.open(%s, 'wb') as w:",
    "    w.setnchannels(1); w.setsampwidth(2); w.setframerate(%d); w.writeframes(x.tobytes())",
    sep = "\n"), seconds, rate, rate, encodeString(path, quote = "'"), rate))
  path
}

test_that("segments are consecutive, or spread evenly without overlap", {
  skip_without_audio_python()
  plan <- function(...) {
    do.call(rbind, lapply(.te_audio_py()$plan_segments(...), unlist))
  }
  # Consecutive: a last piece under one second is dropped
  expect_equal(plan(22.5, 100L, 5), cbind(c(0, 5, 10, 15, 20), c(5, 10, 15, 20, 22.5)))
  expect_equal(plan(20.5, 100L, 5)[, 2], c(5, 10, 15, 20))
  expect_equal(plan(30, 2L, 5), cbind(c(0, 5), c(5, 10)))
  # Shorter than one segment: the whole range
  expect_equal(plan(0.4, 10L, 5), cbind(0, 0.4))
  # Uniform: at most floor(duration / length) segments, first and last at the edges
  expect_equal(plan(60, 3L, 5, TRUE), cbind(c(0, 27.5, 55), c(5, 32.5, 60)))
  expect_equal(nrow(plan(12, 10L, 5, TRUE)), 2)
  expect_equal(plan(7, 10L, 5, TRUE), cbind(0, 5))
})

test_that("audio is decoded to mono at the requested rate and cut", {
  skip_without_audio_python()
  te_audio <- .te_audio_py()
  wav <- write_tone(withr::local_tempfile(fileext = ".wav"), 3)
  expect_equal(length(te_audio$load_audio(wav, 0, NULL, 48000L)), 3 * 48000, tolerance = 1e-3)
  expect_equal(length(te_audio$load_audio(wav, 1, 2, 48000L)), 48000, tolerance = 1e-3)
  expect_equal(length(te_audio$load_audio(wav, 5, NULL, 48000L)), 0)

  video <- system.file("extdata", "two-people.mp4", package = "transforEmotion")
  expect_error(te_audio$load_audio(video), "No audio stream")
  expect_error(te_audio$audio_duration(video), "No audio stream")
})

test_that("only the samples inside the windows are kept", {
  skip_without_audio_python()
  te_audio <- .te_audio_py()
  wav <- write_tone(withr::local_tempfile(fileext = ".wav"), 30)
  expect_equal(te_audio$audio_duration(wav), 30, tolerance = 1e-3)

  windows <- list(c(1, 2), c(14, 16.5), c(29, 31))
  kept <- te_audio$load_windows(wav, windows, 48000L)
  expect_equal(vapply(kept, length, 1), c(1, 2.5, 1) * 48000, tolerance = 1e-3)
  # Each window holds the same samples as decoding that range alone
  expect_equal(kept[[2]], te_audio$load_audio(wav, 14, 16.5, 48000L))
  expect_length(te_audio$load_windows(wav, list(), 48000L), 0)
})

test_that("speech_scores scores each segment with LAION CLAP", {
  skip_without_audio_python()
  # import() errors when the module is missing. The face and speech job
  # has OpenCV and PyAV only, so this stays skipped until CLAP is cached.
  skip_if_not(
    reticulate::py_module_available("huggingface_hub"),
    "huggingface_hub is not available"
  )
  hub <- reticulate::import("huggingface_hub")
  cached <- hub$try_to_load_from_cache("laion/larger_clap_music_and_speech", "config.json")
  skip_if(!is.character(cached), "laion/larger_clap_music_and_speech is not downloaded")

  wav <- write_tone(withr::local_tempfile(fileext = ".wav"), 12.4)
  classes <- c("a person speaking", "a pure tone", "music")
  res <- speech_scores(wav, classes, segment_length = 5)
  expect_equal(res$start, c(0, 5, 10), tolerance = 1e-6)
  expect_equal(res$end, c(5, 10, 12.4), tolerance = 1e-3)
  expect_equal(rowSums(res[classes]), rep(1, 3), tolerance = 1e-5)

  uniform <- speech_scores(wav, classes, nsegments = 2, segment_length = 2,
                           start = 1, end = 11, uniform = TRUE)
  expect_equal(uniform$start, c(1, 9), tolerance = 1e-6)
  # The same audio gets the same scores
  again <- speech_scores(wav, classes, nsegments = 1, segment_length = 5)
  expect_equal(unlist(again[1, classes]), unlist(res[1, classes]), tolerance = 1e-5)

  expect_error(speech_scores(wav, classes, start = 20), "No audio between")
})
