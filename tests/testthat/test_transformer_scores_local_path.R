# transformer_scores() with Python mocked: which model the pipeline loads.
# These tests use no Python and no network.

# Records the arguments of each pipeline() call. The classifier gives 0.7 to
# the first class. With py_ready = TRUE, Python counts as running and a kept
# classifier counts as valid, so transformer_scores() can reuse it.
mock_pipeline <- function(py_ready = FALSE, env = parent.frame()) {
  seen <- new.env()
  seen$calls <- list()
  pipeline <- function(task, ...) {
    seen$calls <- c(seen$calls, list(list(...)))
    function(x, classes, multi_label) list(labels = classes, scores = c(0.7, 0.3))
  }
  local_mocked_bindings(ensure_te_py_env = function() invisible(TRUE), .env = env)
  local_mocked_bindings(
    py_run_string = function(...) invisible(NULL),
    import = function(module, ...) list(pipeline = pipeline),
    py_available = function(...) py_ready,
    is_py_object = function(x) TRUE,
    py_is_null_xptr = function(x) FALSE,
    .package = "reticulate",
    .env = env
  )
  seen
}

# The model that each pipeline() call loaded
loaded_models <- function(seen) vapply(seen$calls, function(a) a$model, "")

# The tests keep classifiers in their own environment, so the classifiers in
# the global environment of the session stay as they are.

test_that("transformer_scores() loads a built-in name from local_model_path", {
  seen <- mock_pipeline()
  dir <- withr::local_tempdir()
  msgs <- capture_messages(
    out <- transformer_scores("x", c("a", "b"), transformer = "cross-encoder-distilroberta",
                              device = "cpu", keep_in_env = FALSE, local_model_path = dir,
                              envir = new.env(parent = emptyenv()))
  )
  expect_match(msgs, "Using local model from", all = FALSE)
  expect_length(seen$calls, 1)
  expect_equal(seen$calls[[1]]$model, dir)
  expect_true(seen$calls[[1]]$local_files_only)
  expect_equal(out[["x"]], c(a = 0.7, b = 0.3))
})

test_that("transformer_scores() loads a built-in name from the Hub without local_model_path", {
  seen <- mock_pipeline()
  suppressMessages(
    transformer_scores("x", c("a", "b"), transformer = "cross-encoder-distilroberta",
                       device = "cpu", keep_in_env = FALSE, envir = new.env(parent = emptyenv()))
  )
  expect_length(seen$calls, 1)
  expect_equal(seen$calls[[1]]$model, "cross-encoder/nli-distilroberta-base")
  expect_null(seen$calls[[1]]$local_files_only)
})

test_that("transformer_scores() stops for a built-in name and a local_model_path that does not exist", {
  seen <- mock_pipeline()
  missing_dir <- file.path(withr::local_tempdir(), "no-model")
  expect_error(
    transformer_scores("x", c("a", "b"), transformer = "facebook-bart",
                       device = "cpu", keep_in_env = FALSE, local_model_path = missing_dir,
                       envir = new.env(parent = emptyenv())),
    "local_model_path directory does not exist"
  )
  expect_length(seen$calls, 0)
})

test_that("transformer_scores() reuses a kept classifier only for the same local_model_path", {
  seen <- mock_pipeline(py_ready = TRUE)
  cache <- new.env(parent = emptyenv())
  dir <- withr::local_tempdir()
  score <- function(...) {
    suppressMessages(transformer_scores("x", c("a", "b"), device = "cpu", envir = cache, ...))
  }
  # No transformer given, so the classifier is kept as "cross-encoder-distilroberta"
  score(local_model_path = dir)
  score(local_model_path = paste0(dir, "/"))
  expect_equal(loaded_models(seen), dir)
  expect_true(exists("cross-encoder-distilroberta", envir = cache))
  # A local model does not answer a call without local_model_path
  score()
  score()
  expect_equal(loaded_models(seen), c(dir, "cross-encoder/nli-distilroberta-base"))
  # A Hub model does not answer a call with local_model_path
  score(local_model_path = dir)
  expect_equal(loaded_models(seen), c(dir, "cross-encoder/nli-distilroberta-base", dir))
})

test_that("vad_scores() with local_model_path loads the local model once", {
  seen <- mock_pipeline(py_ready = TRUE)
  cache <- new.env(parent = emptyenv())
  dir <- withr::local_tempdir()
  out <- suppressMessages(
    vad_scores("I am happy", input_type = "text", dimensions = c("valence", "arousal"),
               device = "cpu", envir = cache, local_model_path = dir)
  )
  expect_equal(loaded_models(seen), dir)
  expect_equal(unname(out$valence), 0.7)
  expect_equal(unname(out$arousal), 0.7)
  # A later call with model = "auto" and no local_model_path loads the Hub model
  suppressMessages(
    vad_scores("I am happy", input_type = "text", dimensions = "valence",
               device = "cpu", envir = cache)
  )
  expect_equal(loaded_models(seen), c(dir, "cross-encoder/nli-distilroberta-base"))
})
