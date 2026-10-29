# transformer_scores() with Python mocked: which model the pipeline loads.
# These tests use no Python and no network.

# Records the arguments of each pipeline() call. The classifier gives 0.7 to
# the first class.
mock_pipeline <- function(env = parent.frame()) {
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
    py_available = function(...) FALSE,
    .package = "reticulate",
    .env = env
  )
  seen
}

test_that("transformer_scores() loads a built-in name from local_model_path", {
  seen <- mock_pipeline()
  dir <- withr::local_tempdir()
  msgs <- capture_messages(
    out <- transformer_scores("x", c("a", "b"), transformer = "cross-encoder-distilroberta",
                              device = "cpu", keep_in_env = FALSE, local_model_path = dir)
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
                       device = "cpu", keep_in_env = FALSE)
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
                       device = "cpu", keep_in_env = FALSE, local_model_path = missing_dir),
    "local_model_path directory does not exist"
  )
  expect_length(seen$calls, 0)
})
