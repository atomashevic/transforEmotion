# sentence_similarity() without Python: reticulate::import() returns a fake
# sentence_transformers module, and a fake classifier returns fixed embeddings.

# An environment, so that to() changes the device as it does in PyTorch
fake_classifier <- function(device = "cpu") {
  classifier <- new.env()
  classifier$device <- list(type = device)
  classifier$moves <- character()
  classifier$to <- function(device) {
    classifier$moves <- c(classifier$moves, device)
    classifier$device <- list(type = device)
    invisible(classifier)
  }
  # Sentence i has the embedding (1, i - 1)
  classifier$encode <- function(sentences, ...) cbind(1, seq_along(sentences) - 1)
  classifier
}

test_that("sentence_similarity asks for comparison_text without a bell character", {
  local_mocked_bindings(ensure_te_py_env = function(...) invisible(TRUE))
  err <- expect_error(sentence_similarity("a sentence"), "comparison_text")
  expect_false(7L %in% utf8ToInt(conditionMessage(err)))
  expect_match(conditionMessage(err), '"a similar sentence", "a random sentence"', fixed = TRUE)
})

test_that("sentence_similarity points a failed model at sentence-similarity models", {
  local_mocked_bindings(ensure_te_py_env = function(...) invisible(TRUE))
  fake_st <- list(SentenceTransformer = function(...) {
    stop("Tokenizer class FakeTokenizer does not exist or is not currently imported.")
  })
  local_mocked_bindings(import = function(module, ...) fake_st, .package = "reticulate")
  err <- expect_error(
    sentence_similarity("a sentence", comparison_text = "another sentence",
                        transformer = "no-such-org/no-such-model", device = "cpu"),
    "exists but does not have a working pipeline"
  )
  expect_match(conditionMessage(err), "pipeline_tag=sentence-similarity", fixed = TRUE)
  expect_no_match(conditionMessage(err), "zero-shot-classification", fixed = TRUE)
})

test_that("sentence_similarity retries without device after a device_map error", {
  local_mocked_bindings(ensure_te_py_env = function(...) invisible(TRUE))
  retry_works <- TRUE
  fake_st <- list(SentenceTransformer = function(model, device = NULL) {
    if (!is.null(device)) stop("got an unexpected keyword argument 'device_map'")
    if (!retry_works) stop("The second load failed")
    fake_classifier()
  })
  local_mocked_bindings(import = function(module, ...) fake_st, .package = "reticulate")

  sims <- suppressMessages(sentence_similarity(
    "a", comparison_text = "b", transformer = "org/model", device = "cpu",
    keep_in_env = FALSE
  ))
  expect_equal(sims[1, 1], 1 / sqrt(2))

  # A failed retry stops with the error of the retry
  retry_works <- FALSE
  err <- expect_error(
    sentence_similarity("a", comparison_text = "b", transformer = "org/model",
                        device = "cpu", keep_in_env = FALSE),
    "The second load failed"
  )
  expect_no_match(conditionMessage(err), "object 'classifier' not found", fixed = TRUE)
})

test_that("sentence_similarity keeps the module with a classifier reused from envir", {
  local_mocked_bindings(ensure_te_py_env = function(...) invisible(TRUE))
  fake_st <- list(SentenceTransformer = function(...) stop("The classifier was loaded again"))
  local_mocked_bindings(import = function(module, ...) fake_st, .package = "reticulate")
  classifier <- fake_classifier("cpu")
  e <- new.env()
  assign("all_minilm_l6", classifier, envir = e)

  sims <- suppressMessages(sentence_similarity("a", comparison_text = "b", envir = e))
  expect_equal(sims[1, 1], 1 / sqrt(2))
  expect_identical(e$sentence_transformers, fake_st)
  expect_identical(e$all_minilm_l6, classifier)
})

test_that("sentence_similarity moves a reused classifier to a device that the user sets", {
  local_mocked_bindings(ensure_te_py_env = function(...) invisible(TRUE))
  fake_st <- list(SentenceTransformer = function(...) stop("The classifier was loaded again"))
  local_mocked_bindings(import = function(module, ...) fake_st, .package = "reticulate")
  classifier <- fake_classifier("cpu")
  e <- new.env()
  assign("all_minilm_l6", classifier, envir = e)

  # The same device and "auto" keep the classifier on its current device,
  # so "auto" after "cuda" does not move it back to "cpu"
  devices <- character()
  for (device in c("cuda", "cuda", "auto", "cpu")) {
    suppressMessages(sentence_similarity(
      "a", comparison_text = "b", device = device, envir = e, keep_in_env = FALSE
    ))
    devices <- c(devices, classifier$device$type)
  }
  expect_equal(devices, c("cuda", "cuda", "cuda", "cpu"))
  expect_equal(classifier$moves, c("cuda", "cpu"))
})
