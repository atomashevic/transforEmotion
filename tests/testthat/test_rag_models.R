# Model lists and structured-output gating in rag() / rag_sentemo(), without Python

test_that("Qwen 2.5 is a known rag() transformer with structured outputs", {
  choices <- eval(formals(rag)$transformer)
  expect_true(all(c("Qwen2.5-1.5B", "Qwen2.5-3B", "Gemma3-1B", "Gemma3-4B") %in% choices))
  expect_true(all(c("qwen2.5-1.5b", "qwen2.5-3b", "gemma3-1b", "gemma3-4b") %in% .te_structured_llms))
  expect_false("tinyllama" %in% .te_structured_llms)
  expect_true(all(.te_structured_llms %in% tolower(choices)))
  expect_true(all(c("Qwen2.5-1.5B", "Qwen2.5-3B") %in% eval(formals(rag_sentemo)$transformer)))
})

test_that("response_cleanup leaves Qwen 2.5 answers unchanged", {
  expect_identical(response_cleanup("  {\"label\": \"joy\"} ", "qwen2.5-3b"), "{\"label\": \"joy\"}")
  expect_identical(response_cleanup("x", "qwen2.5-1.5b"), "x")
})

test_that("rag_sentemo() warns and falls back to text for a model without structured outputs", {
  # rag() is stubbed so that no Python is needed; the warning is raised before it runs
  local_mocked_bindings(rag = function(...) list(...)$output)
  expect_warning(out <- rag_sentemo("a text", transformer = "TinyLLAMA", output = "table"),
                 "Structured outputs")
  expect_identical(out, "text")
  expect_silent(out <- rag_sentemo("a text", transformer = "Qwen2.5-3B", output = "table"))
  expect_identical(out, "table")
})

test_that("rag() and rag_sentemo() offer no Qwen3 model", {
  # Qwen3 needs transformers >= 4.51, which cannot be installed together with
  # the RAG stack (llama-index-core 0.10 and huggingface-hub < 0.24), and the
  # Qwen/Qwen3-*-Instruct repositories named before do not exist on the Hub
  for (fn in list(rag, rag_sentemo)) {
    expect_false(any(grepl("^qwen3", tolower(eval(formals(fn)$transformer)))))
  }
  expect_false(any(grepl("Qwen/Qwen3", deparse(body(rag)), fixed = TRUE)))
  expect_false(any(grepl("qwen3", deparse(body(response_cleanup)), fixed = TRUE)))
  expect_false(any(grepl("qwen3", deparse(body(auto_device)), fixed = TRUE)))
})

test_that("a Qwen3 transformer stops with the reason and the Qwen 2.5 choices", {
  # The check runs before Python starts and before match.arg()
  local_mocked_bindings(.te_require = function(...) stop("Python was started"))
  for (value in c("Qwen3-1.7B", "qwen3-0.6b", "QWEN3-4B")) {
    expect_error(rag("a text", transformer = value), "Qwen2.5-1.5B\" or \"Qwen2.5-3B\"", fixed = TRUE)
    expect_error(rag_sentemo("a text", transformer = value), "Qwen2.5-1.5B\" or \"Qwen2.5-3B\"", fixed = TRUE)
  }
  expect_error(rag("a text", transformer = "Qwen3-1.7B"), "repository IDs that transforEmotion used did not exist")
  expect_error(rag("a text", transformer = "Qwen3-1.7B"), "transformers >= 4.51", fixed = TRUE)
  # Other unknown values still get the match.arg() error
  expect_error(rag_sentemo("a text", transformer = "Llama-3"), "should be one of")
})

test_that("rag() checks the transformers version before it loads Gemma 3", {
  calls <- character()
  version <- "4.46.3"
  local_mocked_bindings(
    .te_require = function(...) invisible(TRUE),
    .te_use_nltk_cache = function(...) invisible(TRUE),
    te_validate_modern_llama_index = function(...) invisible(TRUE),
    .te_py_pkg_version = function(package) version,
    release_llm = function(...) calls <<- c(calls, "release"),
    auto_device = function(device, ...) "cpu",
    setup_gemma3 = function(...) {
      calls <<- c(calls, "load")
      stop("Gemma 3 loaded")
    }
  )
  env <- new.env()
  env$llama_index <- list()

  # The managed environment: stop before the previous model is freed and
  # before any download
  for (value in c("Gemma3-1B", "Gemma3-4B")) {
    expect_error(rag("a text", transformer = value, envir = env),
                 "Gemma 3 needs transformers >= 4.50, and this Python environment has transformers 4.46.3",
                 fixed = TRUE)
  }
  expect_error(rag("a text", transformer = "Gemma3-1B", envir = env), "TRANSFOREMOTION_PYTHON")
  expect_length(calls, 0)

  # A fixed environment with a newer transformers loads the model
  version <- "4.50.0"
  expect_error(rag("a text", transformer = "Gemma3-4B", envir = env), "Gemma 3 loaded")
  expect_identical(calls, c("release", "load"))
})

test_that("the Gemma 3 check reads the transformers version from the package metadata", {
  local_mocked_bindings(
    import = function(module, ...) {
      expect_identical(module, "importlib.metadata")
      list(version = function(package) if (identical(package, "transformers")) "4.46.3" else stop("not found"))
    },
    .package = "reticulate"
  )
  expect_identical(.te_py_pkg_version("transformers"), "4.46.3")
  expect_identical(.te_py_pkg_version("missing-package"), NA_character_)
  expect_error(.te_check_gemma3_transformers(), "4.46.3")

  # Development builds count as their release, and an unknown version is
  # left to the loader
  for (version in c("4.50.0.dev0", "4.57.1", "5.0.0rc1", NA_character_)) {
    local_mocked_bindings(.te_py_pkg_version = function(package) version)
    expect_true(.te_check_gemma3_transformers())
  }
  local_mocked_bindings(.te_py_pkg_version = function(package) "4.49.0")
  expect_error(.te_check_gemma3_transformers(), "4.49.0")
})

test_that("every Hugging Face repository that rag() loads exists on the Hub", {
  # Online check, not run on CRAN: set TE_ONLINE_TESTS=true to run it
  skip_on_cran()
  skip_if_not(identical(Sys.getenv("TE_ONLINE_TESTS"), "true"), "TE_ONLINE_TESTS is not true")
  skip_if_offline("huggingface.co")
  code <- c(deparse(body(rag)), deparse(body(setup_tinyllama)))
  # No lookbehind: PCRE2 before 10.43 (Ubuntu 24.04, Debian 12) rejects a
  # lookbehind whose alternatives differ in length
  found <- unlist(regmatches(code, gregexpr('(model_name|tokenizer_name|repo) = "[^"]+"', code)))
  repos <- unique(sub('^[a-z_]+ = "([^"]+)"$', "\\1", found))
  expect_gt(length(repos), 3)
  for (repo in repos) {
    # The model API answers 200 for public and gated repositories and 401
    # for a repository that does not exist
    res <- httr::GET(paste0("https://huggingface.co/api/models/", repo))
    expect_identical(httr::status_code(res), 200L, info = repo)
  }
})

test_that("every internal setup_*() helper is reachable from rag()", {
  # Guards against model setup code left behind when a model leaves the
  # rag() choices (setup_orca2() and five other helpers had no caller in
  # 0.1.7 to 0.1.9). All internal setup_*() helpers are in rag.R.
  ns <- asNamespace("transforEmotion")
  helpers <- setdiff(grep("^setup_", ls(ns), value = TRUE),
                     getNamespaceExports("transforEmotion"))
  reached <- "rag"
  repeat {
    called <- unique(unlist(lapply(reached, function(f) all.names(body(get(f, envir = ns))))))
    new <- setdiff(intersect(called, helpers), reached)
    if (!length(new)) break
    reached <- c(reached, new)
  }
  expect_true(length(helpers) > 0)
  expect_identical(sort(setdiff(helpers, reached)), character(0))
})

test_that("auto_device() and response_cleanup() name only rag() choices", {
  # Both dispatch on the lower-case rag() transformer. auto_device() needs a
  # memory entry for every choice; without one, it stops with "argument is of
  # length zero" for device "auto" or "cpu"
  switch_keys <- function(fn) {
    keys <- character()
    walk <- function(e) {
      if (!is.call(e)) return(invisible())
      if (identical(e[[1]], as.name("switch")) && identical(e[[2]], as.name("transformer"))) {
        keys <<- c(keys, setdiff(names(e)[-(1:2)], ""))
      }
      for (i in seq_along(e)[-1]) if (!identical(e[[i]], quote(expr = ))) walk(e[[i]])
    }
    walk(body(fn))
    keys
  }
  choices <- tolower(eval(formals(rag)$transformer))
  expect_setequal(switch_keys(auto_device), choices)
  cleanup <- switch_keys(response_cleanup)
  expect_true(length(cleanup) > 0)
  expect_true(all(cleanup %in% choices), info = paste(setdiff(cleanup, choices), collapse = ", "))
})

test_that("setup_hf_llm() uses one set of sampling defaults and applies user values", {
  # HuggingFaceLLM is stubbed, so no Python is needed
  args <- NULL
  local_mocked_bindings(
    get_huggingface_llm = function(llama_index) function(...) args <<- list(...),
    set_default_embedding = function(...) invisible(NULL)
  )
  llama_index <- list(core = list(Settings = new.env()))
  build <- function(model, ...) {
    setup_hf_llm(llama_index, "a prompt", "cpu", model_name = model, tokenizer_name = model,
                 context_window = 32000L, ...)
    args
  }
  for (model in c("google/gemma-3-1b-it", "Qwen/Qwen2.5-3B-Instruct", "ministral/Ministral-3b-instruct")) {
    expect_identical(build(model)$generate_kwargs, list(temperature = 0.1, do_sample = TRUE), info = model)
  }
  # max_new_tokens goes to the constructor only
  out <- build("Qwen/Qwen2.5-3B-Instruct", temperature = 0.7, top_p = 0.9, max_new_tokens = 64)
  expect_identical(out$generate_kwargs, list(temperature = 0.7, do_sample = TRUE, top_p = 0.9))
  expect_identical(out$max_new_tokens, 64L)
  # Greedy decoding drops the sampling parameters
  out <- build("Qwen/Qwen2.5-3B-Instruct", do_sample = FALSE, temperature = 0.7, top_p = 0.9)
  expect_identical(out$generate_kwargs, list(do_sample = FALSE))
  expect_null(out$max_new_tokens)
  expect_identical(llama_index$core$Settings$context_window, 32000L)
})
