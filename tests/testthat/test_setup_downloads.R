# The PyTorch build for the GPU, the NLTK data download, and the lock on
# huggingface_hub's symlink check

test_that("the CUDA build follows the compute capability of the GPU", {
  build <- transforEmotion:::.te_cuda_build_for
  # Blackwell: the cu126 build has no kernels for these GPUs
  expect_equal(build(12.0), "cu130")  # GeForce RTX 50 series
  expect_equal(build(10.0), "cu130")  # B200
  # Maxwell to Hopper keep the build with the oldest driver requirement;
  # cu130 cannot run below 7.5
  for (capability in c(5.0, 6.1, 7.0, 7.5, 8.6, 8.9, 9.0)) {
    expect_equal(build(capability), "cu126", info = capability)
  }
  # No GPU that nvidia-smi can report
  expect_equal(build(NA_real_), "cu126")
})

test_that("the CUDA build names the PyTorch wheels", {
  for (sysname in c("Linux", "Windows")) {
    gpu <- transforEmotion:::.te_torch_requirements(TRUE, sysname, "x86_64", cuda = "cu130")
    expect_match(gpu[1], "^torch @ https://download.pytorch.org/whl/cu130/torch-2\\.14\\.0%2Bcu130-cp312-cp312-")
    expect_match(gpu[2], "^torchvision @ https://download.pytorch.org/whl/cu130/torchvision-0\\.29\\.0%2Bcu130-")
  }
})

test_that("the CUDA build is detected once, and only for a GPU build", {
  state <- transforEmotion:::.te_py_state
  old <- state$cuda_build
  withr::defer(state$cuda_build <- old)
  state$cuda_build <- NULL

  calls <- 0
  local_mocked_bindings(.te_gpu_compute_capability = function() {
    calls <<- calls + 1
    12.0
  })
  # The CPU build does not run nvidia-smi
  transforEmotion:::.te_torch_requirements(FALSE, "Linux", "x86_64")
  expect_equal(calls, 0)
  # The CUDA build runs it once per session
  first <- transforEmotion:::.te_torch_requirements(TRUE, "Windows", "AMD64")
  second <- python_requirements(gpu = TRUE, sysname = "Linux", machine = "x86_64")
  expect_equal(calls, 1)
  expect_true(all(grepl("/whl/cu130/", c(first, second[1:2]), fixed = TRUE)))
})

# The state of the PyTorch build choice as after loading with no variable
# set, with py_require() recorded instead of run
local_build_state <- function(use_gpu, python_started = FALSE, env = parent.frame()) {
  state <- transforEmotion:::.te_py_state
  old <- mget(c("features", "use_gpu", "gpu_env", "cuda_build"), envir = state,
              ifnotfound = list(NULL))
  withr::defer(list2env(old, envir = state), envir = env)
  withr::local_envvar(TRANSFOREMOTION_FORCE_CPU = NA, TE_FORCE_CPU = NA,
                      TRANSFOREMOTION_USE_GPU = NA, TRANSFOREMOTION_PYTHON = NA,
                      .local_envir = env)
  state$features <- "core"
  state$use_gpu <- use_gpu
  state$cuda_build <- "cu130"
  state$gpu_env <- transforEmotion:::.te_gpu_env()

  calls <- new.env()
  calls$require <- list()
  local_mocked_bindings(
    py_available = function(...) python_started,
    py_require = function(packages, ..., action = "add") {
      calls$require[[length(calls$require) + 1]] <- list(action = action, packages = packages)
      invisible(NULL)
    },
    .package = "reticulate", .env = env
  )
  calls
}

test_that("TE_FORCE_CPU set after loading switches the build before Python starts", {
  calls <- local_build_state(use_gpu = TRUE)
  state <- transforEmotion:::.te_py_state

  # No variable changed
  expect_false(transforEmotion:::.te_follow_gpu_env())
  expect_length(calls$require, 0)

  # Set after loading, as after an earlier library() call in the session
  Sys.setenv(TE_FORCE_CPU = "1")
  expect_true(transforEmotion:::.te_follow_gpu_env())
  expect_false(state$use_gpu)
  expect_equal(calls$require[[1]]$action, "remove")
  expect_true(all(grepl("cu130", calls$require[[1]]$packages, fixed = TRUE)))
  expect_equal(calls$require[[2]]$action, "add")
  expect_false(any(grepl("cu1", calls$require[[2]]$packages, fixed = TRUE)))

  # Each change is acted on once
  expect_false(transforEmotion:::.te_follow_gpu_env())
  expect_length(calls$require, 2)

  # The other direction
  Sys.unsetenv("TE_FORCE_CPU")
  Sys.setenv(TRANSFOREMOTION_USE_GPU = "1")
  expect_true(transforEmotion:::.te_follow_gpu_env())
  expect_true(state$use_gpu)
})

test_that("a variable set after Python started gives a warning once", {
  calls <- local_build_state(use_gpu = TRUE, python_started = TRUE)
  Sys.setenv(TE_FORCE_CPU = "1")
  expect_warning(
    transforEmotion:::.te_follow_gpu_env(),
    "TE_FORCE_CPU was set after Python started with the CUDA build"
  )
  expect_true(transforEmotion:::.te_py_state$use_gpu)
  expect_length(calls$require, 0)
  expect_silent(transforEmotion:::.te_follow_gpu_env())

  # A variable that agrees with the running build gives no warning
  Sys.setenv(TE_FORCE_CPU = "", TRANSFOREMOTION_USE_GPU = "1")
  expect_silent(transforEmotion:::.te_follow_gpu_env())
})

test_that("the build that setup_cache() sets is not undone by an earlier variable", {
  calls <- local_build_state(use_gpu = TRUE)
  Sys.setenv(TE_FORCE_CPU = "1")
  # setup_cache(gpu = TRUE) after the variable was set
  transforEmotion:::.te_set_torch_flavor(TRUE)
  expect_false(transforEmotion:::.te_follow_gpu_env())
  expect_true(transforEmotion:::.te_py_state$use_gpu)
  expect_length(calls$require, 0)
})

test_that("the compute capability comes from nvidia-smi", {
  skip_on_os("windows")  # the fake nvidia-smi is a shell script
  bin <- withr::local_tempdir()
  fake <- function(lines, status = 0) {
    writeLines(c("#!/bin/sh", paste0("echo '", lines, "'"), paste("exit", status)),
               file.path(bin, "nvidia-smi"))
    Sys.chmod(file.path(bin, "nvidia-smi"), "755")
  }
  withr::local_path(bin, action = "prefix")

  fake(c("12.0", "8.6"))
  expect_equal(transforEmotion:::.te_gpu_compute_capability(), c(12.0, 8.6))
  # A driver older than R510 does not know the field
  fake('Field "compute_cap" is not a valid field to query.', status = 2)
  expect_identical(transforEmotion:::.te_gpu_compute_capability(), NA_real_)
})

# NLTK data packages as zip files in a folder, served with a file:// URL
local_nltk_source <- function(env = parent.frame()) {
  skip_if(!nzchar(Sys.which(Sys.getenv("R_ZIPCMD", "zip"))), "no zip program")
  src <- withr::local_tempdir(.local_envir = env)
  for (pkg in names(transforEmotion:::.te_nltk_packages)) {
    folder <- file.path(src, transforEmotion:::.te_nltk_packages[[pkg]])
    dir.create(file.path(folder, pkg), recursive = TRUE, showWarnings = FALSE)
    writeLines("data", file.path(folder, pkg, "english"))
    withr::with_dir(folder, utils::zip(paste0(pkg, ".zip"), pkg, flags = "-rq"))
    unlink(file.path(folder, pkg), recursive = TRUE)
  }
  src
}

file_url <- function(path) {
  paste0("file://", if (.Platform$OS.type == "windows") "/", normalizePath(path, winslash = "/"))
}

test_that("NLTK data is downloaded in the layout of nltk.download()", {
  src <- local_nltk_source()
  dir <- withr::local_tempdir()

  expect_message(
    ok <- transforEmotion:::.te_download_nltk_data(dir, base_url = file_url(src)),
    "stopwords, punkt, punkt_tab"
  )
  expect_true(ok)
  for (path in c("corpora/stopwords/english", "corpora/stopwords.zip",
                 "tokenizers/punkt/english", "tokenizers/punkt.zip",
                 "tokenizers/punkt_tab/english", "tokenizers/punkt_tab.zip")) {
    expect_true(file.exists(file.path(dir, path)), info = path)
  }
  expect_length(list.files(dir, "\\.part$", recursive = TRUE, include.dirs = TRUE), 0)

  # Data that is there is not downloaded again
  unlink(src, recursive = TRUE)
  expect_silent(ok <- transforEmotion:::.te_download_nltk_data(dir, base_url = file_url(src)))
  expect_true(ok)
})

test_that("a failed NLTK download names the file, does not warn, and leaves no package folder", {
  src <- local_nltk_source()
  dir <- withr::local_tempdir()
  unlink(file.path(src, "tokenizers", "punkt_tab.zip"))
  writeLines("not a zip file", file.path(src, "tokenizers", "punkt.zip"))

  messages <- character()
  ok <- withCallingHandlers(
    transforEmotion:::.te_download_nltk_data(dir, base_url = file_url(src)),
    message = function(m) {
      messages <<- c(messages, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_false(ok)
  # The download notice, then the failure
  expect_length(messages, 2)
  expect_match(messages[2], "rag() works without it", fixed = TRUE)
  expect_match(messages[2], "tokenizers/punkt.zip -> ", fixed = TRUE)
  expect_match(messages[2], "tokenizers/punkt_tab.zip -> ", fixed = TRUE)
  expect_false(grepl("stopwords", messages[2], fixed = TRUE))
  expect_true(dir.exists(file.path(dir, "corpora", "stopwords")))
  expect_false(dir.exists(file.path(dir, "tokenizers", "punkt")))
  expect_false(dir.exists(file.path(dir, "tokenizers", "punkt_tab")))
  expect_length(list.files(dir, "\\.part$", recursive = TRUE, include.dirs = TRUE), 0)
})

test_that("the NLTK data folder is filled only when HF_HUB_OFFLINE is not set", {
  calls <- 0
  local_mocked_bindings(.te_download_nltk_data = function(...) calls <<- calls + 1)
  withr::local_envvar(HF_HUB_OFFLINE = "1")
  skip_if_not(reticulate::py_available(initialize = TRUE), "no Python")
  transforEmotion:::.te_use_nltk_cache()
  expect_equal(calls, 0)
  withr::local_envvar(HF_HUB_OFFLINE = "")
  transforEmotion:::.te_use_nltk_cache()
  expect_equal(calls, 1)
})

test_that("threads wait for the result of huggingface_hub's symlink check", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("huggingface_hub"))
  transforEmotion:::.te_lock_symlink_check()
  # A second call does not wrap the function again
  transforEmotion:::.te_lock_symlink_check()

  # os.symlink fails slowly, as on Windows without Developer Mode; 8 threads
  # ask about one new folder at the same time, as in snapshot_download().
  # With local = TRUE, a function does not see the names defined at the top
  # of the code, so the functions get them as default arguments
  answers <- reticulate::py_run_string(local = TRUE, paste(
    "import os, tempfile, threading, time",
    "from huggingface_hub import file_download",
    "wrapped_twice = getattr(file_download.are_symlinks_supported.__defaults__[1], '_te_locked', False)",
    "def no_symlink(*args, _sleep=time.sleep, **kwargs):",
    "    _sleep(0.2)",
    "    raise OSError(22, 'A required privilege is not held by the client')",
    "folder = tempfile.mkdtemp()",
    "answers = []",
    "def ask(_answers=answers, _folder=folder, _hub=file_download):",
    "    _answers.append(_hub.are_symlinks_supported(_folder))",
    "real, os.symlink = os.symlink, no_symlink",
    "try:",
    "    threads = [threading.Thread(target=ask) for _ in range(8)]",
    "    [t.start() for t in threads]",
    "    [t.join() for t in threads]",
    "finally:",
    "    os.symlink = real",
    sep = "\n"
  ))
  hub <- reticulate::import("huggingface_hub.file_download")
  expect_true(isTRUE(hub$are_symlinks_supported$`_te_locked`))
  expect_false(answers$wrapped_twice)
  expect_equal(unlist(answers$answers), rep(FALSE, 8))
})
