# Offline sessions (UV_OFFLINE, for example from setup_cache(offline = TRUE))
# whose uv cache lacks a declared Python package. No test runs uv or Python:
# reticulate's py_available(), py_require(), py_discover_config() and
# uv_run_tool() are mocked.

uv_missing_package <- structure(
  c("Resolved 40 packages in 12ms",
    "error: No solution found when resolving tool dependencies",
    "  cause: Because pytubefix was not found in the cache and you require pytubefix, we can conclude that your requirements are unsatisfiable.",
    "",
    "hint: Packages were unavailable because the network was disabled. When the network is disabled, registry packages may only be read from the cache."),
  status = 1L
)

# An offline session prepared with setup_cache("/project/lab/cache"), with the
# given feature sets declared. Restores the package state afterwards.
local_offline_session <- function(features = "core", use_gpu = FALSE,
                                  env = parent.frame()) {
  withr::local_envvar(
    UV_OFFLINE = "1", UV_CACHE_DIR = "/project/lab/cache/uv",
    TRANSFOREMOTION_PYTHON = NA, RETICULATE_USE_MANAGED_VENV = NA,
    .local_envir = env
  )
  state <- transforEmotion:::.te_py_state
  old <- list(features = state$features, use_gpu = state$use_gpu,
              cache_check = state$cache_check)
  withr::defer({
    state$features <- old$features
    state$use_gpu <- old$use_gpu
    state$cache_check <- old$cache_check
  }, envir = env)
  state$features <- features
  state$use_gpu <- use_gpu
  state$cache_check <- NULL
}

# The requirements reticulate reports for the declared feature sets
declared <- function(features = "core", use_gpu = FALSE, other = character()) {
  list(
    packages = c("numpy", unlist(lapply(features, transforEmotion:::.te_py_requirements,
                                        use_gpu = use_gpu)), other),
    python_version = c(">=3.12", "<3.13")
  )
}

test_that("offline, a package missing from the uv cache stops before Python starts", {
  local_offline_session()
  reqs <- declared(other = "pytubefix")
  seen <- NULL
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) reqs,
    py_discover_config = function(...) NULL,
    uv_run_tool = function(tool, args = character(), ..., with = NULL, python_version = NULL) {
      seen <<- list(tool = tool, with = with, python_version = python_version)
      uv_missing_package
    },
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("core"),
                      "uv cannot build the Python environment from its cache")
  msg <- conditionMessage(err)
  # uv's reason, from its error line on
  expect_match(msg, "pytubefix was not found in the cache", fixed = TRUE)
  expect_match(msg, "hint: Packages were unavailable", fixed = TRUE)
  expect_no_match(msg, "Resolved 40 packages", fixed = TRUE)
  # The fix, built from the session
  expect_match(msg, 'setup_cache("/project/lab/cache", gpu = FALSE)\n', fixed = TRUE)
  expect_match(msg, "setup_modules()\n", fixed = TRUE)
  expect_match(msg, "without offline = TRUE", fixed = TRUE)
  expect_match(msg, "Declare pytubefix with reticulate::py_require()", fixed = TRUE)
  # The declared requirements, with the Python constraint as one string
  expect_equal(seen$tool, "python")
  expect_equal(seen$with, reqs$packages)
  expect_equal(seen$python_version, ">=3.12,<3.13")
})

test_that("the offline fix names the declared PyTorch build and extras", {
  local_offline_session(features = c("core", "rag"), use_gpu = TRUE)
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) declared(c("core", "rag"), use_gpu = TRUE),
    py_discover_config = function(...) NULL,
    uv_run_tool = function(...) uv_missing_package,
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("rag"), "uv cannot build")
  msg <- conditionMessage(err)
  expect_match(msg, 'setup_cache("/project/lab/cache", gpu = TRUE)', fixed = TRUE)
  expect_match(msg, 'setup_modules(extras = "rag")', fixed = TRUE)
  # Every declared package belongs to transforEmotion or reticulate
  expect_no_match(msg, "Declare ")

  # A uv cache that setup_cache() did not set
  state <- transforEmotion:::.te_py_state
  state$cache_check <- NULL
  withr::local_envvar(UV_CACHE_DIR = NA)
  err <- expect_error(transforEmotion:::.te_require("rag"), "uv cannot build")
  expect_match(conditionMessage(err), "setup_cache(<the same path>, gpu = TRUE)", fixed = TRUE)
})

test_that("a uv cache with the packages passes, and each requirement set is checked once", {
  local_offline_session()
  reqs <- declared()
  calls <- 0
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) reqs,
    py_discover_config = function(...) NULL,
    uv_run_tool = function(...) {
      calls <<- calls + 1
      character()
    },
    .package = "reticulate"
  )
  expect_no_error(transforEmotion:::.te_require("core"))
  expect_no_error(transforEmotion:::.te_require("core"))
  expect_equal(calls, 1)

  # A new requirement is a new set
  reqs <- declared(other = "pytubefix")
  expect_no_error(transforEmotion:::.te_require("core"))
  expect_equal(calls, 2)
})

test_that("a failed check is remembered for the same requirements", {
  local_offline_session()
  calls <- 0
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) declared(other = "pytubefix"),
    py_discover_config = function(...) NULL,
    uv_run_tool = function(...) {
      calls <<- calls + 1
      uv_missing_package
    },
    .package = "reticulate"
  )
  expect_error(transforEmotion:::.te_require("core"), "pytubefix was not found")
  expect_error(transforEmotion:::.te_require("core"), "pytubefix was not found")
  expect_equal(calls, 1)
})

test_that("a uv that cannot run offline is reported with the same fix", {
  local_offline_session()
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) declared(),
    py_discover_config = function(...) NULL,
    uv_run_tool = function(...) stop("cannot open URL 'https://astral.sh/uv/install.sh'"),
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("core"), "astral.sh", fixed = TRUE)
  expect_match(conditionMessage(err), 'setup_cache("/project/lab/cache", gpu = FALSE)',
               fixed = TRUE)
})

test_that("the cache is not checked online, after Python starts, or while loading", {
  local_offline_session()
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) declared(other = "pytubefix"),
    py_discover_config = function(...) NULL,
    uv_run_tool = function(...) stop("uv must not run"),
    .package = "reticulate"
  )
  # .onLoad and .te_set_torch_flavor() pass check_cache = FALSE
  expect_no_error(transforEmotion:::.te_require("core", check_cache = FALSE))

  # A value that uv reads as false
  withr::local_envvar(UV_OFFLINE = "0")
  expect_no_error(transforEmotion:::.te_require("core"))
  withr::local_envvar(UV_OFFLINE = NA)
  expect_no_error(transforEmotion:::.te_require("core"))

  # Python runs: reticulate installs additions into it
  withr::local_envvar(UV_OFFLINE = "1")
  local_mocked_bindings(py_available = function(...) TRUE, .package = "reticulate")
  expect_no_error(transforEmotion:::.te_require("core"))
})

test_that("the cache is not checked with a fixed Python", {
  local_offline_session()
  withr::local_envvar(TRANSFOREMOTION_PYTHON = file.path(tempdir(), "python"))
  state <- transforEmotion:::.te_py_state
  fixed <- state$fixed_python
  withr::defer(state$fixed_python <- fixed)
  state$fixed_python <- TRUE
  local_mocked_bindings(
    py_available = function(...) FALSE,
    uv_run_tool = function(...) stop("uv must not run"),
    .package = "reticulate"
  )
  expect_no_error(transforEmotion:::.te_require("core"))
})

test_that("setup_cache(offline = TRUE, gpu = ) does not check the cache", {
  # The check waits for the first analysis function, which sees the final
  # PyTorch build and feature sets
  local_offline_session(features = character())
  vars <- c("UV_PYTHON_INSTALL_DIR", "UV_PYTHON_PREFERENCE", "HF_HOME",
            "R_USER_CACHE_DIR", "HF_HUB_OFFLINE")
  withr::local_envvar(setNames(rep(NA_character_, length(vars)), vars))
  local_mocked_bindings(te_should_use_gpu = function() FALSE)
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) invisible(),
    uv_run_tool = function(...) stop("uv must not run"),
    .package = "reticulate"
  )
  expect_no_error(setup_cache(withr::local_tempdir(), offline = TRUE, gpu = TRUE))
  expect_true(transforEmotion:::.te_py_state$use_gpu)
})

test_that("the cache is not checked when reticulate will use another Python", {
  local_offline_session()
  calls <- 0
  discovered <- list(python = "/work/.venv/bin/python",
                     forced = "'./.venv' existing in the current working directory")
  local_mocked_bindings(
    py_available = function(...) FALSE,
    py_require = function(...) declared(other = "pytubefix"),
    py_discover_config = function(...) discovered,
    uv_run_tool = function(...) {
      calls <<- calls + 1
      uv_missing_package
    },
    .package = "reticulate"
  )
  # VIRTUAL_ENV, RETICULATE_PYTHON, use_python(), a venv in the working
  # directory and the others all come back from py_discover_config()
  expect_no_error(transforEmotion:::.te_require("core"))

  # reticulate reports a broken selection itself when Python starts
  local_mocked_bindings(
    py_discover_config = function(...) stop("Python specified in RETICULATE_PYTHON (x) does not exist"),
    .package = "reticulate"
  )
  expect_no_error(transforEmotion:::.te_require("core"))

  # The user turned off reticulate's uv environment
  withr::local_envvar(RETICULATE_USE_MANAGED_VENV = "false")
  local_mocked_bindings(
    py_discover_config = function(...) stop("discovery must not run"),
    .package = "reticulate"
  )
  expect_no_error(transforEmotion:::.te_require("core"))
  expect_equal(calls, 0)

  # reticulate's uv environment, as a later reticulate could report it
  withr::local_envvar(RETICULATE_USE_MANAGED_VENV = NA)
  local_mocked_bindings(
    py_discover_config = function(...) list(python = "/cache/env/bin/python", ephemeral = TRUE),
    .package = "reticulate"
  )
  expect_error(transforEmotion:::.te_require("core"), "pytubefix was not found")
  expect_equal(calls, 1)
})

test_that("offline, a failed install into a running Python adds the cache fix", {
  local_offline_session()
  local_mocked_bindings(
    py_available = function(...) TRUE,
    py_module_available = function(module) FALSE,
    py_require = function(...) warning("Call `py_require()` to remove or replace conflicting requirements."),
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("youtube"), "Python packages for 'youtube'")
  msg <- conditionMessage(err)
  # A version conflict gives the same warning, so the advice for it stays
  expect_match(msg, "Restart R", fixed = TRUE)
  expect_match(msg, 'setup_cache("/project/lab/cache", gpu = FALSE)', fixed = TRUE)
  expect_match(msg, 'setup_modules(extras = "youtube")', fixed = TRUE)
  expect_false("youtube" %in% transforEmotion:::.te_py_state$features)

  # Online, the advice is unchanged
  withr::local_envvar(UV_OFFLINE = NA)
  err <- expect_error(transforEmotion:::.te_require("youtube"), "Restart R")
  expect_no_match(conditionMessage(err), "UV_OFFLINE")
})

test_that("UV_OFFLINE counts only with the values uv reads as true", {
  for (v in c("1", "true", "TRUE", "t", "yes", "y", "on")) {
    withr::local_envvar(UV_OFFLINE = v)
    expect_true(transforEmotion:::.te_uv_offline(), label = v)
  }
  for (v in c("", "0", "false", "off", "no")) {
    withr::local_envvar(UV_OFFLINE = v)
    expect_false(transforEmotion:::.te_uv_offline(), label = v)
  }
  withr::local_envvar(UV_OFFLINE = NA)
  expect_false(transforEmotion:::.te_uv_offline())
})

test_that("reticulate's discovery returns NULL where it would build its uv environment", {
  # .te_uses_uv_python() relies on this; a change in reticulate turns the
  # offline check off
  skip_on_cran()
  skip_if(reticulate::py_available(initialize = FALSE), "Python already started")
  skip_if(nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON")), "TRANSFOREMOTION_PYTHON is set")
  skip_if(dir.exists(path.expand("~/.virtualenvs/r-reticulate")),
          "~/.virtualenvs/r-reticulate exists")
  withr::local_envvar(
    VIRTUAL_ENV = NA, RETICULATE_PYTHON = NA, RETICULATE_PYTHON_ENV = NA,
    RETICULATE_PYTHON_FALLBACK = NA, RETICULATE_USE_MANAGED_VENV = NA,
    WORKON_HOME = NA, RETICULATE_VIRTUALENV_ROOT = NA
  )
  withr::local_dir(withr::local_tempdir())
  expect_null(reticulate::py_discover_config())
  expect_true(transforEmotion:::.te_uses_uv_python())
})
