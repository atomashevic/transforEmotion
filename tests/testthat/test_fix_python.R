old_python <- list(python = "/old/bin/python", version = "3.10",
                   forced = "use_python() function")

test_that("before Python starts, the claim selects the managed environment", {
  state <- local_te_py_state()
  state$claimed <- NULL
  state$replaced_python <- NULL
  withr::local_envvar(RETICULATE_PYTHON = "/old/python", TRANSFOREMOTION_PYTHON = NA)
  local_mocked_bindings(py_available = function(...) FALSE, .package = "reticulate")

  transforEmotion:::.te_claim_python()
  expect_equal(Sys.getenv("RETICULATE_PYTHON"), "managed")
  expect_equal(state$replaced_python, "/old/python")
  transforEmotion:::.te_claim_python()
  expect_equal(state$replaced_python, "/old/python")

  msgs <- capture_messages(transforEmotion:::.onAttach("", "transforEmotion"))
  expect_match(paste(msgs, collapse = ""), "ignores RETICULATE_PYTHON=/old/python", fixed = TRUE)

  transforEmotion:::.te_release_python()
  expect_equal(Sys.getenv("RETICULATE_PYTHON"), "/old/python")
})

test_that("the claim selects TRANSFOREMOTION_PYTHON, and nothing once Python runs", {
  local_te_py_state()
  custom <- withr::local_tempfile()
  withr::local_envvar(RETICULATE_PYTHON = NA, TRANSFOREMOTION_PYTHON = custom)
  local_mocked_bindings(py_available = function(...) FALSE, .package = "reticulate")
  transforEmotion:::.te_claim_python()
  expect_equal(Sys.getenv("RETICULATE_PYTHON"), custom)

  withr::local_envvar(RETICULATE_PYTHON = "/running/python")
  local_mocked_bindings(py_available = function(...) TRUE, .package = "reticulate")
  transforEmotion:::.te_claim_python()
  expect_equal(Sys.getenv("RETICULATE_PYTHON"), "/running/python")
})

test_that("a running Python without the modules stops with its path", {
  state <- local_te_py_state()
  state$features <- character()
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA)
  local_mocked_bindings(te_should_use_gpu = function() FALSE)
  local_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() old_python,
    py_module_available = function(module) module != "torch",
    py_require = function(...) stop("py_require() must not be called"),
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("core"), class = "te_python_error")
  expect_match(conditionMessage(err),
               "/old/bin/python (Python 3.10, selected by use_python() function)", fixed = TRUE)
  expect_match(conditionMessage(err), "It has no module named: torch.", fixed = TRUE)
  expect_match(conditionMessage(err), "Restart R.", fixed = TRUE)
  expect_false("core" %in% state$features)
})

test_that("a running Python with the modules but another version warns once", {
  state <- local_te_py_state()
  state$features <- character()
  state$version_warned <- NULL
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA)
  local_mocked_bindings(te_should_use_gpu = function() FALSE)
  local_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() old_python,
    py_module_available = function(module) TRUE,
    py_require = function(...) stop("py_require() must not be called"),
    import = function(module) list(version = function(p) c(torch = "2.4.1", transformers = "4.40.0")[[p]]),
    .package = "reticulate"
  )
  warnings <- capture_warnings({
    transforEmotion:::.te_require("core")
    transforEmotion:::.te_require("youtube")
  })
  expect_length(warnings, 1)
  expect_match(warnings, "/old/bin/python (Python 3.10, torch 2.4.1, transformers 4.40.0)", fixed = TRUE)
  expect_setequal(state$features, c("core", "youtube"))

  state$features <- character()
  state$version_warned <- NULL
  local_mocked_bindings(
    py_config = function() modifyList(old_python, list(version = "3.12")),
    .package = "reticulate"
  )
  expect_no_warning(transforEmotion:::.te_require("core"))
  expect_true("core" %in% state$features)
})

test_that("a TRANSFOREMOTION_PYTHON without the modules is named in the error", {
  state <- local_te_py_state()
  state$features <- character()
  state$fixed_python <- NULL
  custom <- withr::local_tempfile(lines = "")
  withr::local_envvar(TRANSFOREMOTION_PYTHON = custom)
  local_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() list(python = custom, version = "3.12", forced = "RETICULATE_PYTHON"),
    py_module_available = function(module) FALSE,
    use_python = function(...) invisible(NULL),
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("core"), class = "te_python_error")
  expect_match(conditionMessage(err), custom, fixed = TRUE)
  expect_match(conditionMessage(err), "TRANSFOREMOTION_PYTHON selects this Python.", fixed = TRUE)
})

test_that("the finder reports what older setups left in a home folder", {
  fake <- local_fake_home()
  lf <- transforEmotion:::.te_find_leftovers(fake$where)
  expect_equal(lf$id, c("python-setting", "r-reticulate", "conda-env", "r-miniconda"))
  expect_equal(lf$owner, c("user", "shared", "transforEmotion", "shared"))
  expect_equal(lf$effect, c("ignored", "ignored", "disk", "disk"))
  expect_equal(lf$path[1], paste0(normalizePath(file.path(fake$home, ".Renviron"), winslash = "/"), ":2"))
  expect_equal(lf$detail[1], "RETICULATE_PYTHON=/old/bin/python")
  expect_equal(lf$detail[2], "Python 3.10.12, has torch.")
  expect_equal(lf$detail[4], "Holds the environments transforEmotion.")
  expect_equal(lf$size_gb[3], 0.001)

  fake$where$python <- file.path(fake$venv, "bin", "python")
  fake$where$env <- c(RETICULATE_PYTHON = "/x/python", RETICULATE_PYTHON_ENV = "")
  lf <- transforEmotion:::.te_find_leftovers(fake$where)
  expect_equal(lf$effect[lf$id == "r-reticulate"], "blocks")
  expect_equal(lf$detail[is.na(lf$path)], "RETICULATE_PYTHON=/x/python")
})

test_that("a session setting that a startup file sets is reported once, as the file line", {
  fake <- local_fake_home()
  renviron <- file.path(fake$home, ".Renviron")
  settings <- function(session, lines) {
    writeLines(lines, renviron)
    fake$where$env <- c(RETICULATE_PYTHON = session, RETICULATE_PYTHON_ENV = "")
    lf <- transforEmotion:::.te_find_leftovers(fake$where)
    paste(lf$path, lf$detail)[lf$id == "python-setting"]
  }
  line1 <- paste0(normalizePath(renviron, winslash = "/"), ":1")

  expect_equal(settings("/old/bin/python", "RETICULATE_PYTHON = \"/old/bin/python\" "),
               paste(line1, "RETICULATE_PYTHON = \"/old/bin/python\""))
  expect_equal(settings("/old/bin/python", "LANG=C"),
               "NA RETICULATE_PYTHON=/old/bin/python")
  expect_equal(settings("/new/bin/python", "RETICULATE_PYTHON=/old/bin/python"),
               c("NA RETICULATE_PYTHON=/new/bin/python",
                 paste(line1, "RETICULATE_PYTHON=/old/bin/python")))
})

test_that("removal needs consent: asked only interactively, shared items only by name", {
  lf <- data.frame(
    id = c("python-setting", "r-reticulate", "conda-env", "r-miniconda"),
    path = c("/h/.Renviron:2", "/h/.virtualenvs/r-reticulate", "/h/m/envs/transforEmotion", "/h/m"),
    effect = c("ignored", "ignored", "disk", "disk"),
    size_gb = c(NA, 1.2, 4.1, 0.6),
    stringsAsFactors = FALSE
  )
  plan <- transforEmotion:::.te_plan_removal
  never <- function(question) stop("asked")
  for (interactive in c(TRUE, FALSE)) {
    expect_equal(plan(lf, character(), interactive, never), c("reported", "kept", "kept", "kept"))
    expect_equal(plan(lf, "conda-env", interactive, never), c("reported", "kept", "remove", "kept"))
    expect_equal(plan(lf, "r-reticulate", interactive, never), c("reported", "remove", "kept", "kept"))
    expect_equal(plan(lf, c("r-miniconda", "conda-env"), interactive, never),
                 c("reported", "kept", "remove", "remove"))
  }
  expect_equal(plan(lf, NULL, FALSE, never), c("reported", "kept", "kept", "kept"))

  asked <- character()
  yes <- function(question) {
    asked <<- c(asked, question)
    TRUE
  }
  expect_equal(plan(lf, NULL, TRUE, yes), c("reported", "kept", "remove", "kept"))
  expect_equal(asked, "Remove /h/m/envs/transforEmotion (4.1 GB)?")
  expect_equal(plan(lf, NULL, TRUE, function(question) FALSE), c("reported", "kept", "kept", "kept"))

  lf$effect[3] <- "blocks"
  expect_equal(plan(lf, NULL, TRUE, never), c("reported", "kept", "kept", "kept"))
})

test_that("fix_python() removes only the named item and does not start Python", {
  fake <- local_fake_home()
  local_te_py_state()
  local_mocked_bindings(.te_leftover_locations = function() fake$where)
  local_mocked_bindings(
    py_available = function(...) FALSE,
    virtualenv_remove = function(...) stop("r-reticulate must not be removed"),
    .package = "reticulate"
  )
  expect_message(report <- fix_python(remove = "conda-env", setup = FALSE),
                 "To remove: fix_python(remove = \"r-reticulate\")", fixed = TRUE)
  expect_s3_class(report, "te_fix_report")
  expect_equal(report$leftovers$action, c("reported", "kept", "removed", "kept"))
  expect_false(dir.exists(fake$conda_env))
  expect_true(dir.exists(fake$venv))
  expect_false(report$ready)
  expect_equal(report$python$state, "not_started")
})

test_that("fix_python() in a non-interactive session neither asks nor removes", {
  skip_if(interactive())
  fake <- local_fake_home()
  local_te_py_state()
  local_mocked_bindings(
    .te_leftover_locations = function() fake$where,
    .te_ask = function(question) stop("asked")
  )
  local_mocked_bindings(
    py_available = function(...) FALSE,
    virtualenv_remove = function(...) stop("removed"),
    miniconda_uninstall = function(...) stop("removed"),
    .package = "reticulate"
  )
  report <- suppressMessages(fix_python(setup = FALSE))
  expect_equal(report$leftovers$action, c("reported", "kept", "kept", "kept"))
  expect_true(dir.exists(fake$conda_env))
  expect_true(dir.exists(fake$venv))
})

test_that("removing r-miniconda also removes the conda environment in it", {
  fake <- local_fake_home()
  local_te_py_state()
  local_mocked_bindings(.te_leftover_locations = function() fake$where)
  local_mocked_bindings(py_available = function(...) FALSE, .package = "reticulate")
  report <- suppressMessages(fix_python(remove = "r-miniconda", setup = FALSE))
  expect_equal(report$leftovers$action, c("reported", "kept", "removed", "removed"))
  expect_false(dir.exists(fake$where$miniconda))
})

test_that("removal refuses a symbolic link and the folder of the running Python", {
  skip_on_os("windows")
  fake <- local_fake_home()
  local_mocked_bindings(
    py_available = function(...) FALSE,
    virtualenv_remove = function(envname, ...) unlink(envname, recursive = TRUE),
    .package = "reticulate"
  )
  lf <- transforEmotion:::.te_find_leftovers(fake$where)
  lf$action <- ifelse(lf$id == "r-reticulate", "remove", "kept")
  lf <- transforEmotion:::.te_apply_removal(lf, running = file.path(fake$venv, "bin", "python"))
  expect_equal(lf$action[lf$id == "r-reticulate"], "failed")
  expect_match(lf$detail[lf$id == "r-reticulate"], "holds the Python running in this session", fixed = TRUE)
  expect_true(dir.exists(fake$venv))

  real <- file.path(fake$home, "elsewhere")
  file.rename(fake$venv, real)
  file.symlink(real, fake$venv)
  lf <- transforEmotion:::.te_find_leftovers(fake$where)
  lf$action <- ifelse(lf$id == "r-reticulate", "remove", "kept")
  lf <- transforEmotion:::.te_apply_removal(lf, running = NA_character_)
  expect_equal(lf$detail[lf$id == "r-reticulate"], "Not removed: it is a symbolic link.")
  expect_true(file.exists(file.path(real, "pyvenv.cfg")))
})

test_that("a failed setup is a warning and the report says why", {
  fake <- local_fake_home()
  local_te_py_state()
  local_mocked_bindings(
    .te_leftover_locations = function() fake$where,
    setup_modules = function(...) stop("uv could not resolve torch")
  )
  local_mocked_bindings(py_available = function(...) FALSE, .package = "reticulate")
  expect_warning(
    report <- suppressMessages(fix_python(remove = character())),
    "Python setup failed: uv could not resolve torch", fixed = TRUE
  )
  expect_equal(report$python$error, "uv could not resolve torch")
  expect_false(report$ready)
})

test_that("with another Python running, fix_python() asks for a restart and skips setup", {
  fake <- local_fake_home()
  local_te_py_state()
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA)
  local_mocked_bindings(
    .te_leftover_locations = function() fake$where,
    setup_modules = function(...) stop("setup must not run")
  )
  local_mocked_bindings(
    py_available = function(...) TRUE,
    py_config = function() old_python,
    .package = "reticulate"
  )
  expect_warning(report <- suppressMessages(fix_python(remove = character())),
                 "/old/bin/python", fixed = TRUE)
  expect_true(report$restart)
  expect_false(report$ready)
  expect_equal(report$python$state, "foreign")
})

test_that("remove accepts exactly the removable item names, which the help page lists", {
  kinds <- transforEmotion:::.te_leftover_kinds
  removable <- names(Filter(function(kind) !is.null(kind$remover), kinds))
  expect_error(fix_python(remove = "python-setting"),
               paste0("Choose from: ", paste(removable, collapse = ", ")), fixed = TRUE)
  expect_error(fix_python(remove = NA), "character vector of item names")

  rd <- normalizePath(testthat::test_path("..", "..", "man", "fix_python.Rd"), mustWork = FALSE)
  skip_if_not(file.exists(rd))
  text <- paste(readLines(rd, warn = FALSE), collapse = "\n")
  for (id in removable) expect_match(text, paste0("\\code{\"", id, "\"}"), fixed = TRUE)
})

test_that("te_cleanup_default_venv() removes nothing in a non-interactive session", {
  skip_if(interactive())
  expect_warning(
    expect_message(result <- te_cleanup_default_venv(), "Not removed: non-interactive session"),
    "fix_python"
  )
  expect_false(result)
})
