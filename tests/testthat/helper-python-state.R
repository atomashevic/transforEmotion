# .te_require() and .te_claim_python() change package state and
# RETICULATE_PYTHON; restore both when the calling test ends
local_te_py_state <- function(env = parent.frame()) {
  state <- transforEmotion:::.te_py_state
  saved <- mget(ls(state, all.names = TRUE), envir = state)
  withr::defer({
    rm(list = ls(state, all.names = TRUE), envir = state)
    list2env(saved, envir = state)
  }, envir = env)
  withr::local_envvar(
    RETICULATE_PYTHON = Sys.getenv("RETICULATE_PYTHON", unset = NA),
    .local_envir = env
  )
  invisible(state)
}

# A home folder with what older transforEmotion releases left
local_fake_home <- function(env = parent.frame()) {
  home <- withr::local_tempdir(.local_envir = env)
  venv <- file.path(home, ".virtualenvs", "r-reticulate")
  dir.create(file.path(venv, "bin"), recursive = TRUE)
  dir.create(file.path(venv, "lib", "python3.10", "site-packages", "torch"), recursive = TRUE)
  writeLines(c("home = /usr/bin", "version = 3.10.12"), file.path(venv, "pyvenv.cfg"))
  conda <- file.path(home, "r-miniconda")
  conda_env <- file.path(conda, "envs", "transforEmotion")
  dir.create(file.path(conda, "conda-meta"), recursive = TRUE)
  dir.create(file.path(conda_env, "conda-meta"), recursive = TRUE)
  writeBin(raw(1e6), file.path(conda_env, "conda-meta", "history"))
  writeLines(c("# RETICULATE_PYTHON=/commented/out", "RETICULATE_PYTHON=/old/bin/python"),
             file.path(home, ".Renviron"))
  list(
    home = home, venv = venv, conda_env = conda_env,
    where = list(
      virtualenv_root = file.path(home, ".virtualenvs"),
      miniconda = conda,
      startup_files = file.path(home, c(".Renviron", ".Rprofile")),
      env = c(RETICULATE_PYTHON = "", RETICULATE_PYTHON_ENV = ""),
      python = NA_character_
    )
  )
}
