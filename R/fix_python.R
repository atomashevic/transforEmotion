#' Repair a Python Setup Left by an Older transforEmotion
#'
#' @description
#' Run once after upgrading from transforEmotion 0.1.7 or older. It lists
#' what older releases left (a conda environment, reticulate's default
#' virtualenv, Python settings in \code{.Renviron} or \code{.Rprofile}),
#' removes the items you consent to, and builds the new Python environment
#' with \code{\link{setup_modules}}.
#'
#' transforEmotion does not need any of these items removed: it always
#' selects its own Python environment (see Details). Removal only frees disk
#' space.
#'
#' @param remove Which items to delete.
#' \describe{
#'   \item{\code{NULL} (default)}{In an interactive session, ask about each
#'   item that transforEmotion created (\code{"conda-env"}). In a
#'   non-interactive session, delete nothing.}
#'   \item{\code{character()}}{Delete nothing and do not ask.}
#'   \item{Item names}{Any of \code{"conda-env"}, \code{"r-reticulate"} and
#'   \code{"r-miniconda"}. Delete these without asking.}
#' }
#' The items are:
#' \describe{
#'   \item{\code{"conda-env"}}{The conda environment \code{transforEmotion}
#'   that versions 0.1.4 to 0.1.6 created.}
#'   \item{\code{"r-reticulate"}}{reticulate's default virtualenv. Other R
#'   packages can use it.}
#'   \item{\code{"r-miniconda"}}{reticulate's Miniconda installation,
#'   including every environment in it. Other R packages can use it.}
#' }
#' The shared items \code{"r-reticulate"} and \code{"r-miniconda"} are
#' deleted only when named here. Lines in \code{.Renviron} and
#' \code{.Rprofile} (\code{"python-setting"}) are reported and never edited.
#' @param setup Logical. Build the Python environment and download the
#' default models with \code{\link{setup_modules}} (about 1 GB on first run).
#' \code{FALSE} does not start Python.
#' @param extras Optional feature sets, passed to \code{\link{setup_modules}}.
#'
#' @details
#' When the package is loaded, it sets \code{RETICULATE_PYTHON} for the R
#' session to \code{"managed"}, which makes reticulate use the environment
#' transforEmotion declares, or to \code{TRANSFOREMOTION_PYTHON} when that is
#' set. This overrides an older \code{RETICULATE_PYTHON},
#' \code{reticulate::use_virtualenv()} calls and the \code{r-reticulate}
#' virtualenv. To use your own Python, set \code{TRANSFOREMOTION_PYTHON}.
#'
#' Python can be selected only once per R session. If another Python already
#' runs, restart R and load transforEmotion before anything starts Python.
#'
#' @return Invisibly, a list of class \code{"te_fix_report"}:
#' \describe{
#'   \item{\code{ready}}{\code{TRUE} when Python runs and has the modules for
#'   the core and \code{extras} feature sets.}
#'   \item{\code{restart}}{\code{TRUE} when a Python that transforEmotion did
#'   not select runs in this session, so R must restart.}
#'   \item{\code{python}}{A list: \code{state} (\code{"not_started"},
#'   \code{"managed"} or \code{"foreign"}), \code{path}, \code{version},
#'   \code{reason} (what selected the Python) and \code{error} (the setup
#'   error, or \code{NA}).}
#'   \item{\code{leftovers}}{A data frame with one row per item found:
#'   \code{id}, \code{path}, \code{owner}, \code{effect}, \code{size_gb},
#'   \code{action} (\code{"removed"}, \code{"kept"}, \code{"failed"} or
#'   \code{"reported"}) and \code{detail}.}
#' }
#' A setup failure gives a warning, and the report is still returned. In a
#' script, use \code{stopifnot(fix_python()$ready)}.
#'
#' @examples
#' \dontrun{
#' # After upgrading, in a new R session
#' fix_python()
#'
#' # Non-interactive session: delete the old conda environment and the
#' # default virtualenv
#' fix_python(remove = c("conda-env", "r-reticulate"))
#'
#' # Report only; change nothing and do not start Python
#' report <- fix_python(remove = character(), setup = FALSE)
#' report$leftovers
#' }
#'
#' @export
fix_python <- function(remove = NULL, setup = TRUE, extras = character()) {
  removable <- names(Filter(function(kind) !is.null(kind$remover), .te_leftover_kinds))
  if (!is.null(remove) && (!is.character(remove) || anyNA(remove))) {
    stop("remove must be NULL or a character vector of item names.", call. = FALSE)
  }
  unknown <- setdiff(remove, removable)
  if (length(unknown)) {
    stop("Cannot remove: ", paste(unknown, collapse = ", "), ". Choose from: ",
         paste(removable, collapse = ", "), ".", call. = FALSE)
  }
  extras <- .te_check_extras(extras)

  .te_claim_python()
  leftovers <- .te_find_leftovers()
  leftovers$action <- .te_plan_removal(leftovers, remove, interactive(), ask = .te_ask)
  leftovers <- .te_apply_removal(leftovers)

  python <- .te_python_in_use()
  restart <- python$state == "foreign" &&
    !nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON", unset = ""))
  error <- NA_character_
  if (restart && isTRUE(setup)) {
    warning(.te_foreign_python_message(python), call. = FALSE)
  } else if (isTRUE(setup)) {
    error <- tryCatch({
      setup_modules(extras = extras)
      NA_character_
    }, error = function(e) conditionMessage(e))
    if (!is.na(error)) warning("Python setup failed: ", error, call. = FALSE)
    python <- .te_python_in_use()
  }
  python$error <- error

  modules <- unique(unlist(.te_feature_modules[c("core", extras)], use.names = FALSE))
  ready <- is.na(error) && !restart && python$state != "not_started" &&
    all(vapply(modules, reticulate::py_module_available, logical(1)))

  report <- structure(
    list(ready = ready, restart = restart, python = python, leftovers = leftovers),
    class = "te_fix_report"
  )
  message(paste(format(report), collapse = "\n"))
  invisible(report)
}

#' @export
format.te_fix_report <- function(x, ...) {
  py <- x$python
  version <- paste0(" (Python ", py$version,
                    if (!is.na(py$reason)) paste0(", selected by ", py$reason), ")")
  lines <- c(
    "transforEmotion Python check",
    switch(py$state,
      not_started = "Python in this session: not started. transforEmotion will use its own environment.",
      managed = paste0("Python in this session: ", py$path, version, ". transforEmotion set it up."),
      foreign = paste0("Python in this session: ", py$path, version, ". transforEmotion did not set it up.")
    )
  )

  lf <- x$leftovers
  if (!nrow(lf)) {
    lines <- c(lines, "Nothing found from older setups.")
  } else {
    lines <- c(lines, "Found from older setups:")
    for (i in seq_len(nrow(lf))) {
      what <- if (is.na(lf$path[i])) {
        paste(lf$detail[i], "(set in this R session)")
      } else if (lf$id[i] == "python-setting") {
        paste0(lf$detail[i], " (", lf$path[i], ")")
      } else {
        size <- if (!is.na(lf$size_gb[i])) sprintf("%.1f GB", lf$size_gb[i])
        paste0(lf$path[i], if (!is.null(size)) paste0(" (", size, ")"))
      }
      status <- switch(lf$action[i],
        removed = "Removed.",
        failed = "",
        reported = paste0("transforEmotion ignores it.", if (!is.na(lf$path[i]))
          " Delete the line if no other project needs it."),
        kept = paste0(
          "Kept.", if (lf$owner[i] == "shared") " Other R packages can use it.",
          " To remove: fix_python(remove = \"", lf$id[i], "\")."
        )
      )
      about <- if (lf$id[i] != "python-setting" && nzchar(lf$detail[i])) lf$detail[i]
      lines <- c(lines,
        sprintf("  [%s] %s", lf$effect[i], what),
        paste0("    ", trimws(paste(c(about, status), collapse = " ")))
      )
    }
  }

  lines <- c(lines,
    if (x$restart) "Restart R. Run library(transforEmotion) and fix_python() before anything else starts Python.",
    if (!is.na(py$error)) paste("Setup failed:", py$error),
    if (x$ready) "transforEmotion is ready." else if (py$state == "not_started") "Python was not started, so the environment was not checked.",
    "Old uv environments can be removed with `uv cache prune` in a shell.",
    if (any(lf$id %in% c("conda-env", "r-miniconda"))) {
      "Delete setup_miniconda() and reticulate::use_condaenv(\"transforEmotion\") from your scripts."
    }
  )
  lines
}

#' @export
print.te_fix_report <- function(x, ...) {
  cat(format(x), sep = "\n")
  invisible(x)
}

# The kinds of items older setups left. Argument checks, the finder, the
# planner and the printer read this list. owner: "transforEmotion" items may
# be offered in a prompt, "shared" items are removed only by name, "user"
# items (lines in startup files) are never edited. marker: the file that
# identifies the folder, checked again just before removal.
.te_leftover_kinds <- list(
  "python-setting" = list(owner = "user", effect = "ignored", marker = NULL,
                          remover = NULL),
  "r-reticulate" = list(owner = "shared", effect = "ignored", marker = "pyvenv.cfg",
                        remover = function(path) reticulate::virtualenv_remove(path, confirm = FALSE)),
  # unlink(): reticulate::conda_remove() needs a working conda binary, which
  # an old Miniconda installation may no longer have
  "conda-env" = list(owner = "transforEmotion", effect = "disk", marker = "conda-meta",
                     remover = function(path) unlink(path, recursive = TRUE)),
  "r-miniconda" = list(owner = "shared", effect = "disk", marker = "conda-meta",
                       remover = function(path) reticulate::miniconda_uninstall(path))
)

#' @noRd
.te_leftover_locations <- function() {
  replaced <- .te_py_state$replaced_python
  list(
    virtualenv_root = path.expand(reticulate::virtualenv_root()),
    miniconda = reticulate::miniconda_path(),
    startup_files = c(Sys.getenv("R_ENVIRON_USER"), "~/.Renviron", ".Renviron",
                      Sys.getenv("R_PROFILE_USER"), "~/.Rprofile", ".Rprofile"),
    env = c(RETICULATE_PYTHON = if (is.null(replaced)) "" else replaced,
            RETICULATE_PYTHON_ENV = Sys.getenv("RETICULATE_PYTHON_ENV")),
    python = .te_python_in_use()$path
  )
}

#' @noRd
# Reads the file system only: no Python, no network, no writes. Folder sizes
# can take seconds for a large conda environment, so this never runs at
# attach.
.te_find_leftovers <- function(where = .te_leftover_locations()) {
  rows <- list()
  add <- function(id, path, detail, size_gb = NA_real_) {
    kind <- .te_leftover_kinds[[id]]
    effect <- if (!is.na(path) && .te_path_within(where$python, path)) "blocks" else kind$effect
    rows[[length(rows) + 1L]] <<- data.frame(
      id = id, path = path, owner = kind$owner, effect = effect,
      size_gb = size_gb, action = NA_character_, detail = detail,
      stringsAsFactors = FALSE
    )
  }
  is_kind <- function(path, id) {
    dir.exists(path) && file.exists(file.path(path, .te_leftover_kinds[[id]]$marker))
  }

  settings <- character()
  files <- where$startup_files[nzchar(where$startup_files) & file.exists(where$startup_files)]
  for (file in unique(normalizePath(files, winslash = "/"))) {
    text <- readLines(file, warn = FALSE)
    hits <- grep("RETICULATE_PYTHON|use_condaenv|use_virtualenv|use_python|use_miniconda", text)
    for (line in hits[!grepl("^\\s*#", text[hits])]) {
      settings[paste0(file, ":", line)] <- trimws(text[line])
    }
  }
  # A session value that a startup file line sets is reported once, as that line
  for (name in names(where$env)[nzchar(where$env)]) {
    found <- regmatches(settings, regexec(
      paste0("(^|[^[:alnum:]_])", name, "\\s*=\\s*(\"[^\"]*\"|'[^']*'|[^,)]*)"), settings))
    values <- vapply(found, function(m) if (length(m)) trimws(m[3]) else NA_character_, character(1))
    values <- sub("^([\"'])(.*)\\1$", "\\2", values)
    if (!where$env[[name]] %in% values) {
      add("python-setting", NA_character_, paste0(name, "=", where$env[[name]]))
    }
  }
  for (location in names(settings)) add("python-setting", location, settings[[location]])

  venv <- file.path(where$virtualenv_root, "r-reticulate")
  if (is_kind(venv, "r-reticulate")) {
    add("r-reticulate", venv, .te_venv_summary(venv), .te_dir_size_gb(venv))
  }
  conda_env <- file.path(where$miniconda, "envs", "transforEmotion")
  if (is_kind(conda_env, "conda-env")) {
    add("conda-env", conda_env, "Made by transforEmotion 0.1.4 to 0.1.6.",
        .te_dir_size_gb(conda_env))
  }
  if (is_kind(where$miniconda, "r-miniconda")) {
    envs <- list.dirs(file.path(where$miniconda, "envs"), recursive = FALSE, full.names = FALSE)
    detail <- if (length(envs)) {
      paste0("Holds the environments ", paste(envs, collapse = ", "), ".")
    } else {
      "Holds no environments."
    }
    add("r-miniconda", where$miniconda, detail, .te_dir_size_gb(where$miniconda))
  }

  if (!length(rows)) {
    return(data.frame(
      id = character(), path = character(), owner = character(), effect = character(),
      size_gb = numeric(), action = character(), detail = character(),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

#' @noRd
# "Python 3.10.12, has torch, transformers.", read from pyvenv.cfg and the
# site-packages folder names without starting Python
.te_venv_summary <- function(venv) {
  cfg <- readLines(file.path(venv, "pyvenv.cfg"), warn = FALSE)
  version <- sub("^[^=]*=\\s*", "", grep("^\\s*version(_info)?\\s*=", cfg, value = TRUE))
  site <- c(Sys.glob(file.path(venv, "lib", "python*", "site-packages")),
            file.path(venv, "Lib", "site-packages"))
  has <- intersect(c("torch", "transformers"), list.files(site))
  parts <- c(
    if (length(version)) paste("Python", trimws(version[1])),
    if (length(has)) paste("has", paste(has, collapse = ", "))
  )
  if (length(parts)) paste0(paste(parts, collapse = ", "), ".") else ""
}

#' @noRd
# A virtualenv links bin/python to the base interpreter and lib64 to lib;
# links are skipped and files reached twice are counted once
.te_dir_size_gb <- function(path) {
  files <- list.files(path, recursive = TRUE, all.files = TRUE, full.names = TRUE, no.. = TRUE)
  files <- unique(normalizePath(files[!nzchar(Sys.readlink(files))], winslash = "/"))
  sum(file.info(files, extra_cols = FALSE)$size, na.rm = TRUE) / 1e9
}

#' @noRd
# Whether `file` lies inside the folder `dir`. Only the folder part of `file`
# is resolved: a virtualenv's bin/python is a link to the base interpreter.
.te_path_within <- function(file, dir) {
  if (length(file) != 1L || is.na(file)) return(FALSE)
  parent <- normalizePath(dirname(file), winslash = "/", mustWork = FALSE)
  dir <- normalizePath(dir, winslash = "/", mustWork = FALSE)
  identical(parent, dir) || startsWith(parent, paste0(dir, "/"))
}

#' @noRd
# The consent rules. Returns one action per row: "remove", "kept" or
# "reported". Asks only in an interactive session with remove = NULL, and
# only about items transforEmotion created.
.te_plan_removal <- function(leftovers, remove, interactive, ask) {
  kinds <- .te_leftover_kinds[leftovers$id]
  removable <- !vapply(kinds, function(kind) is.null(kind$remover), logical(1))
  action <- c("reported", "kept")[removable + 1L]
  if (is.character(remove)) {
    action[removable & leftovers$id %in% remove] <- "remove"
    return(action)
  }
  if (!interactive) return(action)
  owner <- vapply(kinds, function(kind) kind$owner, character(1))
  for (i in which(owner == "transforEmotion" & leftovers$effect != "blocks")) {
    size <- if (!is.na(leftovers$size_gb[i])) sprintf(" (%.1f GB)", leftovers$size_gb[i])
    if (isTRUE(ask(paste0("Remove ", leftovers$path[i], size, "?")))) action[i] <- "remove"
  }
  action
}

#' @noRd
# Cancel counts as no
.te_ask <- function(question) isTRUE(utils::askYesNo(question, default = FALSE))

#' @noRd
# Deletes the rows planned "remove". Each target is a path the finder
# located; it is checked again here because the folder can change between
# finding and removing.
.te_apply_removal <- function(leftovers, running = .te_python_in_use()$path) {
  for (i in which(leftovers$action == "remove")) {
    kind <- .te_leftover_kinds[[leftovers$id[i]]]
    path <- leftovers$path[i]
    refused <- if (nzchar(Sys.readlink(path))) {
      "it is a symbolic link"
    } else if (!file.exists(file.path(path, kind$marker))) {
      paste("it has no", kind$marker)
    } else if (.te_path_within(running, path)) {
      "it holds the Python running in this session. Restart R first"
    }
    if (is.null(refused)) {
      refused <- tryCatch({
        kind$remover(path)
        if (dir.exists(path)) "the folder is still there"
      }, error = function(e) conditionMessage(e))
    }
    if (is.null(refused)) {
      leftovers$action[i] <- "removed"
    } else {
      leftovers$action[i] <- "failed"
      leftovers$detail[i] <- paste0("Not removed: ", refused, ".")
    }
  }
  # Removing r-miniconda also removes the conda environment inside it
  gone <- leftovers$action == "kept" & !dir.exists(leftovers$path)
  leftovers$action[gone] <- "removed"
  leftovers
}
