#' @noRd
# Ensure all Python work happens in the 'transforEmotion' conda environment
ensure_te_py_env <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(invisible(FALSE))

  # If Python already initialized, verify it's our env
  initialized <- FALSE
  cfg <- NULL
  try({ initialized <- reticulate::py_available(initialize = FALSE) }, silent = TRUE)
  if (isTRUE(initialized)) {
    cfg <- try(reticulate::py_config(), silent = TRUE)
    if (!inherits(cfg, "try-error")) {
      py <- paste(c(cfg$python, cfg$pythonhome), collapse = " ")
      if (!grepl("transforEmotion", py, fixed = TRUE)) {
        stop(
          paste0(
            "reticulate is already initialized with a different Python (", cfg$python, "). ",
            "transforEmotion requires the 'transforEmotion' conda env for Python tasks.\n",
            "Restart your R session, then call transforEmotion::setup_modules() or ",
            "reticulate::use_condaenv('transforEmotion') before any Python use."
          ), call. = FALSE
        )
      }
      return(invisible(TRUE))
    }
  }

  # Not initialized yet — set preferred conda env for upcoming initialization
  try(reticulate::use_condaenv("transforEmotion", required = FALSE), silent = TRUE)
  invisible(TRUE)
}

