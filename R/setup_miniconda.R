#' Deprecated: Miniconda setup (use uv instead)
#'
#' @description
#' setup_miniconda() is deprecated. The transforEmotion package now uses
#' reticulate's uv-based ephemeral environments managed via `py_require()`.
#' No conda or Miniconda installation is required.
#'
#' @export
setup_miniconda <- function() {
  .Deprecated(msg = paste(
    "setup_miniconda() is deprecated.",
    "Python environments are now provisioned automatically using uv via",
    "reticulate::py_require().",
    "Run transforEmotion::setup_modules() to pre-warm dependencies,",
    "or just call any function and dependencies will be installed on first use."
  ))
  invisible(TRUE)
}

