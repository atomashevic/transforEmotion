#' Deprecated: Miniconda setup
#'
#' @description
#' setup_miniconda() is no longer needed. transforEmotion installs Python
#' itself with uv through \code{reticulate::py_require()}. Run
#' \code{\link{fix_python}} once to remove the conda environment that older
#' versions created and to build the new environment. Delete calls to
#' \code{setup_miniconda()} and
#' \code{reticulate::use_condaenv("transforEmotion")} from your scripts.
#'
#' @export
setup_miniconda <- function() {
  .Deprecated("fix_python", msg = paste(
    "setup_miniconda() is no longer needed. transforEmotion installs Python itself.",
    "Run fix_python() once to remove the old conda environment and build the new one.",
    "Delete calls to setup_miniconda() and reticulate::use_condaenv(\"transforEmotion\")",
    "from your scripts."
  ))
  invisible(TRUE)
}
