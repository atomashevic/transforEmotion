#' Remove reticulate's default virtualenv (r-reticulate)
#'
#' @description
#' Deprecated. Use \code{fix_python(remove = "r-reticulate")} instead.
#' transforEmotion no longer needs this virtualenv removed: it always selects
#' its own Python environment (see \code{\link{fix_python}}). Other R packages
#' can use the virtualenv.
#'
#' @param confirm Logical. Ask for confirmation before removal. Default TRUE.
#' A non-interactive session cannot answer, so nothing is removed there
#' unless \code{confirm = FALSE}.
#' @return Invisibly returns TRUE on success, FALSE otherwise.
#' @examples
#' \dontrun{
#' fix_python(remove = "r-reticulate")
#' }
#' @export
te_cleanup_default_venv <- function(confirm = TRUE) {
  .Deprecated("fix_python")
  if (isTRUE(confirm)) {
    if (!interactive()) {
      message("Not removed: non-interactive session. Use fix_python(remove = \"r-reticulate\").")
      return(invisible(FALSE))
    }
    if (!.te_ask("Remove reticulate's default virtualenv r-reticulate? Other R packages can use it.")) {
      message("Cancelled. The environment was not removed.")
      return(invisible(FALSE))
    }
  }
  report <- fix_python(remove = "r-reticulate", setup = FALSE)
  invisible(any(report$leftovers$action[report$leftovers$id == "r-reticulate"] == "removed"))
}
