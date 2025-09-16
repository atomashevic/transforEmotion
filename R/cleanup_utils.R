#' Remove reticulate's default virtualenv (r-reticulate)
#'
#' @description
#' Removes the default `reticulate` virtual environment at
#' `~/.virtualenvs/r-reticulate` to avoid conflicts with uv-managed
#' environments used by transforEmotion. This is optional and only needed if
#' you want to ensure uv's environment is preferred.
#'
#' @param confirm Logical. Ask for confirmation before removal. Default TRUE.
#' @return Invisibly returns TRUE on success, FALSE otherwise.
#' @examples
#' \dontrun{
#' te_cleanup_default_venv()
#' te_cleanup_default_venv(confirm = FALSE)
#' }
#' @export
te_cleanup_default_venv <- function(confirm = TRUE) {
  venv_path <- path.expand(file.path("~", ".virtualenvs", "r-reticulate"))
  if (!dir.exists(venv_path)) {
    message("No default reticulate venv detected at ", venv_path)
    return(invisible(FALSE))
  }

  proceed <- TRUE
  if (isTRUE(confirm) && interactive()) {
    ans <- tryCatch(readline(
      paste0("Remove default reticulate venv at ", venv_path, "? [y/N]: ")
    ), error = function(e) "")
    ans <- tolower(trimws(ans))
    proceed <- ans %in% c("y", "yes")
  }

  if (!proceed) {
    message("Cancelled. The environment was not removed.")
    return(invisible(FALSE))
  }

  ok <- tryCatch({
    reticulate::virtualenv_remove("r-reticulate", confirm = FALSE)
    TRUE
  }, error = function(e) {
    message("Failed to remove environment: ", e$message)
    FALSE
  })

  if (isTRUE(ok)) {
    message("Removed ", venv_path, ". Restart R so changes take effect.")
  }
  invisible(ok)
}

