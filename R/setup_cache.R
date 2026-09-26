#' Use a Shared or Offline Cache
#'
#' @description
#' Stores the Python environment, Python itself and the Hugging Face models in
#' one folder, and optionally runs without network access. This is the route
#' for HPC clusters: prepare the folder on a login node with internet access,
#' then use it offline on compute nodes.
#'
#' @param path Folder for the caches, for example on project or scratch storage
#' shared by login and compute nodes. It needs several GB and must be writable.
#' @param offline Logical. Use only what is already in the cache, with no
#' network access (sets \code{UV_OFFLINE} and \code{HF_HUB_OFFLINE}).
#' @param gpu Logical or \code{NULL}. Choose the CUDA (\code{TRUE}) or CPU
#' (\code{FALSE}) build of PyTorch instead of detecting the GPU. Login nodes
#' usually have no GPU, so when the compute nodes do, pass \code{gpu = TRUE}
#' both when preparing the cache and in jobs.
#'
#' @details
#' Call \code{setup_cache()} in a new R session, after
#' \code{library(transforEmotion)} and before any analysis. It sets these
#' environment variables for the session:
#' \describe{
#'   \item{\code{UV_CACHE_DIR}}{uv package cache and Python environments}
#'   \item{\code{UV_PYTHON_INSTALL_DIR}, \code{UV_PYTHON_PREFERENCE}}{a Python
#'   interpreter kept in the cache, so compute nodes do not need one}
#'   \item{\code{HF_HOME}}{Hugging Face models}
#'   \item{\code{R_USER_CACHE_DIR}}{reticulate's own copy of uv and the NLTK
#'   data used by \code{rag()}. This also moves the cache of any other R
#'   package that uses \code{tools::R_user_dir()} in this session.}
#' }
#'
#' Offline sessions can use any combination of the feature sets prepared
#' with \code{setup_modules()}, and any model it downloaded.
#'
#' The cache must stay writable, because uv locks it even when reusing an
#' environment. For read-only environments such as Apptainer images, build a
#' fixed Python environment instead (see \code{\link{python_requirements}}).
#'
#' @return Invisibly, a named character vector of the environment variables set.
#'
#' @examples
#' \dontrun{
#' # On a login node, with internet access
#' library(transforEmotion)
#' setup_cache("/project/mylab/transforEmotion-cache", gpu = TRUE)
#' setup_modules(extras = "rag", models = "facebook/bart-large-mnli")
#'
#' # In a job on a compute node, without internet access
#' library(transforEmotion)
#' setup_cache("/project/mylab/transforEmotion-cache", offline = TRUE, gpu = TRUE)
#' transformer_scores(text, classes, transformer = "facebook-bart")
#' }
#'
#' @export
setup_cache <- function(path, offline = FALSE, gpu = NULL) {
  if (!is.character(path) || length(path) != 1L || !nzchar(path)) {
    stop("'path' must be a single folder path.", call. = FALSE)
  }
  if (reticulate::py_available(initialize = FALSE)) {
    stop("Python has already started in this session. Call setup_cache() ",
         "in a new R session before running any analysis.", call. = FALSE)
  }

  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)

  vars <- c(
    UV_CACHE_DIR = file.path(path, "uv"),
    UV_PYTHON_INSTALL_DIR = file.path(path, "python"),
    UV_PYTHON_PREFERENCE = "only-managed",
    HF_HOME = file.path(path, "huggingface"),
    R_USER_CACHE_DIR = file.path(path, "R")
  )
  if (isTRUE(offline)) {
    vars <- c(vars, UV_OFFLINE = "1", HF_HUB_OFFLINE = "1")
  }
  do.call(Sys.setenv, as.list(vars))

  if (!is.null(gpu)) .te_set_torch_flavor(gpu)

  invisible(vars)
}
