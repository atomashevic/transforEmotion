# Python environment
#
# transforEmotion declares its Python requirements with reticulate::py_require().
# reticulate resolves them with uv into a cached environment keyed by the
# requirement set: the first call downloads packages, later sessions reuse the
# environment in well under a second. reticulate downloads its own uv when none
# is on the PATH.
#
# Requirements are split into feature sets. "core" is declared before Python
# starts; the others are added when a function first needs them.

# The PyTorch wheel URLs in .te_torch_requirements() are built for CPython 3.12.
.te_python_version <- ">=3.12,<3.13"

.te_torch_version <- "2.14.0"
.te_torchvision_version <- "0.29.0"

# Feature sets declared with py_require() in this session
.te_py_state <- new.env(parent = emptyenv())
.te_py_state$features <- character()

#' @noRd
te_should_use_gpu <- function() {
  # Allow explicit override via env var
  get_bool <- function(x) {
    x <- tolower(as.character(x))
    nzchar(x) && x %in% c("1", "true", "t", "yes", "y")
  }
  force_cpu <- get_bool(Sys.getenv("TRANSFOREMOTION_FORCE_CPU", unset = "")) ||
               get_bool(Sys.getenv("TE_FORCE_CPU", unset = ""))
  if (force_cpu) return(FALSE)

  explicit_gpu <- Sys.getenv("TRANSFOREMOTION_USE_GPU", unset = NA)
  if (!is.na(explicit_gpu) && nzchar(explicit_gpu)) return(get_bool(explicit_gpu))

  # macOS: skip CUDA decisions (MPS not handled here)
  os <- tolower(Sys.info()["sysname"])  # linux, windows, darwin
  if (identical(os, "darwin")) return(FALSE)

  # Default: detect NVIDIA GPU
  res <- FALSE
  try({ res <- check_nvidia_gpu() }, silent = TRUE)
  isTRUE(res)
}

#' @noRd
.te_torch_requirements <- function(use_gpu = FALSE,
                                   sysname = Sys.info()[["sysname"]],
                                   machine = Sys.info()[["machine"]]) {
  x86_64 <- machine %in% c("x86_64", "x86-64", "AMD64")

  # PyTorch stopped publishing Intel macOS wheels after 2.2
  if (identical(sysname, "Darwin") && x86_64) {
    return(c("torch==2.2.2", "torchvision==0.17.2"))
  }

  # PyPI's Linux x86_64 torch bundles about 2.5 GB of CUDA libraries, and its
  # Windows torch is CPU-only. The other builds exist only on the PyTorch
  # index; direct wheel URLs avoid adding that index, which would shadow PyPI
  # for every other package. CUDA 12.6 builds run on older NVIDIA drivers than
  # the CUDA 13 builds on PyPI.
  flavor <- if (identical(sysname, "Linux") && x86_64) {
    if (isTRUE(use_gpu)) "cu126" else "cpu"
  } else if (identical(sysname, "Windows") && x86_64 && isTRUE(use_gpu)) {
    "cu126"
  }

  if (is.null(flavor)) {
    return(c(
      paste0("torch==", .te_torch_version),
      paste0("torchvision==", .te_torchvision_version)
    ))
  }

  platform <- if (identical(sysname, "Linux")) "manylinux_2_28_x86_64" else "win_amd64"
  wheel <- function(pkg, version) {
    sprintf(
      "%s @ https://download.pytorch.org/whl/%s/%s-%s%%2B%s-cp312-cp312-%s.whl",
      pkg, flavor, pkg, version, flavor, platform
    )
  }
  c(wheel("torch", .te_torch_version), wheel("torchvision", .te_torchvision_version))
}

#' @noRd
.te_py_requirements <- function(feature, use_gpu = FALSE,
                                sysname = Sys.info()[["sysname"]],
                                machine = Sys.info()[["machine"]]) {
  switch(feature,
    core = c(
      .te_torch_requirements(use_gpu = use_gpu, sysname = sysname, machine = machine),
      # The transformers, huggingface-hub and numpy caps keep the core
      # compatible with the RAG stack below (llama-index 0.10 needs
      # huggingface-hub<0.24 and numpy<2), so rag() can add it to a running
      # session without changing loaded packages. reticulate does not let
      # packages set exclude_newer, so the remaining caps stop new major
      # releases from being picked up.
      "transformers>=4.40,<4.47",
      "huggingface-hub<0.24",
      "numpy<2",
      "pandas<3",
      "accelerate<2",
      "safetensors<1",
      "sentencepiece<1",
      "sentence-transformers<6",
      "timm<2",
      "einops<1",
      "opencv-python-headless<5"
    ),
    rag = c(
      # llama-index-core rather than the llama-index meta-package, which also
      # installs llama-index-legacy and the OpenAI integrations
      "llama-index-core>=0.10.30,<0.11",
      "llama-index-llms-huggingface",
      "llama-index-embeddings-huggingface",
      "pypdf",
      "rank-bm25"
    ),
    youtube = "pytubefix",
    findingemo = c("findingemo-light", "termcolor"),
    # 4-bit quantization for EVA-CLIP-8B
    gpu = "bitsandbytes",
    stop("Unknown Python feature set: ", feature, call. = FALSE)
  )
}

#' @noRd
.te_check_extras <- function(extras) {
  unknown <- setdiff(extras, c("rag", "youtube", "findingemo", "gpu"))
  if (length(unknown)) {
    stop("Unknown extras: ", paste(unknown, collapse = ", "),
         ". Choose from: rag, youtube, findingemo, gpu.", call. = FALSE)
  }
  unique(extras)
}

#' Python Requirements for transforEmotion
#'
#' @description
#' Returns the Python requirements that transforEmotion installs, in
#' \code{requirements.txt} format. Use it to build a fixed Python environment,
#' for example in a container image, and point transforEmotion at it with the
#' \code{TRANSFOREMOTION_PYTHON} environment variable (see Details).
#'
#' @param extras Character vector of optional feature sets to include; see
#' \code{\link{setup_modules}}.
#' @param gpu Logical. Use the CUDA build of PyTorch (Linux and Windows)
#' instead of the CPU build.
#' @param sysname,machine Target operating system and architecture, as
#' returned by \code{Sys.info()}. Default to the current machine.
#'
#' @details
#' The PyTorch requirements are direct wheel URLs built for Python 3.12, so the
#' environment must use Python 3.12.
#'
#' When \code{TRANSFOREMOTION_PYTHON} is set to a Python executable,
#' transforEmotion uses that interpreter and does not install anything. This is
#' the route for read-only environments such as Apptainer images, where uv
#' cannot write to its cache.
#'
#' @return A character vector of requirement specifiers.
#'
#' @examples
#' # Requirements for a CPU environment with the rag() packages
#' python_requirements(extras = "rag")
#'
#' \dontrun{
#' # Write a requirements file and build the environment with uv (shell):
#' #   uv venv --python 3.12 /opt/te-venv
#' #   uv pip install --python /opt/te-venv/bin/python -r requirements.txt
#' writeLines(python_requirements(extras = "rag"), "requirements.txt")
#' }
#'
#' @export
python_requirements <- function(extras = character(), gpu = FALSE,
                                sysname = Sys.info()[["sysname"]],
                                machine = Sys.info()[["machine"]]) {
  extras <- .te_check_extras(extras)
  unique(unlist(lapply(
    c("core", extras), .te_py_requirements,
    use_gpu = isTRUE(gpu), sysname = sysname, machine = machine
  )))
}

#' @noRd
# Declare Python requirements for one or more feature sets. "core" is always
# declared first. Before Python starts this only records requirements; after
# it starts, reticulate installs the additions into the running session.
.te_require <- function(features = "core") {
  if (identical(Sys.getenv("RETICULATE_AUTOCONFIGURE", unset = ""), "")) {
    Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  }

  # A fixed environment (for example in a container image) already holds
  # every package; use it instead of declaring requirements
  python <- Sys.getenv("TRANSFOREMOTION_PYTHON", unset = "")
  if (nzchar(python)) {
    if (!isTRUE(.te_py_state$fixed_python)) {
      reticulate::use_python(python, required = TRUE)
      .te_py_state$fixed_python <- TRUE
    }
    return(invisible(TRUE))
  }

  features <- unique(c("core", features))
  todo <- setdiff(features, .te_py_state$features)
  if (!length(todo)) return(invisible(TRUE))

  # Only the core set depends on the GPU (it picks the PyTorch build)
  if ("core" %in% todo) .te_py_state$use_gpu <- te_should_use_gpu()
  for (feature in todo) {
    reticulate::py_require(
      packages = .te_py_requirements(feature, use_gpu = .te_py_state$use_gpu),
      python_version = .te_python_version
    )
    .te_py_state$features <- c(.te_py_state$features, feature)
  }
  invisible(TRUE)
}

#' @noRd
# Whether the declared PyTorch build is the CUDA one
.te_uses_gpu <- function() {
  if (!"core" %in% .te_py_state$features) return(te_should_use_gpu())
  isTRUE(.te_py_state$use_gpu)
}

#' @noRd
# Switch the declared PyTorch build between CPU and CUDA before Python starts
.te_set_torch_flavor <- function(use_gpu) {
  # A fixed environment already contains its PyTorch build
  if (nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON", unset = ""))) return(invisible(TRUE))
  use_gpu <- isTRUE(use_gpu)
  if (!"core" %in% .te_py_state$features) {
    .te_require("core")
  }
  if (identical(.te_py_state$use_gpu, use_gpu)) return(invisible(TRUE))
  if (reticulate::py_available(initialize = FALSE)) {
    stop("Python has already started in this session. Restart R and set ",
         "the PyTorch build before running any analysis.", call. = FALSE)
  }
  reticulate::py_require(.te_torch_requirements(.te_py_state$use_gpu), action = "remove")
  reticulate::py_require(.te_torch_requirements(use_gpu))
  .te_py_state$use_gpu <- use_gpu
  invisible(TRUE)
}

#' @noRd
# llama-index downloads NLTK data into each Python environment on first
# import. A single folder in the package cache lets data downloaded once (for
# example by setup_modules(extras = "rag")) serve every environment and
# offline sessions. An NLTK_DATA set by the user takes precedence.
.te_use_nltk_cache <- function() {
  nltk_dir <- file.path(tools::R_user_dir("transforEmotion", "cache"), "nltk_data")
  reticulate::py_run_string(sprintf(
    "import os; os.environ.setdefault('NLTK_DATA', %s)",
    encodeString(normalizePath(nltk_dir, winslash = "/", mustWork = FALSE), quote = "'")
  ))
  invisible(TRUE)
}

#' @noRd
# Download the NLTK data llama-index looks for. llama-index 0.10 checks for
# "punkt" but downloads "punkt_tab", so without "punkt" it tries to download
# again on every import, which fails offline.
.te_download_nltk_data <- function() {
  .te_use_nltk_cache()
  os <- reticulate::import("os")
  nltk <- reticulate::import("nltk")
  for (pkg in c("stopwords", "punkt", "punkt_tab")) {
    nltk$download(pkg, download_dir = os$environ[["NLTK_DATA"]], quiet = TRUE)
  }
  invisible(TRUE)
}

#' @noRd
# Declare the core Python requirements; called at the top of every function
# that uses Python.
ensure_te_py_env <- function() {
  .te_require("core")
}

#' @noRd
.te_py_list_packages <- function() {
  pkgs <- try(reticulate::py_list_packages(), silent = TRUE)
  if (inherits(pkgs, "try-error") || !is.data.frame(pkgs) || !nrow(pkgs)) {
    return(data.frame())
  }
  if (!("package" %in% names(pkgs))) return(data.frame())
  pkgs
}

#' @noRd
.te_llama_pkg_versions <- function() {
  pkgs <- .te_py_list_packages()
  if (!nrow(pkgs)) return("unavailable")
  idx <- grepl("^llama-index", pkgs$package)
  if (!any(idx)) return("none detected")
  version_col <- if ("version" %in% names(pkgs)) pkgs$version else rep("unknown", nrow(pkgs))
  entries <- paste0(pkgs$package[idx], "==", version_col[idx])
  paste(entries, collapse = ", ")
}

#' @noRd
te_validate_modern_llama_index <- function(llama_index = NULL, stop_on_error = TRUE) {
  failures <- character()

  if (is.null(llama_index)) {
    llama_index <- try(reticulate::import("llama_index"), silent = TRUE)
  }

  if (inherits(llama_index, "try-error") || is.null(llama_index)) {
    failures <- c(failures, "Could not import 'llama_index'.")
  }

  core <- try(reticulate::import("llama_index.core"), silent = TRUE)
  if (inherits(core, "try-error") || is.null(core)) {
    failures <- c(failures, "Could not import 'llama_index.core'.")
  } else {
    settings_ok <- FALSE
    try({
      settings_ok <- reticulate::py_has_attr(core, "Settings")
    }, silent = TRUE)
    if (!isTRUE(settings_ok)) {
      failures <- c(failures, "'llama_index.core.Settings' is not available.")
    }
  }

  hf_llm <- try(reticulate::import("llama_index.llms.huggingface"), silent = TRUE)
  if (inherits(hf_llm, "try-error") || is.null(hf_llm)) {
    failures <- c(failures, "Could not import 'llama_index.llms.huggingface'.")
  }

  hf_embed <- try(reticulate::import("llama_index.embeddings.huggingface"), silent = TRUE)
  if (inherits(hf_embed, "try-error") || is.null(hf_embed)) {
    failures <- c(failures, "Could not import 'llama_index.embeddings.huggingface'.")
  }

  if (!length(failures)) return(invisible(TRUE))

  msg <- paste0(
    "Incompatible llama-index Python environment for transforEmotion.\n",
    paste(failures, collapse = " "),
    "\nDetected llama-index packages: ", .te_llama_pkg_versions(), "\n",
    "Restart R and run setup_modules(extras = \"rag\")."
  )

  if (isTRUE(stop_on_error)) stop(msg, call. = FALSE)
  invisible(FALSE)
}
