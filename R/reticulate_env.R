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

# Import names of every package in each feature set (.te_py_requirements()),
# to tell whether a running Python already has the whole set
.te_feature_modules <- list(
  core = c("torch", "torchvision", "transformers", "huggingface_hub", "numpy",
           "pandas", "accelerate", "safetensors", "sentencepiece",
           "sentence_transformers", "timm", "einops", "cv2"),
  rag = c("llama_index.core", "llama_index.llms.huggingface",
          "llama_index.embeddings.huggingface", "pypdf", "rank_bm25"),
  youtube = "pytubefix",
  findingemo = c("findingemo_light", "termcolor"),
  gpu = "bitsandbytes"
)

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
# The only writer of RETICULATE_PYTHON. Before Python starts, point reticulate
# at transforEmotion's Python for this R session: TRANSFOREMOTION_PYTHON if
# set, otherwise "managed" (reticulate's uv environment built from
# py_require()). RETICULATE_PYTHON outranks every other discovery route
# (RETICULATE_PYTHON_ENV, use_virtualenv(required = TRUE), VIRTUAL_ENV,
# ./.venv, the r-reticulate virtualenv), so no leftover from an older setup
# can select an old Python. A replaced value is kept for .onAttach,
# fix_python() and .te_release_python().
.te_claim_python <- function() {
  if (reticulate::py_available(initialize = FALSE)) return(invisible(FALSE))
  target <- Sys.getenv("TRANSFOREMOTION_PYTHON", unset = "")
  if (!nzchar(target)) target <- "managed"
  old <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  if (nzchar(old) && !identical(old, target) && !identical(old, .te_py_state$claimed)) {
    .te_py_state$replaced_python <- old
  }
  Sys.setenv(RETICULATE_PYTHON = target)
  .te_py_state$claimed <- target
  invisible(TRUE)
}

#' @noRd
# Undo the claim when the package is unloaded, unless something else has
# changed RETICULATE_PYTHON since
.te_release_python <- function() {
  claimed <- .te_py_state$claimed
  if (is.null(claimed) || !identical(Sys.getenv("RETICULATE_PYTHON"), claimed)) {
    return(invisible(FALSE))
  }
  replaced <- .te_py_state$replaced_python
  if (is.null(replaced)) Sys.unsetenv("RETICULATE_PYTHON")
  else Sys.setenv(RETICULATE_PYTHON = replaced)
  .te_py_state$claimed <- NULL
  invisible(TRUE)
}

#' @noRd
# Which Python this session uses. py_config() is called only after Python has
# started, so this never starts Python. reticulate (>= 1.45.0) sets
# `ephemeral` only for its managed uv environment.
.te_python_in_use <- function() {
  if (!reticulate::py_available(initialize = FALSE)) {
    return(list(state = "not_started", path = NA_character_,
                version = NA_character_, reason = NA_character_))
  }
  cfg <- reticulate::py_config()
  list(
    state = if (isTRUE(cfg$ephemeral)) "managed" else "foreign",
    path = cfg$python,
    version = as.character(cfg$version),
    reason = if (is.null(cfg$forced)) NA_character_ else cfg$forced
  )
}

#' @noRd
# Text for a Python that transforEmotion did not build. `missing` lists the
# modules it lacks, or is NULL when that was not checked.
.te_foreign_python_message <- function(python, missing = NULL) {
  selected <- if (!is.na(python$reason)) paste0(", selected by ", python$reason)
  fixed <- nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON", unset = ""))
  paste(c(
    "transforEmotion cannot use the Python running in this session:",
    paste0(python$path, " (Python ", python$version, selected, ")."),
    if (length(missing)) paste0("It has no module named: ", paste(missing, collapse = ", "), "."),
    if (fixed) c(
      "TRANSFOREMOTION_PYTHON selects this Python.",
      "Install the packages from python_requirements() into it, or unset TRANSFOREMOTION_PYTHON."
    ) else c(
      "Python started before transforEmotion could select its own environment.",
      "Restart R. Run library(transforEmotion) before anything that starts Python.",
      "Run transforEmotion::fix_python() to find what selected this Python.",
      "To use your own Python on purpose, set TRANSFOREMOTION_PYTHON to it."
    )
  ), collapse = "\n")
}

#' @noRd
# A running Python that transforEmotion did not build: started before the
# package was loaded, or TRANSFOREMOTION_PYTHON. Stops when modules are
# missing. When all are present but the Python version is outside
# .te_python_version (older setups used 3.10), warns once per session and
# continues; not for TRANSFOREMOTION_PYTHON, which is an explicit choice.
.te_check_foreign <- function(features) {
  todo <- setdiff(unique(c("core", features)), .te_py_state$features)
  if (!length(todo)) return(invisible(TRUE))
  python <- .te_python_in_use()
  modules <- unique(unlist(.te_feature_modules[todo], use.names = FALSE))
  missing <- modules[!vapply(modules, reticulate::py_module_available, logical(1))]
  if (length(missing)) {
    stop(structure(
      class = c("te_python_error", "error", "condition"),
      list(message = .te_foreign_python_message(python, missing), call = NULL)
    ))
  }
  fixed <- nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON", unset = ""))
  if (!fixed && !isTRUE(.te_py_state$version_warned) &&
      !.te_python_supported(python$version)) {
    .te_py_state$version_warned <- TRUE
    warning(
      "transforEmotion runs on a Python it did not set up: ", python$path,
      " (Python ", python$version, ", torch ", .te_py_package_version("torch"),
      ", transformers ", .te_py_package_version("transformers"), "). ",
      "It was built for Python ", .te_python_version, ", so results can differ. ",
      "Restart R and run library(transforEmotion) before anything that starts Python.",
      call. = FALSE
    )
  }
  .te_py_state$features <- union(.te_py_state$features, todo)
  invisible(TRUE)
}

#' @noRd
.te_python_supported <- function(version, constraint = .te_python_version) {
  parts <- strsplit(constraint, ",", fixed = TRUE)[[1]]
  ops <- sub("[0-9.]+$", "", parts)
  bounds <- sub("^[<>=]+", "", parts)
  v <- numeric_version(version, strict = FALSE)
  if (is.na(v)) return(FALSE)
  all(mapply(function(op, bound) match.fun(op)(v, numeric_version(bound)), ops, bounds))
}

#' @noRd
.te_py_package_version <- function(package) {
  tryCatch(
    as.character(reticulate::import("importlib.metadata")$version(package)),
    error = function(e) "unknown"
  )
}

#' @noRd
# Declare Python requirements for one or more feature sets. "core" is always
# declared first. Before Python starts this only records requirements; after
# it starts, reticulate installs the additions into the running session.
.te_require <- function(features = "core") {
  if (identical(Sys.getenv("RETICULATE_AUTOCONFIGURE", unset = ""), "")) {
    Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  }
  .te_claim_python()

  # A fixed environment (for example in a container image) already holds
  # every package; use it instead of declaring requirements
  python <- Sys.getenv("TRANSFOREMOTION_PYTHON", unset = "")
  if (nzchar(python)) {
    if (!isTRUE(.te_py_state$fixed_python)) {
      if (!file.exists(python)) {
        stop("TRANSFOREMOTION_PYTHON is set to '", python, "', which does not ",
             "exist. Point it at a Python executable, or unset it to let ",
             "transforEmotion manage Python.", call. = FALSE)
      }
      reticulate::use_python(python, required = TRUE)
      .te_py_state$fixed_python <- TRUE
    }
    if (.te_python_in_use()$state == "foreign") return(.te_check_foreign(features))
    return(invisible(TRUE))
  }

  features <- unique(c("core", features))
  todo <- setdiff(features, .te_py_state$features)
  if (!length(todo)) return(invisible(TRUE))

  # Only the core set depends on the GPU (it picks the PyTorch build)
  if ("core" %in% todo) .te_py_state$use_gpu <- te_should_use_gpu()
  state <- .te_python_in_use()$state
  if (state == "foreign") return(.te_check_foreign(todo))
  # Once the managed environment runs, reticulate installs additions at once
  # and only warns when it cannot: offline, or when a package is already
  # declared with another version constraint. A running Python that already
  # has the packages is used as is; otherwise stop, so the failure is
  # reported here and the feature set is retried on the next call
  for (feature in todo) {
    withCallingHandlers(
      reticulate::py_require(
        packages = .te_py_requirements(feature, use_gpu = .te_py_state$use_gpu),
        python_version = .te_python_version
      ),
      warning = function(w) {
        if (state != "managed") return()
        modules <- .te_feature_modules[[feature]]
        if (all(vapply(modules, reticulate::py_module_available, logical(1)))) {
          invokeRestart("muffleWarning")
        }
        stop("Could not install the Python packages for '", feature, "': ",
             conditionMessage(w), "\nRestart R and load transforEmotion before ",
             "anything starts Python, or set TRANSFOREMOTION_PYTHON to an ",
             "environment that has the packages.", call. = FALSE)
      }
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
# On CUDA, Triton (installed with PyTorch) compiles a small C helper with gcc
# the first time it runs, and needs Python.h. Embedded by reticulate in a uv
# environment, Python reports the environment's include folder, which has no
# headers, so every Triton kernel fails with "Python.h: No such file or
# directory". This adds the interpreter's real include folder to CPATH, which
# gcc reads. It only changes the Python process's environment, and does
# nothing when the headers are already where Python says.
.te_expose_python_headers <- function() {
  tryCatch(
    reticulate::py_run_string(local = TRUE, paste(
      "import os, sys, sysconfig",
      "if not os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')):",
      "    base = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))",
      "    inc = os.path.join(base, 'include', 'python%d.%d' % sys.version_info[:2])",
      "    paths = [p for p in os.environ.get('CPATH', '').split(os.pathsep) if p]",
      "    if os.path.exists(os.path.join(inc, 'Python.h')) and inc not in paths:",
      "        os.environ['CPATH'] = os.pathsep.join([inc] + paths)",
      sep = "\n"
    )),
    # Best effort: without it only Triton's compile step can fail, with its
    # own error
    error = function(e) invisible(NULL)
  )
  invisible(TRUE)
}

#' @noRd
# On Linux, R has already loaded its BLAS (libblas.so or libRblas.so) into the
# global symbol scope when Python starts. PyTorch's CPU build carries MKL
# inside libtorch_cpu.so and exports sgemm_/dgemm_, so without this the dynamic
# linker binds PyTorch's BLAS calls to R's library instead. With R's reference
# BLAS that makes every matrix product tens of times slower (a 1500 x 1500
# float matmul: 1.6 s instead of 0.03 s). Importing torch with RTLD_DEEPBIND
# makes it use its own symbols first; the default flags are restored after the
# import. Best effort: if torch is missing or the import fails here, the first
# real import reports the problem.
.te_import_torch_own_blas <- function() {
  if (!identical(Sys.info()[["sysname"]], "Linux")) return(invisible(FALSE))
  tryCatch(
    reticulate::py_run_string(local = TRUE, paste(
      "import os, sys",
      "if 'torch' not in sys.modules and hasattr(os, 'RTLD_DEEPBIND'):",
      "    _flags = sys.getdlopenflags()",
      "    sys.setdlopenflags(_flags | os.RTLD_DEEPBIND)",
      "    try:",
      "        import torch",
      "    finally:",
      "        sys.setdlopenflags(_flags)",
      sep = "\n"
    )),
    error = function(e) invisible(NULL)
  )
  invisible(TRUE)
}

#' @noRd
# The package's cache folder. tools::R_user_dir() exists from R 4.0; on older
# R, use the folder it returns on Linux
.te_user_cache_dir <- function(r_version = getRversion()) {
  if (r_version >= "4.0.0") return(tools::R_user_dir("transforEmotion", "cache"))
  root <- Sys.getenv("R_USER_CACHE_DIR", unset = "")
  if (!nzchar(root)) root <- Sys.getenv("XDG_CACHE_HOME", unset = path.expand("~/.cache"))
  file.path(root, "R", "transforEmotion")
}

#' @noRd
# llama-index downloads NLTK data into each Python environment on first
# import. A single folder in the package cache lets data downloaded once (for
# example by setup_modules(extras = "rag")) serve every environment and
# offline sessions. An NLTK_DATA set by the user takes precedence.
.te_use_nltk_cache <- function() {
  nltk_dir <- file.path(.te_user_cache_dir(), "nltk_data")
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
