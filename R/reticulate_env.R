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
# Extra feature sets found in a fixed environment (TRANSFOREMOTION_PYTHON)
.te_py_state$fixed_features <- character()

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
           "google.protobuf", "sentence_transformers", "timm", "einops", "cv2"),
  rag = c("llama_index.core", "llama_index.llms.huggingface",
          "llama_index.embeddings.huggingface", "pypdf", "rank_bm25"),
  youtube = "pytubefix",
  audio = "av",
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
      # transformers needs protobuf to read a SentencePiece model when it
      # builds a fast tokenizer for a repository without tokenizer.json, for
      # example joeddav/xlm-roberta-large-xnli. transformers[sentencepiece]
      # declares it; plain sentencepiece does not. 3.20 is the first release
      # with google.protobuf.internal.builder, which sentencepiece imports.
      "protobuf>=3.20,<8",
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
    # Audio decoding for speech_scores(); the wheels bundle FFmpeg
    audio = "av>=14,<20",
    findingemo = c("findingemo-light", "termcolor"),
    # 4-bit quantization for EVA-CLIP-8B
    gpu = "bitsandbytes",
    stop("Unknown Python feature set: ", feature, call. = FALSE)
  )
}

#' @noRd
.te_check_extras <- function(extras) {
  unknown <- setdiff(extras, c("rag", "youtube", "audio", "findingemo", "gpu"))
  if (length(unknown)) {
    stop("Unknown extras: ", paste(unknown, collapse = ", "),
         ". Choose from: rag, youtube, audio, findingemo, gpu.", call. = FALSE)
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
#' A function that needs an optional feature set (\code{extras}), for example
#' \code{rag()}, stops when the fixed environment does not have the Python
#' modules of that set. The error names the missing modules, a command that
#' installs their packages, and the \code{EXTRAS} build argument of
#' \code{docker/Dockerfile}. Use \code{\link{setup_modules}} with the same
#' \code{extras} to compare the whole environment with the requirements.
#'
#' A fixed environment does not change when transforEmotion is updated, and a
#' new version can add Python packages. After an update, rebuild the
#' environment or the image from \code{python_requirements()}, or install the
#' new packages into it. For a read-only image that you cannot rebuild,
#' install a missing package into a writable folder on a machine with network
#' access, for example
#' \code{uv pip install --python <python> --target <folder> "protobuf>=3.20,<8"},
#' and start R with the environment variable \code{PYTHONPATH} set to that
#' folder.
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
# Whether the torch of the running Python is a CUDA build, from its package
# metadata, so torch is not imported. NA when torch is not installed. The
# PyTorch index marks its CUDA builds with "+cu126"; the CUDA build on PyPI
# (Linux x86_64) has a plain version and requires the nvidia-* packages.
.te_torch_is_cuda <- function(metadata = reticulate::import("importlib.metadata")) {
  torch <- tryCatch(as.character(metadata$version("torch")), error = function(e) NA_character_)
  if (is.na(torch)) return(NA)
  requires <- tryCatch(unlist(metadata$requires("torch")), error = function(e) NULL)
  grepl("+cu", torch, fixed = TRUE) || any(startsWith(as.character(requires), "nvidia-"))
}

#' @noRd
# Requirements of the core set and of extras that the running Python does not
# meet: the package is missing, or its version is outside the specifier. Reads
# package metadata, so no package is imported. A direct URL (the PyTorch
# wheels) is checked only for presence. Returns what is installed for each
# unmet requirement ("" when nothing is), named by the requirement, and the
# PyTorch build (gpu) of the requirements: that of the installed torch.
.te_unmet_requirements <- function(extras = character()) {
  metadata <- reticulate::import("importlib.metadata")
  installed <- function(name) {
    tryCatch(as.character(metadata$version(name)), error = function(e) NA_character_)
  }
  # packaging comes with transformers; without it, check only presence
  requirement <- tryCatch(reticulate::import("packaging.requirements")$Requirement,
                          error = function(e) NULL)
  gpu <- .te_torch_is_cuda(metadata)
  if (is.na(gpu)) gpu <- te_should_use_gpu()

  unmet <- character()
  for (req in python_requirements(extras, gpu = gpu)) {
    parsed <- if (!is.null(requirement)) tryCatch(requirement(req), error = function(e) NULL)
    name <- if (is.null(parsed)) sub("^([A-Za-z0-9._-]+).*$", "\\1", req) else parsed$name
    # Every OpenCV build provides cv2
    if (identical(name, "opencv-python-headless")) {
      name <- c(name, "opencv-python", "opencv-contrib-python")
    }
    versions <- vapply(name, installed, character(1))
    found <- which(!is.na(versions))[1L]
    if (is.na(found)) {
      unmet[req] <- ""
    } else if (!is.null(parsed) && is.null(parsed$url)) {
      # packaging before 25 raises InvalidVersion for a version outside
      # PEP 440; later releases return False
      ok <- tryCatch(parsed$specifier$contains(versions[[found]], prereleases = TRUE),
                     error = function(e) FALSE)
      if (!isTRUE(ok)) unmet[req] <- paste(name[[found]], versions[[found]])
    }
  }
  list(unmet = unmet, gpu = gpu)
}

#' @noRd
# A Python path for a printed shell command: absolute, with "~" expanded,
# which the shell does not do inside quotes. Only the folder is resolved: in
# a virtual environment bin/python is a link to the base interpreter, and
# uv pip install --python with the resolved path installs into that
# interpreter instead of the environment.
.te_shell_python <- function(python) {
  file.path(normalizePath(dirname(python), winslash = "/", mustWork = FALSE),
            basename(python))
}

#' @noRd
# Warn about the requirements that the fixed Python environment
# (TRANSFOREMOTION_PYTHON) does not meet; transforEmotion installs nothing
# into it. A fixed environment built for an older version can lack packages
# or versions that a newer version needs.
.te_warn_unmet_requirements <- function(extras = character()) {
  check <- .te_unmet_requirements(extras)
  unmet <- check$unmet
  if (!length(unmet)) return(invisible(TRUE))
  warning(
    "The Python environment that TRANSFOREMOTION_PYTHON points to does not ",
    "meet these requirements:\n",
    paste0("  ", names(unmet), ": ",
           ifelse(nzchar(unmet), paste(unmet, "is installed"), "not installed"),
           collapse = "\n"),
    "\n", .te_fixed_python_fix(names(unmet), extras, check$gpu),
    " For a read-only image, see ?python_requirements.",
    call. = FALSE
  )
  invisible(FALSE)
}

#' @noRd
# The fix for requirements that the fixed Python environment does not meet: a
# uv command that installs them, and the python_requirements() call that
# gives the requirements to rebuild the environment
.te_fixed_python_fix <- function(requirements, extras = character(), gpu = FALSE) {
  python <- .te_shell_python(Sys.getenv("TRANSFOREMOTION_PYTHON"))
  args <- c(
    if (length(extras)) paste0("extras = ", paste(deparse(extras), collapse = "")),
    if (isTRUE(gpu)) "gpu = TRUE"
  )
  paste0(
    "transforEmotion does not install packages into a fixed environment. ",
    "Install them with\n",
    "  uv pip install --python \"", python, "\" ",
    paste0("\"", requirements, "\"", collapse = " "), "\n",
    "or rebuild the environment from python_requirements(",
    paste(args, collapse = ", "), ")."
  )
}

#' @noRd
# Stop when the fixed Python environment lacks a module of a requested extra
# feature set, before the function fails on the import. importlib.util's
# find_spec() finds a module without running it; for a dotted name it imports
# only the parent packages, which are namespace packages here. A set that
# passes is not checked again in this session. On failure, the requirements
# to install come from .te_unmet_requirements(), as in setup_modules(), but
# only those of the missing sets.
.te_check_fixed_features <- function(features) {
  todo <- setdiff(features, c("core", .te_py_state$fixed_features))
  if (!length(todo)) return(invisible(TRUE))
  util <- reticulate::import("importlib.util")
  # find_spec() raises ModuleNotFoundError when a parent package is missing
  found <- function(module) {
    tryCatch(!is.null(util$find_spec(module)), error = function(e) FALSE)
  }
  missing <- list()
  for (feature in todo) {
    modules <- .te_feature_modules[[feature]]
    absent <- modules[!vapply(modules, found, logical(1))]
    if (length(absent)) {
      missing[[feature]] <- absent
    } else {
      .te_py_state$fixed_features <- c(.te_py_state$fixed_features, feature)
    }
  }
  if (!length(missing)) return(invisible(TRUE))

  sets <- names(missing)
  check <- .te_unmet_requirements(sets)
  gpu <- isTRUE(check$gpu)
  # The check also reports unmet requirements of the core set. Installing
  # them could change packages that work, for example a newer transformers.
  requirements <- unlist(lapply(sets, function(set) {
    own <- .te_py_requirements(set, use_gpu = gpu)
    unmet <- intersect(names(check$unmet), own)
    # A module can be missing while its package is installed (a broken install)
    if (length(unmet)) unmet else own
  }), use.names = FALSE)
  # A rebuild also needs the sets that were found
  rebuild <- intersect(names(.te_feature_modules), c(.te_py_state$fixed_features, sets))
  # GPU=1 adds the gpu set
  docker_extras <- setdiff(rebuild, if (gpu) "gpu")
  docker_args <- c(
    if (length(docker_extras)) paste0("--build-arg EXTRAS=", paste(docker_extras, collapse = ",")),
    if (gpu) "--build-arg GPU=1"
  )
  stop(
    "The Python environment that TRANSFOREMOTION_PYTHON points to does not ",
    "have these modules of the feature sets that this function needs:\n",
    paste0("  ", sets, ": ", vapply(missing, paste, "", collapse = ", "),
           collapse = "\n"),
    "\n", .te_fixed_python_fix(requirements, rebuild, gpu),
    " For a Docker image, build it from docker/Dockerfile with ",
    paste(docker_args, collapse = " "), ".",
    if (gpu) " GPU=1 gives the CUDA build of PyTorch and adds the gpu set.",
    " In extras and in EXTRAS, also list every other feature set that the ",
    "environment needs.",
    call. = FALSE
  )
}

#' @noRd
# Declare Python requirements for one or more feature sets. "core" is always
# declared first. Before Python starts this only records requirements; after
# it starts, reticulate installs the additions into the running session.
# check_cache = FALSE skips the offline check of the uv cache, for .onLoad and
# for callers that change the requirements again before Python starts.
.te_require <- function(features = "core", check_cache = TRUE) {
  if (identical(Sys.getenv("RETICULATE_AUTOCONFIGURE", unset = ""), "")) {
    Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  }

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
    # Nothing can be added to it; stop if it lacks a requested extra set
    .te_check_fixed_features(features)
    return(invisible(TRUE))
  }

  features <- unique(c("core", features))
  todo <- setdiff(features, .te_py_state$features)

  # Only the core set depends on the GPU (it picks the PyTorch build)
  if ("core" %in% todo) .te_py_state$use_gpu <- te_should_use_gpu()
  # Once Python runs, reticulate installs additions at once and only warns
  # when it cannot: offline, or when a package is already declared with
  # another version constraint (Python started before transforEmotion was
  # loaded). A running Python that already has the packages is used as is;
  # otherwise stop, so the failure is reported here and the feature set is
  # retried on the next call
  started <- reticulate::py_available(initialize = FALSE)
  for (feature in todo) {
    withCallingHandlers(
      reticulate::py_require(
        packages = .te_py_requirements(feature, use_gpu = .te_py_state$use_gpu),
        python_version = .te_python_version
      ),
      warning = function(w) {
        if (!started) return()
        modules <- .te_feature_modules[[feature]]
        if (all(vapply(modules, reticulate::py_module_available, logical(1)))) {
          invokeRestart("muffleWarning")
        }
        stop("Could not install the Python packages for '", feature, "': ",
             conditionMessage(w), "\nRestart R and load transforEmotion before ",
             "anything starts Python, or set TRANSFOREMOTION_PYTHON to an ",
             "environment that has the packages.",
             if (.te_uv_offline()) {
               paste0("\n", .te_offline_fix(c(.te_py_state$features, feature)))
             },
             call. = FALSE)
      }
    )
    .te_py_state$features <- c(.te_py_state$features, feature)
  }

  if (check_cache) .te_check_offline_cache()
  invisible(TRUE)
}

#' @noRd
# uv runs without network access (setup_cache(offline = TRUE) sets
# UV_OFFLINE). These are the values uv reads as true.
.te_uv_offline <- function() {
  tolower(Sys.getenv("UV_OFFLINE", unset = "")) %in% c("1", "true", "t", "yes", "y", "on")
}

#' @noRd
# Whether reticulate will start Python from its uv environment. Before
# Python starts, py_discover_config() returns the Python that reticulate
# finds first (RETICULATE_PYTHON, VIRTUAL_ENV, use_python(), a venv in the
# working directory, ~/.virtualenvs/r-reticulate and others), and NULL where
# reticulate would build its uv environment.
.te_uses_uv_python <- function() {
  if (tolower(Sys.getenv("RETICULATE_USE_MANAGED_VENV")) %in% c("false", "0", "no")) {
    return(FALSE)
  }
  config <- tryCatch(suppressWarnings(reticulate::py_discover_config()),
                     error = function(e) e)
  # An error here (for example a RETICULATE_PYTHON that does not exist)
  # comes again from reticulate when Python starts
  if (inherits(config, "error")) return(FALSE)
  is.null(config) || isTRUE(config$ephemeral)
}

#' @noRd
# Offline, check before Python starts that uv can build the declared Python
# environment from its cache. Without this check, a package that the cache
# does not have stops the start of Python with reticulate's "Installation of
# Python not found". uv_run_tool() runs the uv command that reticulate runs
# when Python starts, in a subprocess, and builds the same cached
# environment; Python does not start here. A passed check is kept for the
# session. After a failure the next call checks again, so a cache filled in
# the meantime, or another Python, is used without a restart of R.
.te_check_offline_cache <- function() {
  if (!.te_uv_offline() || reticulate::py_available(initialize = FALSE)) {
    return(invisible(TRUE))
  }
  reqs <- reticulate::py_require()
  # uv_run_tool() keeps the resolved Python by this value; it must be one string
  python_version <- if (length(reqs$python_version)) {
    paste(reqs$python_version, collapse = ",")
  }
  key <- paste(c(reqs$packages, python_version, reqs$exclude_newer,
                 Sys.getenv("UV_CACHE_DIR")), collapse = "\n")
  if (identical(.te_py_state$cache_ok, key) || !.te_uses_uv_python()) {
    return(invisible(TRUE))
  }
  # uv's output is kept for the error, so it does not show while uv builds
  # the environment
  message("UV_OFFLINE is set; checking the Python packages in the uv cache ...")
  out <- tryCatch(
    suppressWarnings(reticulate::uv_run_tool(
      "python", c("-c", "pass"),
      with = reqs$packages,
      python_version = python_version,
      exclude_newer = reqs$exclude_newer,
      stdout = TRUE, stderr = TRUE
    )),
    # For example, reticulate cannot download uv offline
    error = function(e) structure(conditionMessage(e), status = -1L)
  )
  if (!is.null(attr(out, "status"))) {
    stop(.te_offline_cache_error(out, reqs), call. = FALSE)
  }
  .te_py_state$cache_ok <- key
  invisible(TRUE)
}

#' @noRd
# The error for a failed offline check: uv's output from its first "error:"
# line on (the lines before it report progress), and the fix
.te_offline_cache_error <- function(out, reqs) {
  first <- grep("^error:", out)
  if (length(first)) out <- out[first[1L]:length(out)]
  out <- out[nzchar(trimws(out))]
  # Packages declared with py_require() by the user or by another package.
  # reticulate itself declares numpy, and ipykernel in Positron.
  ours <- unlist(lapply(.te_py_state$features, .te_py_requirements,
                        use_gpu = isTRUE(.te_py_state$use_gpu)))
  own <- unlist(lapply(reqs$history, function(entry) {
    if (identical(entry$requested_from, "reticulate")) entry$packages
  }))
  other <- setdiff(reqs$packages, c(ours, own, "numpy"))
  paste0(
    "uv cannot build the Python environment from its cache without network ",
    "access. uv reported:\n", paste0("  ", out, collapse = "\n"), "\n",
    .te_offline_fix(other = other)
  )
}

#' @noRd
# What to run with network access when the uv cache does not have the Python
# packages that an offline session needs. setup_cache() keeps uv's cache in
# <path>/uv. Any other cache, and UV_OFFLINE with it, comes from the
# environment of the session, so the lines set them directly.
.te_offline_fix <- function(features = .te_py_state$features, other = character()) {
  cache <- Sys.getenv("UV_CACHE_DIR", unset = "")
  gpu <- isTRUE(.te_py_state$use_gpu)
  extras <- setdiff(features, "core")
  setup <- identical(basename(cache), "uv")
  lines <- if (setup) {
    c("library(transforEmotion)",
      paste0("setup_cache(", deparse(dirname(cache)), ", gpu = ", gpu, ")"))
  } else {
    c(if (nzchar(cache)) {
        c('Sys.unsetenv("UV_OFFLINE")', paste0("Sys.setenv(UV_CACHE_DIR = ", deparse(cache), ")"))
      } else {
        'Sys.unsetenv(c("UV_OFFLINE", "UV_CACHE_DIR"))'
      },
      # The PyTorch build is chosen when the package loads
      paste0('Sys.setenv(TRANSFOREMOTION_USE_GPU = "', as.integer(gpu), '")'),
      "library(transforEmotion)")
  }
  lines <- c(lines, paste0(
    "setup_modules(",
    if (length(extras)) paste0("extras = ", paste(deparse(extras), collapse = "")),
    ")"
  ))
  paste0(
    "UV_OFFLINE is set, so uv can install only packages that are in its ",
    "cache. A new version of transforEmotion can require packages that the ",
    "cache does not have. To add them, run these lines once in a new R ",
    "session with network access", if (setup) ", without offline = TRUE", ":\n",
    paste0("  ", lines, "\n", collapse = ""),
    if (length(other)) {
      paste0("Declare ", paste(other, collapse = ", "), " with ",
             "reticulate::py_require() in that session too.\n")
    },
    "Then run the analysis again."
  )
}

#' @noRd
# Whether the PyTorch build is the CUDA one: in a fixed environment
# (TRANSFOREMOTION_PYTHON) the installed build, otherwise the declared one.
# A GPU on the host does not count for a fixed environment with the CPU
# build, for example the CPU Docker image run with --gpus all.
.te_uses_gpu <- function() {
  if (nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON", unset = ""))) {
    .te_require("core")
    return(isTRUE(.te_torch_is_cuda()))
  }
  if (!"core" %in% .te_py_state$features) return(te_should_use_gpu())
  isTRUE(.te_py_state$use_gpu)
}

#' @noRd
# Switch the declared PyTorch build between CPU and CUDA before Python starts
.te_set_torch_flavor <- function(use_gpu) {
  # A fixed environment already contains its PyTorch build
  if (nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON", unset = ""))) return(invisible(TRUE))
  use_gpu <- isTRUE(use_gpu)
  # The check of the cache waits for the final PyTorch build
  if (!"core" %in% .te_py_state$features) {
    .te_require("core", check_cache = FALSE)
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
