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
.configure_uv_env <- function(use_gpu = FALSE) {
  # Define baseline packages and versions (aligned with previous setup)
  base_modules <- c(
    "numpy>=1.26,<2.0",
    "scipy==1.10.1",
    "accelerate==0.29.3",
    "llama-index==0.10.30",
    "nltk==3.8.1",
    "timm",
    "einops",
    "safetensors==0.4.3",
    "opencv-python==4.10.0.84",
    "pytubefix",
    "pandas==1.5.3",
    "pypdf==4.0.1",
    "pytz==2024.1",
    "qdrant-client==1.8.2",
    "sentencepiece==0.2.0",
    "sentence-transformers==2.2.2",
    "rank-bm25==0.2.2",
    "tokenizers==0.21.0",
    "findingemo-light",
    "transformers==4.51.0"
  )

  # Platform-specific additions
  OS <- tolower(Sys.info()["sysname"])  # linux, windows, darwin
  if (OS %in% c("linux", "windows")) {
    base_modules <- c(base_modules, "bitsandbytes==0.45.2")
  }

  # ML stack (CPU default)
  ml_modules <- if (isTRUE(use_gpu)) {
    # Note: Installing GPU wheels via uv may require configuring indices externally.
    c("tensorflow==2.14.1", "torch==2.1.1", "torchvision==0.16.1")
  } else {
    c("tensorflow-cpu==2.14.1", "torch==2.1.1", "torchvision==0.16.1")
  }

  # Workaround for uv first-match with PyTorch index:
  # qdrant-client==1.8.2 requires urllib3>=1.26.14,<3, but the PyTorch wheel index may expose 1.26.13 only.
  # Force urllib3 via a vetted direct wheel URL with SHA256 (PEP 508), bypassing index order while preserving
  # the PyTorch-first index for all other packages.
  pinned_urllib3 <- "urllib3 @ https://files.pythonhosted.org/packages/b0/53/aa91e163dcfd1e5b82d8a890ecf13314e3e149c05270cc644581f77f17fd/urllib3-1.26.18-py2.py3-none-any.whl#sha256=34b97092d7e0a3a8cf7cd10e386f401b3737364026c45e622aa02903dffe0f07"

  # llama-index-core (required by llama-index==0.10.30) needs requests>=2.31.0, but the PyTorch index may only provide 2.28.1.
  # Pin requests via direct URL to satisfy the constraint while keeping the PyTorch index first.
  pinned_requests <- "requests @ https://files.pythonhosted.org/packages/70/8e/0e2d847013cb52cd35b38c009bb167a1a26b2ce6cd6965bf26b47bc0bf44/requests-2.31.0-py3-none-any.whl#sha256=58cd2187c01e70e6e26505bca751777aa9f2ee0b7f4300988b709f44e013003f"

  modules <- c(base_modules, ml_modules, pinned_requests, pinned_urllib3)

  # For PyTorch wheels, set extra index for CPU/GPU flavors (best-effort)
  extra_index <- if (isTRUE(use_gpu)) "https://download.pytorch.org/whl/cu121" else "https://download.pytorch.org/whl/cpu"
  prev_pip_idx <- Sys.getenv("PIP_EXTRA_INDEX_URL", unset = "")
  prev_uv_idx  <- Sys.getenv("UV_EXTRA_INDEX_URL", unset = "")
  on.exit({
    if (nzchar(prev_pip_idx)) Sys.setenv(PIP_EXTRA_INDEX_URL = prev_pip_idx) else Sys.unsetenv("PIP_EXTRA_INDEX_URL")
    if (nzchar(prev_uv_idx))  Sys.setenv(UV_EXTRA_INDEX_URL  = prev_uv_idx)  else Sys.unsetenv("UV_EXTRA_INDEX_URL")
  }, add = TRUE)
  Sys.setenv(PIP_EXTRA_INDEX_URL = extra_index)
  Sys.setenv(UV_EXTRA_INDEX_URL  = extra_index)

  # Use Python 3.10 by default (allow latest micro): >=3.10,<3.11
  reticulate::py_require(
    packages = modules,
    python_version = ">=3.10,<3.11",
    action = "set"
    # You may add exclude_newer = "YYYY-MM-DD" here for strict reproducibility
  )

  # Initialize Python to realize the environment without relying on internal APIs.
  # reticulate will select the uv-managed environment declared via py_require().
  try(reticulate::py_available(initialize = TRUE), silent = TRUE)

  # Best-effort validation: show OpenCV version and haarcascades path
  try({
    reticulate::py_run_string(
      "import cv2; p=getattr(getattr(cv2,'data',None),'haarcascades', None); print('[transforEmotion] OpenCV', cv2.__version__, 'haarcascades:', p)"
    )
  }, silent = TRUE)

  invisible(TRUE)
}

#' @noRd
# Ensure Python is initialized via uv with the required packages
ensure_te_py_env <- function() {
  if (identical(Sys.getenv("RETICULATE_AUTOCONFIGURE", unset = ""), "")) {
    Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
  }

  if (!requireNamespace("reticulate", quietly = TRUE)) return(invisible(FALSE))

  # If Python already initialized, nothing to do
  initialized <- FALSE
  try({ initialized <- reticulate::py_available(initialize = FALSE) }, silent = TRUE)
  if (isTRUE(initialized)) return(invisible(TRUE))

  # Recommend uv if not available
  te_ensure_uv_available(prompt = TRUE)

  # Not initialized yet — configure uv environment (CPU default)
  use_gpu <- te_should_use_gpu()
  try(.configure_uv_env(use_gpu = use_gpu), silent = TRUE)
  invisible(TRUE)
}

#' @noRd
uv_is_available <- function() {
  nzchar(Sys.which("uv"))
}

#' @noRd
install_uv_interactive <- function(quiet = FALSE) {
  os <- tolower(Sys.info()[["sysname"]])  # linux, windows, darwin

  # Helpers return TRUE only on success (exit status 0)
  run_cmd <- function(cmd) {
    rc <- try(suppressWarnings(system(cmd)), silent = TRUE)
    !inherits(rc, "try-error") && is.numeric(rc) && identical(as.integer(rc), 0L)
  }
  run2 <- function(bin, args) {
    rc <- try(suppressWarnings(system2(bin, args)), silent = TRUE)
    !inherits(rc, "try-error") && is.numeric(rc) && identical(as.integer(rc), 0L)
  }

  if (os == "darwin") {
    # Prefer Homebrew when available, per uv docs
    if (nzchar(Sys.which("brew"))) {
      if (!quiet) message("Installing uv via Homebrew ...")
      return(invisible(run2("brew", c("install", "uv"))))
    }
    # Fallback to the official install script (user scope)
    if (!quiet) message("Installing uv via official script (user) ...")
    return(invisible(run_cmd("curl -LsSf https://astral.sh/uv/install.sh | sh")))
  } else if (os == "linux") {
    # Prefer user install location; honor available fetcher
    cmd <- if (nzchar(Sys.which("curl"))) {
      "curl -LsSf https://astral.sh/uv/install.sh | sh"
    } else if (nzchar(Sys.which("wget"))) {
      "wget -qO- https://astral.sh/uv/install.sh | sh"
    } else {
      if (!quiet) message("Neither curl nor wget is available. Please install one of them and rerun.")
      return(invisible(FALSE))
    }
    if (!quiet) message("Installing uv via official script (user) ...")
    return(invisible(run_cmd(cmd)))
  } else if (os == "windows") {
    # Avoid invoking winget in non-interactive environments (e.g., R CMD check/CI)
    if (!interactive()) {
      if (!quiet) message("Skipping winget in non-interactive session. Install uv manually: https://docs.astral.sh/uv/")
      return(invisible(FALSE))
    }
    if (nzchar(Sys.which("winget"))) {
      if (!quiet) message("Installing uv via winget ...")
      # Use non-interactive flags to pre-accept agreements and suppress prompts
      return(invisible(run2(
        "winget",
        c(
          "install",
          "--id=astral-sh.uv",
          "-e",
          "--disable-interactivity",
          "--silent",
          "--accept-package-agreements",
          "--accept-source-agreements"
        )
      )))
    } else {
      if (!quiet) message("Please install uv from: https://docs.astral.sh/uv/getting-started/installation/")
      return(invisible(FALSE))
    }
  } else {
    if (!quiet) message("Unsupported OS for automatic uv install. See https://docs.astral.sh/uv/")
    return(invisible(FALSE))
  }
}

# Improved uv availability helper used by package lifecycle
te_ensure_uv_available <- function(prompt = TRUE) {
  if (uv_is_available()) return(invisible(TRUE))

  # If uv exists in ~/.local/bin but not in PATH, add for this session
  local_bin <- path.expand("~/.local/bin")
  local_uv  <- file.path(local_bin, "uv")
  current_path <- Sys.getenv("PATH")
  if (file.exists(local_uv) && !grepl(local_bin, current_path, fixed = TRUE)) {
    Sys.setenv(PATH = paste(local_bin, current_path, sep = ":"))
    if (uv_is_available()) {
      if (interactive()) message("Found uv in ~/.local/bin; added to PATH for this session.")
      return(invisible(TRUE))
    }
  }

  # Non-interactive: optionally auto-install if opted in
  if (!interactive() || !isTRUE(prompt)) {
    auto_install <- {
      v <- tolower(as.character(Sys.getenv("TE_AUTO_INSTALL_UV", unset = "")))
      opt <- isTRUE(getOption("transforEmotion.auto_install_uv", FALSE))
      nzchar(v) && v %in% c("1", "true", "t", "yes", "y") || opt
    }
    if (isTRUE(auto_install)) {
      ok <- isTRUE(install_uv_interactive(quiet = TRUE))
      # Ensure PATH for the session if installed to ~/.local/bin
      current_path <- Sys.getenv("PATH")
      if (file.exists(local_uv) && !grepl(local_bin, current_path, fixed = TRUE)) {
        Sys.setenv(PATH = paste(local_bin, current_path, sep = ":"))
      }
      if (ok && uv_is_available()) {
        return(invisible(TRUE))
      } else {
        return(invisible(FALSE))
      }
    } else {
      return(invisible(FALSE))
    }
  }

  ans <- tryCatch(tolower(readline("uv not found. Install uv now? [Y/n]: ")), error = function(e) "")
  if (ans %in% c("", "y", "yes")) {
    ok <- isTRUE(install_uv_interactive())

    # If installed into ~/.local/bin, ensure PATH for this session
    current_path <- Sys.getenv("PATH")
    if (file.exists(local_uv) && !grepl(local_bin, current_path, fixed = TRUE)) {
      Sys.setenv(PATH = paste(local_bin, current_path, sep = ":"))
    }

    if (ok && uv_is_available()) {
      uv_path <- Sys.which("uv")
      if (nzchar(uv_path) && grepl(local_bin, uv_path, fixed = TRUE)) {
        message("uv installed. Added ~/.local/bin for this session. To persist: export PATH=\"$HOME/.local/bin:$PATH\"")
      } else {
        message("uv installed.")
      }
      return(invisible(TRUE))
    } else {
      message("uv install failed or not on PATH. Tip: export PATH=\"$HOME/.local/bin:$PATH\"")
      return(invisible(FALSE))
    }
  } else {
    message("Continuing without uv. reticulate may use a default venv.")
    return(invisible(FALSE))
  }
}

#' @noRd
ensure_uv_available <- function(prompt = TRUE) {
  if (uv_is_available()) return(invisible(TRUE))
  if (!interactive() || !isTRUE(prompt)) {
    message(paste0(
      "uv not found on PATH. reticulate will fall back to a default venv if needed.
",
      "To use uv (recommended), install it and restart R.
",
      "- Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh
",
      "- macOS (Homebrew): brew install uv
",
      "- Windows (winget): winget install --id=astral-sh.uv -e"))
    return(invisible(FALSE))
  }
  ans <- tryCatch(tolower(readline("uv not found. Install uv now? [Y/n]: ")), error = function(e) "")
  if (ans %in% c("", "y", "yes")) {
    ok <- isTRUE(install_uv_interactive())
    if (ok && uv_is_available()) {
      message("uv installed. Please ensure ~/.local/bin is on PATH and restart R.")
      return(invisible(TRUE))
    } else {
      message(paste0("uv installation did not complete or uv not on PATH.
",
                     "You may need to add ~/.local/bin to PATH and restart R."))
      return(invisible(FALSE))
    }
  } else {
    message("Continuing without uv. reticulate may prompt to create a default venv.")
    return(invisible(FALSE))
  }
}
