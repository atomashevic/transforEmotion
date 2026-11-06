# Device choice for PyTorch: does the running PyTorch build have kernels for
# the NVIDIA GPU?
#
# torch.cuda.is_available() is TRUE when a CUDA build of PyTorch finds a GPU
# and a driver. It does not check that the build has kernels for that GPU.
# The CUDA 12.6 build of PyTorch 2.14 has kernels for sm_50 to sm_90 and no
# PTX, so on a Blackwell GPU (compute capability 10.0 or 12.0: B200, GeForce
# RTX 50 series) is_available() is TRUE and the first CUDA operation fails
# with "CUDA error: no kernel image is available for execution on the
# device". The cu126 build is still installed in a fixed environment
# (TRANSFOREMOTION_PYTHON), in a GPU image built from docker/Dockerfile, and
# by setup_cache(gpu = TRUE) on a machine where nvidia-smi does not report
# the GPU. Every place that resolves device = "auto" uses .te_torch_device().

.te_cuda_state <- new.env(parent = emptyenv())

#' @noRd
# Whether a GPU of compute capability major.minor can run code compiled for
# the targets in `arch`, as torch._C._cuda_getArchFlags() lists them (for
# example "sm_86", "sm_90a", "compute_90"). The last digit of a target is the
# minor version. A cubin for sm_XY runs on X.Y and on later minor versions
# of major X. PTX for compute_XY is compiled by the driver for any GPU of
# capability X.Y or higher. Targets with the suffix "a" (sm_90a) run only on
# that exact capability; "f" (sm_100f) runs on the same major version.
.te_cuda_arch_supports <- function(arch, major, minor) {
  m <- regmatches(arch, regexec("^(sm|compute)_([0-9]+)([af]?)$", arch))
  m <- m[lengths(m) == 4]
  if (!length(m)) return(FALSE)
  kind <- vapply(m, `[`, "", 2)
  digits <- vapply(m, `[`, "", 3)
  suffix <- vapply(m, `[`, "", 4)
  t_major <- as.integer(substr(digits, 1, nchar(digits) - 1))
  t_minor <- as.integer(substring(digits, nchar(digits)))
  exact <- t_major == major & t_minor == minor
  same_major <- t_major == major & t_minor <= minor
  lower <- t_major * 10 + t_minor <= major * 10 + minor
  ok <- ifelse(suffix == "a", exact,
               ifelse(kind == "sm" | suffix == "f", same_major, lower))
  any(ok)
}

#' @noRd
# What PyTorch reports about CUDA in the running Python: whether CUDA is
# available, the PyTorch version, the targets of the build and the name and
# compute capability of each visible GPU. torch.cuda.get_arch_list() returns
# an empty list when CUDA is not available, so the targets are read from
# torch._C._cuda_getArchFlags(), which a CUDA build has also without a GPU.
.te_cuda_probe <- function(torch = reticulate::import("torch")) {
  probe <- list(available = isTRUE(torch$cuda$is_available()),
                version = as.character(torch$`__version__`))
  if (!probe$available) return(probe)
  flags <- tryCatch(
    reticulate::py_get_attr(reticulate::py_get_attr(torch, "_C"), "_cuda_getArchFlags")(),
    error = function(e) NULL
  )
  probe$arch <- if (is.character(flags)) strsplit(trimws(flags), "\\s+")[[1]] else character()
  n <- as.integer(torch$cuda$device_count())
  devices <- lapply(seq_len(n) - 1L, function(i) {
    capability <- unlist(torch$cuda$get_device_capability(i))
    data.frame(index = i, name = as.character(torch$cuda$get_device_name(i)),
               major = as.integer(capability[1]), minor = as.integer(capability[2]),
               stringsAsFactors = FALSE)
  })
  probe$devices <- do.call(rbind, devices)
  probe
}

#' @noRd
# .te_cuda_probe() with a column `supported` for each GPU, once per session:
# the PyTorch build and the visible GPUs do not change while Python runs. A
# build that does not list its targets, or a probe that fails, counts as
# supported, so that a GPU that worked before is still used.
.te_cuda_status <- function() {
  if (!is.null(.te_cuda_state$status)) return(.te_cuda_state$status)
  status <- tryCatch(.te_cuda_probe(), error = function(e) NULL)
  if (is.null(status)) {
    torch <- reticulate::import("torch")
    status <- list(available = isTRUE(torch$cuda$is_available()))
  }
  if (isTRUE(status$available) && NROW(status$devices)) {
    status$devices$supported <- if (length(status$arch)) {
      mapply(.te_cuda_arch_supports, MoreArgs = list(arch = status$arch),
             major = status$devices$major, minor = status$devices$minor)
    } else {
      TRUE
    }
  }
  .te_cuda_state$status <- status
  status
}

#' @noRd
# Whether GPU `index` (0 is the device that "cuda" means) can run the
# PyTorch build. NA when CUDA is not available.
.te_cuda_supported <- function(status = .te_cuda_status(), index = 0L) {
  if (!isTRUE(status$available)) return(NA)
  devices <- status$devices
  if (!NROW(devices) || !index %in% devices$index) return(TRUE)
  isTRUE(devices$supported[devices$index == index])
}

#' @noRd
# Explanation for a GPU that the PyTorch build cannot run: the GPU, its
# compute capability, the build, and how to install a build for it, which
# depends on where the Python environment comes from. The cu130 build runs
# on compute capability 7.5 to 12.0 and needs NVIDIA driver R580 or later;
# the cu126 build runs on 5.0 to 9.0 and needs R525 or later.
.te_cuda_unsupported_text <- function(status, index = 0L) {
  gpu <- status$devices[status$devices$index == index, ]
  version <- if (length(status$version)) status$version else "unknown"
  new_gpu <- gpu$major * 10 + gpu$minor >= 75
  build <- if (new_gpu) "cu130" else "cu126"
  python <- Sys.getenv("TRANSFOREMOTION_PYTHON", unset = "")
  te_home <- Sys.getenv("TE_HOME", unset = "")
  fix <- if (nzchar(te_home) && file.exists(file.path(te_home, "BUILD_INFO"))) {
    paste0(
      "For a Docker image, build docker/Dockerfile with ",
      "`--build-arg GPU=1 --build-arg CUDA=", build, "`."
    )
  } else if (nzchar(python)) {
    paste0(
      "Install it into the fixed environment (TRANSFOREMOTION_PYTHON): ",
      shQuote(python), " -m pip install torch==", sub("\\+.*$", "", version),
      " torchvision --index-url https://download.pytorch.org/whl/", build
    )
  } else if (new_gpu) {
    paste0(
      "transforEmotion installs it when nvidia-smi reports the compute ",
      "capability of the GPU. Check that nvidia-smi runs, and restart R. An ",
      "environment prepared with setup_cache(gpu = TRUE) on a machine ",
      "without nvidia-smi (for example an HPC login node) has the CUDA 12.6 ",
      "build: prepare it again on a machine with this GPU."
    )
  } else {
    paste0(
      "transforEmotion installs it when no GPU of the machine has compute ",
      "capability 10.0 or higher. Otherwise, set CUDA_VISIBLE_DEVICES to the ",
      "GPUs that this build supports, and restart R."
    )
  }
  paste0(
    "PyTorch ", version, " has no kernels for GPU ", index, ", ", gpu$name,
    " (compute capability ", gpu$major, ".", gpu$minor, "). The build has ",
    "kernels for ", if (length(status$arch)) paste(status$arch, collapse = " ") else "other GPUs",
    ". The ", if (new_gpu) "CUDA 13.0 build of PyTorch (cu130)" else "CUDA 12.6 build of PyTorch (cu126)",
    " runs on this GPU and needs NVIDIA driver ", if (new_gpu) "R580" else "R525",
    " or later. ", fix
  )
}

#' @noRd
# The device that PyTorch can use: "auto" gives "cuda" when CUDA is
# available and the PyTorch build has kernels for GPU 0, otherwise "cpu",
# with a message once per session when the GPU cannot be used. An explicit
# "cuda" or "cuda:N" for a GPU without kernels stops with the explanation,
# instead of the CUDA error at the first operation. Other values are returned
# unchanged, and "cpu" does not import torch.
.te_torch_device <- function(device = "auto") {
  device <- tolower(device)
  if (identical(device, "cpu")) return("cpu")
  explicit <- grepl("^cuda(:[0-9]+)?$", device)
  if (!identical(device, "auto") && !explicit) return(device)

  status <- .te_cuda_status()
  if (identical(device, "auto")) {
    if (!isTRUE(status$available)) return("cpu")
    if (.te_cuda_supported(status)) return("cuda")
    if (!isTRUE(.te_cuda_state$told)) {
      .te_cuda_state$told <- TRUE
      message(.te_cuda_unsupported_text(status), " Until then, transforEmotion uses the CPU.")
    }
    return("cpu")
  }

  index <- if (grepl(":", device)) as.integer(sub("^cuda:", "", device)) else 0L
  if (isFALSE(.te_cuda_supported(status, index))) {
    stop(.te_cuda_unsupported_text(status, index),
         " Until then, use device = \"cpu\".", call. = FALSE)
  }
  device
}

#' @noRd
# Device choice for the Python modules (inst/python/image.py, te_audio.py),
# which create their models themselves: the result of
# .te_torch_device("auto"), passed in the environment variable
# TRANSFOREMOTION_TORCH_DEVICE of the running Python.
.te_python_device <- function() {
  device <- .te_torch_device("auto")
  os <- reticulate::import("os", convert = FALSE)
  os$environ$`__setitem__`("TRANSFOREMOTION_TORCH_DEVICE", device)
  invisible(device)
}
