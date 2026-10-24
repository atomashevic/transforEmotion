#' @noRd
# Automatically detect appropriate device
# Updated 06.02.2024
auto_device <- function(device, transformer, verbose = TRUE)
{
  ensure_te_py_env()

  # Set transformer memory (MB)
  # Assume GPU and 16-bit unless otherwise noted
  transformer_memory <- switch(
    transformer,
    "tinyllama" = 2640, "llama-2" = 4200, # supported by 4-bit
    "mistral-7b" = 16800, "openchat-3.5" = 16800,
    "orca-2" = 33600, # supported by 32-bit
    "phi-2" = 6480,
    # Gemma3, Ministral, Qwen3 (approximate memory, MB)
    "gemma3-1b" = 815,
    "gemma3-4b" = 3300,
    "ministral-3b" = 7000,
    "ministral-8b" = 18400,
    "qwen3-1.7b" = 1800,
    "qwen2.5-1.5b" = 3100,
    "qwen2.5-3b" = 6200,
    # New: PleIAs and JSON-tuned 1B (approximate)
    "pleias-rag-350m" = 360,
    "pleias-rag-1b" = 900,
    "llama1b-json" = 900
  )

  # First, check for "auto"
  if(device == "auto"){

    # Import {torch}
    torch <- reticulate::import("torch")

    # Check for CUDA
    if(torch$cuda$is_available()){

      # Number of GPU devices
      n_gpu <- torch$cuda$device_count()

      # Branch for number of GPUs
      if(n_gpu == 1){

        # Check for available memory (returns MB)
        gpu_memory <- get_gpu_memory(torch, device_no = 0)

        # Switch to CPU if not enough GPU
        device <- ifelse(gpu_memory > transformer_memory, "cuda:0", "cpu")

      }else{

        # Initialize GPU memory
        gpu_memory <- numeric(length = n_gpu)

        # Loop over and get GPUs
        for(i in 1:n_gpu){
          gpu_memory[i] <- get_gpu_memory(torch, device_no = i - 1)
        }

        # Check how many GPUs are needed
        device <- ifelse(gpu_memory[1] > transformer_memory, "cuda:0", "auto")

      }

    }else{device <- "cpu"}

  }

  # Second, check for "cpu"
  if(device == "cpu"){

    # On the CPU, HuggingFaceLLM loads the weights in 32-bit floats, twice the
    # 16-bit figures above, and loading takes about 15% more than the model
    # keeps. Measured on a CPU (transformers 4.46): Qwen2.5-3B peaked at about
    # 14 GB while loading (13 GB resident afterwards), TinyLLAMA at 6.4 GB
    cpu_needed <- round(transformer_memory * 2.3)

    # Memory available now: models already resident (and a previous LLM that
    # rag() has not freed) are counted, and so is a container or systemd
    # memory limit, which `free` does not show
    cpu_memory <- round(get_cpu_memory())

    # Check for not enough memory
    if(cpu_memory < cpu_needed){

      # Send error
      stop(
        paste0(
          "Cannot load the LLM (about ", cpu_needed, "MB in 32-bit on the CPU). ",
          "Not enough memory (", cpu_memory, "MB available). ",
          "Close other applications or restart R, or use a smaller model."
        ), call. = FALSE
      )

    }

  }

  # Send device to user
  if (isTRUE(verbose)) message(paste0("Using device: ", device))

  # Return device
  return(device)

}

#' @noRd
# GPU memory calculate
# Based on https://www.substratus.ai/blog/calculating-gpu-memory-for-llm
# Updated 28.01.2024
calculate_gpu_memory <- function(parameters, bits = 32)
{
  return(((parameters * 4) / (32 / bits)) * 1.2)
}

#' @noRd
# Get GPU memory
# Free memory on the device (MB), so that models already on it are counted;
# total memory if the free amount cannot be read
get_gpu_memory <- function(torch, device_no)
{

  # Free and total bytes
  free_total <- try(torch$cuda$mem_get_info(as.integer(device_no)), silent = TRUE)
  if (!inherits(free_total, "try-error")) {
    return(as.numeric(free_total[[1]]) / 1048576)
  }

  # Check for total memory
  gpu_memory <- capture.output(torch$cuda$get_device_properties(device_no))

  # Extract memory
  gpu_memory <- gsub(".*total_memory=", "", gpu_memory)
  gpu_memory <- gsub("MB,.*", "", gpu_memory)

  # Return value (MB)
  return(as.numeric(gpu_memory))

}

#' @noRd
# Get CPU memory
# Updated 28.01.2024
get_cpu_memory <- function()
{

  # Get operating system
  OS <- tolower(Sys.info()["sysname"])

  # Branch based on OS
  if(OS == "windows"){ # Windows

    # Alternative (outputs memory in kB)
    bytes <- as.numeric(
      trimws(system("wmic OS get FreePhysicalMemory", intern = TRUE))[2]
    ) * 1e+03

  }else if(OS == "linux"){ # Linux

    # The smaller of the system's available memory and what the process's
    # cgroup still allows (a container, Colab, or a systemd memory limit)
    # (Inf when neither can be read: the check is then skipped)
    bytes <- suppressWarnings(min(linux_available_bytes(), cgroup_available_bytes(), na.rm = TRUE))

  }else{ # Mac

    # System information
    system_info <- system("top -l 1 -s 0 | grep PhysMem", intern = TRUE)

    # Get everything after comma
    unused <- gsub(" .*,", "", system_info)

    # Get values only
    value <- gsub(" unused.", "", gsub("PhysMem: ", "", unused))

    # Check for bytes
    if(grepl("M", value)){
      bytes <- as.numeric(gsub("M", "", value)) * 1e+06
    }else if(grepl("G", value)){
      bytes <- as.numeric(gsub("G", "", value)) * 1e+09
    }else if(grepl("K", value)){
      bytes <- as.numeric(gsub("K", "", value)) * 1e+03
    }else if(grepl("B", value)){ # edge case
      bytes <- as.numeric(gsub("B", "", value)) * 1
    }else if(grepl("T", value)){ # edge case
      bytes <- as.numeric(gsub("T", "", value)) * 1e+12
    }

  }

  # Return value (MB)
  return(bytes / 1e+06)

}

#' @noRd
# MemAvailable from /proc/meminfo (bytes): the kernel's estimate of memory
# that can be used without swapping, which already excludes resident models
linux_available_bytes <- function(meminfo = "/proc/meminfo")
{
  if (!file.exists(meminfo)) return(NA_real_)
  lines <- try(readLines(meminfo), silent = TRUE)
  if (inherits(lines, "try-error")) return(NA_real_)
  line <- grep("^MemAvailable:", lines, value = TRUE)
  if (length(line) != 1) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", line)) * 1024 # reported in KiB
}

#' @noRd
# Memory the process's cgroup still allows (bytes), or NA without a limit.
# A process killed for exceeding its cgroup limit was refused by the cgroup,
# not by the machine, so `free` and MemAvailable can show plenty of memory.
# Page cache the kernel can reclaim (inactive_file) does not count as used.
# Every level up to the root is checked: a parent's limit also applies.
cgroup_available_bytes <- function(root = "/sys/fs/cgroup", self_cgroup = "/proc/self/cgroup")
{
  read_number <- function(file) {
    if (!file.exists(file)) return(NA_real_)
    value <- try(readLines(file, n = 1, warn = FALSE), silent = TRUE)
    if (inherits(value, "try-error") || length(value) == 0) return(NA_real_)
    suppressWarnings(as.numeric(value))
  }
  read_stat <- function(file, key) {
    if (!file.exists(file)) return(0)
    lines <- try(readLines(file, warn = FALSE), silent = TRUE)
    if (inherits(lines, "try-error")) return(0)
    line <- grep(paste0("^", key, " "), lines, value = TRUE)
    if (length(line) != 1) return(0)
    as.numeric(sub(".* ", "", line))
  }

  # Walk from the process's own cgroup folder up to the top of the hierarchy,
  # keeping the smallest limit minus usage. A limit of "max" (v2) reads as NA;
  # "no limit" in v1 reads as a number near 2^63
  walk <- function(dir, top, limit_file, usage_file, inactive_key) {
    available <- NA_real_
    top <- normalizePath(top, mustWork = FALSE)
    repeat {
      limit <- read_number(file.path(dir, limit_file))
      if (is.finite(limit) && limit < 2^60) {
        used <- read_number(file.path(dir, usage_file)) -
          read_stat(file.path(dir, "memory.stat"), inactive_key)
        if (is.finite(used)) available <- min(available, limit - used, na.rm = TRUE)
      }
      if (normalizePath(dir, mustWork = FALSE) == top) break
      parent <- dirname(dir)
      if (identical(parent, dir)) break
      dir <- parent
    }
    if (is.finite(available)) max(available, 0) else NA_real_
  }
  # Inside a container the process's path names the host's folder, which is
  # not mounted there: the container's own cgroup is the top folder
  own_dir <- function(top, path) {
    dir <- file.path(top, sub("^/", "", path))
    if (dir.exists(dir)) dir else top
  }

  membership <- if (file.exists(self_cgroup)) try(readLines(self_cgroup, warn = FALSE), silent = TRUE) else character(0)
  if (inherits(membership, "try-error")) membership <- character(0)

  # cgroup v2: one "0::/path" line
  v2 <- grep("^0::", membership, value = TRUE)
  if (length(v2) == 1 && file.exists(file.path(root, "cgroup.controllers"))) {
    return(walk(own_dir(root, sub("^0::", "", v2)), root,
                "memory.max", "memory.current", "inactive_file"))
  }

  # cgroup v1: the memory controller has its own hierarchy, "N:memory:/path"
  v1 <- grep("^[0-9]+:([^:]*,)?memory(,[^:]*)?:", membership, value = TRUE)
  top <- file.path(root, "memory")
  if (!dir.exists(top)) return(NA_real_)
  path <- if (length(v1) == 1) sub("^[^:]*:[^:]*:", "", v1) else "/"
  walk(own_dir(top, path), top,
       "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file")
}
