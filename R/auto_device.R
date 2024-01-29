#' @noRd
# Automatically detect appropriate device
# Updated 28.01.2024
auto_device <- function(device, transformer)
{

  # Set transformer memory (MB)
  # Numbers derived from overall memory usage on Alex's 1x A6000
  # Single run of `rag` with each model using 3800 tweets
  transformer_memory <- round(
    switch(
      transformer,
      "tinyllama" = 5324, "llama-2" = 5798,
      "mistral-7b" = 30018, "orca-2" = 29836,
      "phi-2" = 13594
    ), digits = -2
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

    # Get CPU memory
    cpu_memory <- get_cpu_memory()

    # Check for not enough memory
    if(cpu_memory < transformer_memory){

      # Send error
      stop(
        paste0(
          "Cannot load the LLM (", transformer_memory,  "MB). ",
          "Not enough memory (", cpu_memory, "MB)"
        ), call. = FALSE
      )

    }

  }

  # Send device to user
  message(paste0("Using device: ", device))

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
# Updated 28.01.2024
get_gpu_memory <- function(torch, device_no)
{

  # Check for available memory
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

    # Split system information
    info_split <- strsplit(system("free", intern = TRUE), split = " ")

    # Remove "Mem:" and "Swap:"
    info_split <- lapply(info_split, function(x){gsub("Mem:", "", x)})
    info_split <- lapply(info_split, function(x){gsub("Swap:", "", x)})

    # Get actual values
    info_split <- lapply(info_split, function(x){x[x != ""]})

    # Bind values
    info_split <- do.call(rbind, info_split[1:2])

    # Get free values (Linux reports in *kilo*bytes -- thanks, Aleksandar Tomasevic)
    bytes <- as.numeric(info_split[2, info_split[1,] == "available"]) * 1e+03

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
