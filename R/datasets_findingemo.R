#' Download FindingEmo-Light Dataset
#'
#' @description
#' Downloads the FindingEmo-Light dataset using the official PyPI package.
#' This dataset contains 25k images with emotion annotations including valence,
#' arousal, and discrete emotion labels, focusing on complex naturalistic scenes
#' with multiple people in social settings.
#'
#' @param target_dir Character. Directory to download the dataset to.
#' @param max_images Integer. Maximum number of images to download (optional).
#' @param randomize Logical. If TRUE and max_images is specified, randomly 
#'   select images for download. Useful for creating test/benchmark subsets 
#'   (default: FALSE).
#' @param skip_existing Logical. Whether to skip download if dataset already 
#'   exists (default: TRUE).
#' @param force Logical. Force download even if dataset exists (default: FALSE).
#'
#' @return
#' A list containing:
#' \itemize{
#'   \item \code{success}: Logical indicating if download was successful
#'   \item \code{message}: Character string with status message
#'   \item \code{target_dir}: Path to downloaded data
#'   \item \code{annotation_file}: Path to annotation file (if successful)
#'   \item \code{urls_file}: Path to URLs file (if successful)
#'   \item \code{image_count}: Number of images downloaded (if any)
#'   \item \code{annotations}: Full annotations data.frame (raw)
#'   \item \code{evaluation_data}: Data.frame filtered to downloaded images
#'         with columns suitable for evaluation workflows (id, truth, image_file,
#'         image_path, valence, arousal, emo8_label, emotion)
#'   \item \code{evaluation_csv}: Path to saved CSV of evaluation_data
#'   \item \code{matched_count}: Number of annotations matched to downloaded images
#' }
#'
#' @details
#' This function requires the \code{findingemo-light} Python package to be 
#' installed. Use \code{setup_modules()} to install required dependencies
#' before calling this function.
#' 
#' The FindingEmo dataset is described in:
#' Mertens, L. et al. (2024). "FindingEmo: An Image Dataset for Emotion 
#' Recognition in the Wild". NeurIPS 2024 Datasets and Benchmarks Track.
#' 
#' The dataset uses a flat directory structure with all images stored directly 
#' in the images/ subdirectory, annotations.csv and urls.json at the root level.
#' 
#' **Note**: For copyright reasons, the dataset provides URLs and annotations
#' only. Images are downloaded on-demand from their original sources.
#'
#' @examples
#' \dontrun{
#' # First install required modules
#' setup_modules()
#' 
#' # Download dataset to local directory
#' result <- download_findingemo_data("./findingemo_data")
#' 
#' if (result$success) {
#'   cat("Dataset downloaded to:", result$target_dir)
#'   cat("Images downloaded:", result$image_count)
#' }
#' 
#' # Download random subset for testing/benchmarking
#' result <- download_findingemo_data(
#'   target_dir = "./findingemo_test",
#'   max_images = 100,
#'   randomize = TRUE
#' )
#' 
#' # Download subset with flat directory structure (always used)
#' result <- download_findingemo_data(
#'   target_dir = "./findingemo_subset",
#'   max_images = 50
#' )
#' 
#' # Force re-download
#' result <- download_findingemo_data(
#'   target_dir = "./findingemo_data",
#'   force = TRUE
#' )
#' }
#'
#' @seealso
#' \code{\link{load_findingemo_annotations}}, \code{\link{setup_modules}}
#'
#' @export
download_findingemo_data <- function(target_dir,
                                   max_images = NULL,
                                   randomize = FALSE,
                                   skip_existing = TRUE,
                                   force = FALSE) {
  
  # Suppress TensorFlow messages
  Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
  
  # Input validation
  if (!is.character(target_dir) || length(target_dir) != 1) {
    stop("target_dir must be a single character string", call. = FALSE)
  }
  
  if (!is.null(max_images) && (!is.numeric(max_images) || max_images <= 0)) {
    stop("max_images must be a positive integer", call. = FALSE)
  }
  
  if (!is.logical(randomize) || length(randomize) != 1) {
    stop("randomize must be a single logical value", call. = FALSE)
  }
  
  if (force) skip_existing <- FALSE
  
  # Expand path
  target_dir <- path.expand(target_dir)
  
  # Try to import required Python module
  module_import <- try({
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    download_module <- reticulate::source_python(system.file("python", "download_findingemo.py", package = "transforEmotion"))
    download_module
  }, silent = TRUE)
  
  # If import fails, try setting up modules
  if(inherits(module_import, "try-error")) {
    message("Required Python modules not found. Setting up modules...")
    setup_modules()
    reticulate::use_condaenv("transforEmotion", required = FALSE)
    download_module <- reticulate::source_python(system.file("python", "download_findingemo.py", package = "transforEmotion"))
  }

  # Execute Python function
  message("Downloading FindingEmo dataset...")
  message("This may take several minutes depending on your connection.")
  
  result <- reticulate::py$download_findingemo_data(
    target_dir = target_dir,
    max_images = max_images,
    randomize = randomize,
    skip_existing = skip_existing,
    force = force
  )
  
  # Add target directory to result
  result$target_dir <- target_dir
  
  # Print status message
  if (result$success) {
    if (!is.null(result$partial) && result$partial) {
      message("⚠ FindingEmo dataset download partially completed")
      message("  ", result$message)
    } else {
      message("✓ FindingEmo dataset download completed successfully")
    }
    
    if (!is.null(result$image_count)) {
      if (result$image_count == 0) {
        message("  Images downloaded: 0 (metadata only)")
        message("  Note: Some image URLs may be broken or inaccessible")
      } else {
        message("  Images downloaded: ", result$image_count)
        if (!is.null(max_images) && result$image_count < max_images) {
          message("  Note: Some images could not be downloaded (broken URLs)")
        }
      }
    }
    
    # Load annotations and merge with actually downloaded images
    annotations_df <- tryCatch({
      load_findingemo_annotations(target_dir, output_format = "dataframe")
    }, error = function(e) {
      warning("Failed to load annotations after download: ", e$message)
      NULL
    })
    
    images_dir <- file.path(target_dir, "images")
    image_files <- if (dir.exists(images_dir)) list.files(
      images_dir,
      pattern = "\\.(jpg|jpeg|png|bmp|gif)$",
      ignore.case = TRUE,
      full.names = FALSE
    ) else character(0)
    
    evaluation_df <- NULL
    evaluation_csv <- NULL
    matched_count <- 0
    
    if (!is.null(annotations_df)) {
      # Ensure expected columns exist
      if (!"image_path" %in% names(annotations_df)) {
        # Try common alternatives
        alt <- intersect(c("filepath", "path", "image"), names(annotations_df))
        if (length(alt) > 0) {
          annotations_df$image_path <- annotations_df[[alt[1]]]
        }
      }
      if (!"index" %in% names(annotations_df)) {
        # Fallback to row index if no explicit id
        annotations_df$index <- seq_len(nrow(annotations_df))
      }
      if (!"emotion" %in% names(annotations_df)) {
        # Try to find an emotion/label column
        emo_col <- grep("emotion|label", names(annotations_df), value = TRUE, ignore.case = TRUE)
        if (length(emo_col) > 0) annotations_df$emotion <- annotations_df[[emo_col[1]]]
      }
      
  # Derive file name (guard against missing image_path)
  ipath <- annotations_df$image_path
  ipath[is.null(ipath)] <- ""
  ipath[is.na(ipath)] <- ""
  annotations_df$image_file <- basename(ipath)
      annotations_df$image_file[is.na(annotations_df$image_file)] <- ""
      
      # Map to Emo8 where possible
      if ("emotion" %in% names(annotations_df)) {
        suppressWarnings({
          annotations_df$emo8_label <- tryCatch(map_to_emo8(as.character(annotations_df$emotion)), error = function(e) rep(NA_character_, nrow(annotations_df)))
        })
      } else {
        annotations_df$emo8_label <- NA_character_
      }
      
      # Keep only those with downloaded image files
      matched <- annotations_df[annotations_df$image_file %in% image_files & annotations_df$image_file != "", , drop = FALSE]
      matched_count <- nrow(matched)
      
      if (matched_count > 0) {
        # Build evaluation-ready dataframe
        # Prefer emo8_label as truth if available, else original emotion
        truth_col <- if (any(!is.na(matched$emo8_label))) "emo8_label" else "emotion"
        evaluation_df <- data.frame(
          id = matched$index,
          truth = matched[[truth_col]],
          image_file = matched$image_file,
          image_path = file.path(images_dir, matched$image_file),
          stringsAsFactors = FALSE
        )
        # Attach valence/arousal if present
        if ("valence" %in% names(matched)) evaluation_df$valence <- matched$valence
        if ("arousal" %in% names(matched)) evaluation_df$arousal <- matched$arousal
        # Keep reference of both label variants
        if ("emotion" %in% names(matched)) evaluation_df$emotion <- matched$emotion
        if ("emo8_label" %in% names(matched)) evaluation_df$emo8_label <- matched$emo8_label
        
        # Save CSV for convenience
        evaluation_csv <- file.path(target_dir, "evaluation_annotations.csv")
        tryCatch({
          utils::write.csv(evaluation_df, evaluation_csv, row.names = FALSE)
          message("✓ Prepared evaluation data for ", matched_count, " images")
          message("  Saved: ", evaluation_csv)
        }, error = function(e) {
          warning("Failed to save evaluation CSV: ", e$message)
        })
      } else {
        message("No annotations matched to downloaded image files; evaluation data not created")
      }
    }
    
    # Attach to result
    result$annotations <- annotations_df
    result$evaluation_data <- evaluation_df
    result$evaluation_csv <- evaluation_csv
    result$matched_count <- matched_count
  } else {
    warning("✗ FindingEmo dataset download failed: ", result$message)
  }
  
  return(result)
}

#' Load FindingEmo-Light Annotations
#'
#' @description
#' Loads and preprocesses annotations from a downloaded FindingEmo-Light dataset.
#' Returns a clean R data.frame with emotion annotations, valence/arousal scores,
#' and associated metadata.
#'
#' @param data_dir Character. Directory containing the downloaded FindingEmo data.
#' @param output_format Character. Format for processed data: "dataframe" 
#'   returns R data.frame, "list" returns full processed data (default: "dataframe").
#' @param python_path Character. Path to Python executable (optional).
#'
#' @return
#' If \code{output_format = "dataframe"}: A data.frame with columns:
#' \itemize{
#'   \item \code{image_id}: Unique image identifier
#'   \item \code{valence}: Valence score (emotion positivity)
#'   \item \code{arousal}: Arousal score (emotion intensity)
#'   \item Additional columns as present in the dataset
#' }
#' 
#' If \code{output_format = "list"}: A list containing:
#' \itemize{
#'   \item \code{annotations}: Data.frame with annotation data
#'   \item \code{urls}: List with image URL information
#'   \item \code{metadata}: List with dataset metadata
#' }
#'
#' @details
#' This function loads the CSV annotation file and JSON URLs file from a 
#' downloaded FindingEmo dataset, performs basic validation and preprocessing,
#' and returns the data in a format suitable for emotion analysis.
#' 
#' The function handles missing values, validates valence/arousal ranges,
#' and provides summary statistics for the loaded data.
#'
#' @examples
#' \dontrun{
#' # Download dataset first
#' download_result <- download_findingemo_data("./findingemo_data")
#' 
#' if (download_result$success) {
#'   # Load annotations as data.frame
#'   annotations <- load_findingemo_annotations("./findingemo_data")
#'   
#'   # Examine the data
#'   head(annotations)
#'   summary(annotations)
#'   
#'   # Get full processed data including metadata
#'   full_data <- load_findingemo_annotations(
#'     data_dir = "./findingemo_data",
#'     output_format = "list"
#'   )
#'   
#'   print(full_data$metadata)
#' }
#' }
#'
#' @seealso
#' \code{\link{download_findingemo_data}}, \code{\link{prepare_findingemo_evaluation}}
#'
#' @export
load_findingemo_annotations <- function(data_dir,
                                      output_format = c("dataframe", "list"),
                                      python_path = NULL) {
  
  output_format <- match.arg(output_format)
  
  # Input validation
  if (!is.character(data_dir) || length(data_dir) != 1) {
    stop("data_dir must be a single character string", call. = FALSE)
  }
  
  # Expand path
  data_dir <- path.expand(data_dir)
  
  if (!dir.exists(data_dir)) {
    stop("Data directory does not exist: ", data_dir, call. = FALSE)
  }
  
  # Check if this is a flat structure dataset
  flat_annotations <- file.path(data_dir, "annotations.csv")
  flat_urls <- file.path(data_dir, "urls.json")
  
  if (file.exists(flat_annotations) && file.exists(flat_urls)) {
    message("Detected flat structure dataset")
    return(.load_flat_structure_data(data_dir, output_format))
  }
  
  # Get Python script path
  script_path <- system.file("python", "load_findingemo_annotations.py", 
                            package = "transforEmotion")
  
  if (!file.exists(script_path)) {
    stop("Python script not found. Please reinstall transforEmotion.", 
         call. = FALSE)
  }
  
  # Create temporary file for JSON output
  json_output <- tempfile(fileext = ".json")
  
  # Build command arguments
  args <- c(
    "--data_dir", shQuote(data_dir),
    "--output_format", "json",
    "--output_json", shQuote(json_output)
  )
  
  # Execute Python script
  message("Loading FindingEmo annotations...")
  
  if (is.null(python_path)) {
    # Use reticulate's Python
    tryCatch({
      reticulate::py_run_file(script_path, local = list(
        sys = list(argv = c("load_findingemo_annotations.py", args))
      ))
      exit_code <- 0
    }, error = function(e) {
      warning("Failed to run via reticulate, trying system call: ", e$message)
      # Fallback to system call
      python_cmd <- reticulate::py_config()$python
      exit_code <- system2(python_cmd, args = c(shQuote(script_path), args))
    })
  } else {
    # Use specified Python path
    exit_code <- system2(python_path, args = c(shQuote(script_path), args))
  }
  
  # Read results from JSON output
  if (!file.exists(json_output)) {
    stop("Failed to load annotations: No output file generated", call. = FALSE)
  }
  
  result <- tryCatch({
    jsonlite::fromJSON(json_output, simplifyVector = FALSE)
  }, error = function(e) {
    stop("Failed to read annotation results: ", e$message, call. = FALSE)
  })
  
  unlink(json_output)
  
  if (!result$success) {
    stop("Failed to load annotations: ", result$message, call. = FALSE)
  }
  
  # Convert to R data structures
  if (output_format == "dataframe") {
    # Return just the annotations as a data.frame
    annotations_df <- do.call(rbind, lapply(result$data$annotations, function(x) {
      data.frame(x, stringsAsFactors = FALSE)
    }))
    
    # Print summary information
    message("✓ Loaded ", nrow(annotations_df), " annotations")
    if ("valence" %in% names(annotations_df)) {
      message("  Valence range: ", 
              round(min(annotations_df$valence, na.rm = TRUE), 3), " to ",
              round(max(annotations_df$valence, na.rm = TRUE), 3))
    }
    if ("arousal" %in% names(annotations_df)) {
      message("  Arousal range: ", 
              round(min(annotations_df$arousal, na.rm = TRUE), 3), " to ",
              round(max(annotations_df$arousal, na.rm = TRUE), 3))
    }
    
    return(annotations_df)
    
  } else {
    # Return full data structure
    processed_data <- list(
      annotations = do.call(rbind, lapply(result$data$annotations, function(x) {
        data.frame(x, stringsAsFactors = FALSE)
      })),
      urls = result$data$urls,
      metadata = result$data$metadata
    )
    
    message("✓ Loaded full dataset with ", 
            processed_data$metadata$n_annotations, " annotations and ",
            processed_data$metadata$n_urls, " URL entries")
    
    return(processed_data)
  }
}

#' Prepare FindingEmo Data for Evaluation
#'
#' @description
#' Prepares FindingEmo dataset annotations for use with \code{evaluate_emotions()}.
#' Converts the dataset format to match the expected input structure for 
#' evaluation functions.
#'
#' @param annotations Data.frame. Annotations from \code{load_findingemo_annotations()}.
#' @param predictions Data.frame. Model predictions with same image IDs as annotations.
#' @param id_col Character. Column name for image IDs (default: "image_id").
#' @param truth_col Character. Column name for ground truth emotions (default: "emotion_label").
#' @param pred_col Character. Column name for predicted emotions (default: "predicted_emotion").
#' @param include_va Logical. Whether to include valence/arousal columns (default: TRUE).
#'
#' @return
#' A data.frame formatted for use with \code{evaluate_emotions()}, containing:
#' \itemize{
#'   \item \code{id}: Image identifiers
#'   \item \code{truth}: Ground truth emotion labels
#'   \item \code{pred}: Predicted emotion labels
#'   \item \code{valence}: Valence scores (if available and include_va = TRUE)
#'   \item \code{arousal}: Arousal scores (if available and include_va = TRUE)
#'   \item Additional probability columns if present in predictions
#' }
#'
#' @details
#' This function merges FindingEmo annotations with model predictions and formats
#' the result for evaluation. It handles missing values, validates data consistency,
#' and ensures the output matches the expected format for \code{evaluate_emotions()}.
#'
#' @examples
#' \dontrun{
#' # Load annotations
#' annotations <- load_findingemo_annotations("./findingemo_data")
#' 
#' # Create mock predictions (replace with actual model predictions)
#' predictions <- data.frame(
#'   image_id = annotations$image_id[1:100],
#'   predicted_emotion = sample(c("happy", "sad", "angry"), 100, replace = TRUE),
#'   prob_happy = runif(100),
#'   prob_sad = runif(100),
#'   prob_angry = runif(100)
#' )
#' 
#' # Prepare for evaluation
#' eval_data <- prepare_findingemo_evaluation(
#'   annotations = annotations,
#'   predictions = predictions
#' )
#' 
#' # Evaluate model performance
#' results <- evaluate_emotions(
#'   data = eval_data,
#'   probs_cols = c("prob_happy", "prob_sad", "prob_angry")
#' )
#' 
#' print(results)
#' }
#'
#' @seealso
#' \code{\link{load_findingemo_annotations}}, \code{\link{evaluate_emotions}}
#'
#' @export
prepare_findingemo_evaluation <- function(annotations,
                                        predictions,
                                        id_col = "image_id",
                                        truth_col = "emotion_label",
                                        pred_col = "predicted_emotion",
                                        include_va = TRUE) {
  
  # Input validation
  if (!is.data.frame(annotations)) {
    stop("annotations must be a data.frame", call. = FALSE)
  }
  
  if (!is.data.frame(predictions)) {
    stop("predictions must be a data.frame", call. = FALSE)
  }
  
  if (!id_col %in% names(annotations)) {
    stop("ID column '", id_col, "' not found in annotations", call. = FALSE)
  }
  
  if (!id_col %in% names(predictions)) {
    stop("ID column '", id_col, "' not found in predictions", call. = FALSE)
  }
  
  if (!pred_col %in% names(predictions)) {
    stop("Prediction column '", pred_col, "' not found in predictions", call. = FALSE)
  }
  
  # Check if truth column exists in annotations
  if (!truth_col %in% names(annotations)) {
    warning("Truth column '", truth_col, "' not found in annotations. ",
            "Available columns: ", paste(names(annotations), collapse = ", "))
    # Try to find emotion-related column
    emotion_cols <- grep("emotion|label", names(annotations), value = TRUE, ignore.case = TRUE)
    if (length(emotion_cols) > 0) {
      truth_col <- emotion_cols[1]
      message("Using '", truth_col, "' as truth column")
    } else {
      stop("No suitable truth column found in annotations", call. = FALSE)
    }
  }
  
  # Merge annotations with predictions
  message("Merging annotations with predictions...")
  
  eval_data <- merge(
    annotations[, c(id_col, truth_col, 
                   if (include_va && "valence" %in% names(annotations)) "valence",
                   if (include_va && "arousal" %in% names(annotations)) "arousal"), 
                drop = FALSE],
    predictions,
    by = id_col,
    all = FALSE  # Inner join - only keep matching records
  )
  
  if (nrow(eval_data) == 0) {
    stop("No matching records found between annotations and predictions", call. = FALSE)
  }
  
  # Rename columns to standard format
  names(eval_data)[names(eval_data) == id_col] <- "id"
  names(eval_data)[names(eval_data) == truth_col] <- "truth"
  names(eval_data)[names(eval_data) == pred_col] <- "pred"
  
  # Remove rows with missing critical values
  initial_count <- nrow(eval_data)
  eval_data <- eval_data[!is.na(eval_data$truth) & !is.na(eval_data$pred), ]
  final_count <- nrow(eval_data)
  
  if (initial_count > final_count) {
    message("Removed ", initial_count - final_count, " rows with missing truth/prediction values")
  }
  
  if (nrow(eval_data) == 0) {
    stop("No valid evaluation data after removing missing values", call. = FALSE)
  }
  
  # Report summary
  message("✓ Prepared evaluation data with ", nrow(eval_data), " samples")
  
  truth_labels <- unique(eval_data$truth)
  pred_labels <- unique(eval_data$pred)
  message("  Ground truth labels: ", paste(sort(truth_labels), collapse = ", "))
  message("  Predicted labels: ", paste(sort(pred_labels), collapse = ", "))
  
  # Check for probability columns
  prob_cols <- grep("^prob_", names(eval_data), value = TRUE)
  if (length(prob_cols) > 0) {
    message("  Found probability columns: ", paste(prob_cols, collapse = ", "))
  }
  
  return(eval_data)
}

#' Load annotations from flat structure dataset
#' @noRd
.load_flat_structure_data <- function(data_dir, output_format) {
  
  # File paths
  annotations_file <- file.path(data_dir, "annotations.csv")
  urls_file <- file.path(data_dir, "urls.json")
  metadata_file <- file.path(data_dir, "metadata.json")
  
  # Load annotations
  annotations_df <- tryCatch({
    utils::read.csv(annotations_file, stringsAsFactors = FALSE)
  }, error = function(e) {
    stop("Failed to read annotations file: ", e$message, call. = FALSE)
  })
  
  # Load URLs
  urls_data <- tryCatch({
    jsonlite::fromJSON(urls_file, simplifyVector = FALSE)
  }, error = function(e) {
    stop("Failed to read URLs file: ", e$message, call. = FALSE)
  })
  
  # Load metadata if available
  metadata <- list()
  if (file.exists(metadata_file)) {
    metadata <- tryCatch({
      jsonlite::fromJSON(metadata_file, simplifyVector = FALSE)
    }, error = function(e) {
      list()  # Ignore metadata read errors
    })
  }
  
  # Create metadata if not loaded
  if (length(metadata) == 0) {
    metadata <- list(
      n_annotations = nrow(annotations_df),
      n_urls = length(urls_data),
      columns = names(annotations_df),
      structure = "flat"
    )
  }
  
  # Print summary
  message("✓ Loaded ", nrow(annotations_df), " annotations from flat structure")
  if ("valence" %in% names(annotations_df)) {
    message("  Valence range: ", 
            round(min(annotations_df$valence, na.rm = TRUE), 3), " to ",
            round(max(annotations_df$valence, na.rm = TRUE), 3))
  }
  if ("arousal" %in% names(annotations_df)) {
    message("  Arousal range: ", 
            round(min(annotations_df$arousal, na.rm = TRUE), 3), " to ",
            round(max(annotations_df$arousal, na.rm = TRUE), 3))
  }
  
  # Return based on output format
  if (output_format == "dataframe") {
    return(annotations_df)
  } else {
    return(list(
      annotations = annotations_df,
      urls = urls_data,
      metadata = metadata
    ))
  }
}

#' Check FindingEmo Dataset Quality
#'
#' @description
#' Checks the quality and completeness of a downloaded FindingEmo dataset,
#' reporting on file availability, image accessibility, and potential issues.
#'
#' @param data_dir Character. Directory containing the FindingEmo dataset.
#' @param check_images Logical. Whether to verify image file accessibility 
#'   (default: FALSE, as this can be slow).
#' @param sample_size Integer. If check_images is TRUE, number of images to 
#'   sample for verification (default: 10).
#'
#' @return
#' A list containing:
#' \itemize{
#'   \item \code{structure}: Dataset structure type ("standard" or "flat")
#'   \item \code{files_found}: List of available files
#'   \item \code{annotations_count}: Number of annotations
#'   \item \code{urls_count}: Number of image URLs
#'   \item \code{images_count}: Number of downloaded images
#'   \item \code{completeness}: Percentage of images successfully downloaded
#'   \item \code{image_check}: Results of image accessibility check (if performed)
#' }
#'
#' @examples
#' \dontrun{
#' # Check dataset quality
#' quality_report <- check_findingemo_quality("./findingemo_data")
#' print(quality_report)
#' 
#' # Check with image verification
#' quality_report <- check_findingemo_quality(
#'   data_dir = "./findingemo_data",
#'   check_images = TRUE,
#'   sample_size = 5
#' )
#' }
#'
#' @export
check_findingemo_quality <- function(data_dir, check_images = FALSE, sample_size = 10) {
  
  if (!dir.exists(data_dir)) {
    stop("Data directory does not exist: ", data_dir, call. = FALSE)
  }
  
  # Determine structure type
  flat_annotations <- file.path(data_dir, "annotations.csv")
  flat_urls <- file.path(data_dir, "urls.json")
  standard_annotations <- file.path(data_dir, "data", "annotations_single.ann")
  standard_urls <- file.path(data_dir, "data", "dataset_urls_exploded.json")
  
  is_flat <- file.exists(flat_annotations) && file.exists(flat_urls)
  is_standard <- file.exists(standard_annotations) && file.exists(standard_urls)
  
  if (!is_flat && !is_standard) {
    stop("No FindingEmo dataset files found in directory", call. = FALSE)
  }
  
  structure_type <- if (is_flat) "flat" else "standard"
  
  # Set file paths based on structure
  annotations_file <- if (is_flat) flat_annotations else standard_annotations
  urls_file <- if (is_flat) flat_urls else standard_urls
  images_dir <- file.path(data_dir, "images")
  
  # Check files
  files_found <- list()
  files_found$annotations <- file.exists(annotations_file)
  files_found$urls <- file.exists(urls_file)
  files_found$images_dir <- dir.exists(images_dir)
  
  # Count annotations
  annotations_count <- 0
  if (files_found$annotations) {
    tryCatch({
      if (is_flat) {
        annotations_df <- utils::read.csv(annotations_file)
      } else {
        annotations_df <- utils::read.csv(annotations_file)
      }
      annotations_count <- nrow(annotations_df)
    }, error = function(e) {
      warning("Could not read annotations file: ", e$message)
    })
  }
  
  # Count URLs
  urls_count <- 0
  if (files_found$urls) {
    tryCatch({
      urls_data <- jsonlite::fromJSON(urls_file, simplifyVector = FALSE)
      urls_count <- length(urls_data)
    }, error = function(e) {
      warning("Could not read URLs file: ", e$message)
    })
  }
  
  # Count downloaded images
  images_count <- 0
  if (files_found$images_dir) {
    image_files <- list.files(
      images_dir, 
      pattern = "\\.(jpg|jpeg|png|bmp|gif)$", 
      recursive = TRUE, 
      ignore.case = TRUE
    )
    images_count <- length(image_files)
  }
  
  # Calculate completeness
  completeness <- if (urls_count > 0) {
    round((images_count / urls_count) * 100, 1)
  } else {
    0
  }
  
  # Optional image accessibility check
  image_check <- NULL
  if (check_images && images_count > 0) {
    message("Checking image accessibility...")
    
    image_files <- list.files(
      images_dir, 
      pattern = "\\.(jpg|jpeg|png|bmp|gif)$", 
      recursive = TRUE, 
      ignore.case = TRUE,
      full.names = TRUE
    )
    
    sample_files <- if (length(image_files) <= sample_size) {
      image_files
    } else {
      sample(image_files, sample_size)
    }
    
    accessible_count <- 0
    for (img_file in sample_files) {
      if (file.exists(img_file) && file.size(img_file) > 0) {
        accessible_count <- accessible_count + 1
      }
    }
    
    image_check <- list(
      sample_size = length(sample_files),
      accessible = accessible_count,
      accessibility_rate = round((accessible_count / length(sample_files)) * 100, 1)
    )
  }
  
  # Create report
  report <- list(
    structure = structure_type,
    files_found = files_found,
    annotations_count = annotations_count,
    urls_count = urls_count,
    images_count = images_count,
    completeness = completeness,
    image_check = image_check
  )
  
  # Print summary
  cat("FindingEmo Dataset Quality Report\n")
  cat("================================\n")
  cat("Structure:", structure_type, "\n")
  cat("Annotations:", annotations_count, "\n")
  cat("URLs:", urls_count, "\n")
  cat("Downloaded images:", images_count, "\n")
  cat("Completeness:", completeness, "%\n")
  
  if (!is.null(image_check)) {
    cat("Image accessibility:", image_check$accessibility_rate, 
        "% (", image_check$accessible, "/", image_check$sample_size, " sampled)\n")
  }
  
  if (completeness < 50) {
    cat("\n⚠ Warning: Low completeness rate suggests many broken URLs\n")
  }
  
  return(invisible(report))
}

#' Map FindingEmo Emotions to Emo8 Labels
#'
#' @description
#' Maps FindingEmo dataset emotion labels to the standard 8 basic emotions 
#' (Emo8) from Plutchik's emotion wheel. This function converts complex emotion 
#' labels to the 8 fundamental emotions using intensity-based mappings from 
#' the circumplex model.
#'
#' @param findingemo_emotions Character vector of FindingEmo emotion labels to map.
#'
#' @return
#' Character vector of mapped Emo8 emotion labels. Unmapped emotions return NA.
#'
#' @details
#' The mapping is based on Plutchik's circumplex model of emotions, where 
#' complex emotions are mapped to their corresponding basic emotions:
#'
#' **Basic Emotions (direct mapping):**
#' - Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation
#'
#' **Intensity Variations:**
#' - High intensity: Ecstasy→Joy, Admiration→Trust, Terror→Fear, etc.
#' - Low intensity: Serenity→Joy, Acceptance→Trust, Apprehension→Fear, etc.
#'
#' The 8 basic emotions (Emo8) are: joy, trust, fear, surprise, sadness, 
#' disgust, anger, anticipation.
#'
#' @examples
#' \dontrun{
#' # Map single emotions
#' map_to_emo8("Joy")           # "joy"
#' map_to_emo8("Ecstasy")       # "joy" 
#' map_to_emo8("Serenity")      # "joy"
#' 
#' # Map multiple emotions
#' findingemo_labels <- c("Joy", "Rage", "Terror", "Interest")
#' emo8_labels <- map_to_emo8(findingemo_labels)
#' # Returns: c("joy", "anger", "fear", "anticipation")
#'
#' # Use in evaluation pipeline
#' annotations <- load_findingemo_annotations("./data")
#' annotations$emo8_label <- map_to_emo8(annotations$emotion)
#' }
#'
#' @export
map_to_emo8 <- function(findingemo_emotions) {
  
  # Validate input
  if (!is.character(findingemo_emotions)) {
    stop("findingemo_emotions must be a character vector", call. = FALSE)
  }
  
  # Define comprehensive mapping from FindingEmo labels to Emo8
  emo8_mapping <- c(
    # Direct mappings (exact matches)
    "Joy" = "joy",
    "Trust" = "trust", 
    "Fear" = "fear",
    "Surprise" = "surprise",
    "Sadness" = "sadness",
    "Disgust" = "disgust",
    "Anger" = "anger",
    "Anticipation" = "anticipation",
    
    # Intensity variations - High intensity (outer wheel)
    "Ecstasy" = "joy",        # High intensity joy
    "Admiration" = "trust",   # High intensity trust
    "Terror" = "fear",        # High intensity fear  
    "Amazement" = "surprise", # High intensity surprise
    "Grief" = "sadness",      # High intensity sadness
    "Loathing" = "disgust",   # High intensity disgust
    "Rage" = "anger",         # High intensity anger
    "Vigilance" = "anticipation", # High intensity anticipation
    
    # Intensity variations - Low intensity (inner wheel)
    "Serenity" = "joy",       # Low intensity joy
    "Acceptance" = "trust",   # Low intensity trust
    "Apprehension" = "fear",  # Low intensity fear
    "Distraction" = "surprise", # Low intensity surprise  
    "Pensiveness" = "sadness", # Low intensity sadness
    "Boredom" = "disgust",    # Low intensity disgust
    "Annoyance" = "anger",    # Low intensity anger
    "Interest" = "anticipation" # Low intensity anticipation
  )
  
  # Apply mapping
  mapped_emotions <- emo8_mapping[findingemo_emotions]
  
  # Count successful mappings
  n_mapped <- sum(!is.na(mapped_emotions))
  n_total <- length(findingemo_emotions)
  
  if (n_mapped < n_total) {
    unmapped <- unique(findingemo_emotions[is.na(mapped_emotions)])
    warning(sprintf("Could not map %d/%d emotions to Emo8: %s", 
                   n_total - n_mapped, n_total, 
                   paste(unmapped, collapse = ", ")), 
           call. = FALSE)
  }
  
  return(as.character(mapped_emotions))
}
