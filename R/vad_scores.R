#' Direct VAD (Valence-Arousal-Dominance) Prediction
#'
#' @description
#' Directly predicts VAD dimensions using classification with definitional labels,
#' bypassing the intermediate step of discrete emotion classification. This approach
#' uses rich, educational descriptions of each VAD pole to help transformer models
#' understand the psychological concepts and make more accurate predictions.
#'
#' @param input Input data. Can be:
#'   \itemize{
#'     \item Character: Text string, image file path, or video URL
#'     \item Character vector: Multiple texts or image paths
#'     \item List: Multiple text strings
#'   }
#' @param input_type Character. Type of input data:
#'   \itemize{
#'     \item \code{"auto"}: Automatically detect based on input (default)
#'     \item \code{"text"}: Text input for transformer classification
#'     \item \code{"image"}: Image file path(s) for visual classification
#'     \item \code{"video"}: Video URL(s) for video analysis
#'   }
#' @param dimensions Character vector. Which VAD dimensions to predict:
#'   \itemize{
#'     \item \code{"valence"}: Positive vs negative emotional experience
#'     \item \code{"arousal"}: High vs low activation/energy
#'     \item \code{"dominance"}: Control vs powerlessness
#'   }
#'   Default: all three dimensions
#' @param label_type Character. Type of labels to use:
#'   \itemize{
#'     \item \code{"definitional"}: Rich descriptive labels with definitions (default)
#'     \item \code{"simple"}: Basic polar labels (positive/negative, etc.)
#'     \item \code{"custom"}: User-provided custom labels
#'   }
#' @param custom_labels Optional list. Custom labels when label_type = "custom".
#'   Must follow structure: list(valence = list(positive = "...", negative = "..."), ...)
#' @param model Character. Model to use for classification. Depends on input_type:
#'   \itemize{
#'     \item Text: transformer model (see transformer_scores documentation)
#'     \item Image: CLIP model (see image_scores documentation)  
#'     \item Video: CLIP model (see video_scores documentation)
#'   }
#' @param ... Additional arguments passed to underlying classification functions
#'   (transformer_scores, image_scores, or video_scores)
#'
#' @return
#' A data.frame with columns:
#' \itemize{
#'   \item \code{input_id}: Identifier for each input (text content, filename, or index)
#'   \item \code{valence}: Valence score (0-1, where 1 = positive)
#'   \item \code{arousal}: Arousal score (0-1, where 1 = high arousal) 
#'   \item \code{dominance}: Dominance score (0-1, where 1 = high dominance)
#' }
#' Only requested dimensions are included in output.
#'
#' @details
#' This function implements direct VAD prediction using the approach:
#' Input → VAD Classification → VAD Scores
#' 
#' Instead of mapping from discrete emotions, each VAD dimension is treated as
#' a separate binary classification task using definitional labels that explain
#' the psychological concepts.
#'
#' **Definitional Labels (default):**
#' The function uses rich descriptions that educate the model about each dimension:
#' \itemize{
#'   \item **Valence**: "Positive valence, which refers to pleasant, enjoyable..."
#'   \item **Arousal**: "High arousal, which refers to intense, energetic..."  
#'   \item **Dominance**: "High dominance, which refers to feeling in control..."
#' }
#'
#' **Input Type Detection:**
#' When input_type = "auto", the function detects input type based on:
#' \itemize{
#'   \item URLs starting with "http": Video
#'   \item File paths with image extensions: Image
#'   \item Everything else: Text
#' }
#'
#' **Score Interpretation:**
#' Scores represent the probability that the input exhibits the "high" pole:
#' \itemize{
#'   \item **Valence**: 1.0 = very positive, 0.0 = very negative
#'   \item **Arousal**: 1.0 = high energy, 0.0 = very calm
#'   \item **Dominance**: 1.0 = very controlling, 0.0 = very powerless
#' }
#'
#' @examples
#' \dontrun{
#' # Text VAD analysis
#' texts <- c("I'm absolutely thrilled!", "I feel so helpless and sad", "This is boring")
#' text_vad <- vad_scores(texts, input_type = "text")
#' print(text_vad)
#'
#' # Image VAD analysis  
#' image_path <- system.file("extdata", "boris-1.png", package = "transforEmotion")
#' image_vad <- vad_scores(image_path, input_type = "image")
#' print(image_vad)
#'
#' # Single dimension prediction
#' valence_only <- vad_scores(texts, dimensions = "valence")
#' 
#' # Using simple labels for speed
#' simple_vad <- vad_scores(texts, label_type = "simple")
#'
#' # Custom labels for domain-specific applications
#' custom_labels <- list(
#'   valence = list(
#'     positive = "Customer satisfaction and positive brand sentiment",
#'     negative = "Customer complaints and negative brand sentiment"
#'   )
#' )
#' brand_vad <- vad_scores(texts, dimensions = "valence", 
#'                         label_type = "custom", custom_labels = custom_labels)
#' }
#'
#' @references
#' Russell, J. A. (1980). A circumplex model of affect. Journal of Personality 
#' and Social Psychology, 39(6), 1161-1178.
#' 
#' Bradley, M. M., & Lang, P. J. (1994). Measuring emotion: the self-assessment 
#' manikin and the semantic differential. Journal of Behavior Therapy and 
#' Experimental Psychiatry, 25(1), 49-59.
#'
#' @section Data Privacy:
#'   All processing is done locally with downloaded models. Data is never sent 
#'   to external servers.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @export
vad_scores <- function(input,
                      input_type = "auto",
                      dimensions = c("valence", "arousal", "dominance"),
                      label_type = "definitional",
                      custom_labels = NULL,
                      model = "auto",
                      ...) {
  
  # Validate inputs
  if (missing(input)) {
    stop("input argument is required", call. = FALSE)
  }
  
  # Validate dimensions
  valid_dimensions <- c("valence", "arousal", "dominance")
  if (!all(dimensions %in% valid_dimensions)) {
    stop("dimensions must be one or more of: ", 
         paste(valid_dimensions, collapse = ", "), call. = FALSE)
  }
  
  # Validate input_type
  valid_input_types <- c("auto", "text", "image", "video")
  input_type <- match.arg(input_type, valid_input_types)
  
  # Validate label_type
  valid_label_types <- c("definitional", "simple", "custom")
  label_type <- match.arg(label_type, valid_label_types)
  
  # Auto-detect input type if needed
  if (input_type == "auto") {
    input_type <- detect_input_type(input)
  }
  
  # Get VAD labels
  vad_labels <- get_vad_labels(label_type, custom_labels)
  
  # Process each dimension
  results_list <- list()
  
  for (dim in dimensions) {
    dim_labels <- vad_labels[[dim]]
    classes <- format_labels_for_classification(dim_labels)
    
    # Run classification for this dimension
    dim_scores <- run_vad_classification(input, input_type, classes, model, ...)
    
    # Extract scores for the "high" pole (positive/high arousal/high dominance)
    high_pole_score <- extract_high_pole_score(dim_scores, dim, classes)
    
    # Check if we got NA values and fall back to simple labels if needed
    if (all(is.na(high_pole_score))) {
      warning("Definitional labels failed for dimension '", dim, "'. Falling back to simple labels.", 
              call. = FALSE)
      
      # Get simple labels as fallback
      simple_labels <- get_vad_simple_labels()
      simple_classes <- format_labels_for_classification(simple_labels[[dim]])
      
      # Retry with simple labels
      dim_scores_simple <- run_vad_classification(input, input_type, simple_classes, model, ...)
      high_pole_score <- extract_high_pole_score(dim_scores_simple, dim, simple_classes)
    }
    
    results_list[[dim]] <- high_pole_score
  }
  
  # Combine results into data.frame
  result_df <- combine_vad_results(results_list, input, input_type)
  
  return(result_df)
}

#' Detect input type automatically
#' @noRd
detect_input_type <- function(input) {
  
  # Handle different input formats
  if (is.list(input)) {
    sample_input <- input[[1]]
  } else {
    sample_input <- input[1]
  }
  
  if (!is.character(sample_input)) {
    stop("Input must be character (text, file path, or URL)", call. = FALSE)
  }
  
  # Check for URLs (video)
  if (grepl("^https?://", sample_input)) {
    return("video")
  }
  
  # Check for image file extensions
  image_extensions <- c("jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp")
  ext_pattern <- paste0("\\.(", paste(image_extensions, collapse = "|"), ")$")
  if (grepl(ext_pattern, sample_input, ignore.case = TRUE)) {
    return("image")
  }
  
  # Default to text
  return("text")
}

#' Run classification for a single VAD dimension
#' @noRd
run_vad_classification <- function(input, input_type, classes, model, ...) {
  
  if (input_type == "text") {
    # Use transformer_scores for text
    model_arg <- if (model == "auto") "cross-encoder-distilroberta" else model
    return(transformer_scores(input, classes, transformer = model_arg, ...))
    
  } else if (input_type == "image") {
    # Use image_scores for images
    model_arg <- if (model == "auto") "oai-base" else model
    
    if (length(input) == 1) {
      # Single image
      return(image_scores(input, classes, model = model_arg, ...))
    } else {
      # Multiple images - use image_scores_dir if all in same directory
      # Otherwise, process individually
      results_list <- list()
      for (i in seq_along(input)) {
        img_result <- image_scores(input[i], classes, model = model_arg, ...)
        img_result$image_id <- basename(input[i])
        results_list[[i]] <- img_result
      }
      return(combine_image_results(results_list, input))
    }
    
  } else if (input_type == "video") {
    # Use video_scores for videos
    model_arg <- if (model == "auto") "oai-base" else model
    
    if (length(input) == 1) {
      # Single video
      return(video_scores(input, classes, model = model_arg, ...))
    } else {
      # Multiple videos
      results_list <- list()
      for (i in seq_along(input)) {
        vid_result <- video_scores(input[i], classes, model = model_arg, ...)
        vid_result$video_id <- input[i]
        results_list[[i]] <- vid_result
      }
      return(combine_video_results(results_list, input))
    }
    
  } else {
    stop("Unsupported input_type: ", input_type, call. = FALSE)
  }
}

#' Extract score for the "high" pole of a dimension
#' @noRd
extract_high_pole_score <- function(scores, dimension, classes) {
  
  # Handle data.frames (image/video) before generic list check since
  # data.frame inherits from list in R.
  if (is.data.frame(scores)) {
    # Image/video scores format - for definitional labels, always use first column
    # because format_labels_for_classification puts the "high" pole first
    
    # Since column names may be truncated/modified due to long definitional strings,
    # we'll just use the first column (which corresponds to the "high" pole)
    if (ncol(scores) >= 1) {
      # Extract just the first column values (not the entire column object)
      high_pole_scores <- scores[, 1, drop = TRUE]  # First column = high pole
    } else {
      stop("No columns found in image/video scores", call. = FALSE)
    }
    
    # Ensure we return a single numeric value for each input
    return(as.numeric(high_pole_scores))
  }
  
  if (is.list(scores)) {
    # Transformer scores format - extract high pole score for each text
    if (dimension == "valence") {
      high_pole_class <- classes[1] # positive
    } else {
      high_pole_class <- classes[1] # high arousal/dominance
    }
    return(sapply(scores, function(x) x[high_pole_class]))
  }
  
  stop("Unexpected scores format: ", class(scores), call. = FALSE)
}

#' Combine results from multiple images
#' @noRd
combine_image_results <- function(results_list, input) {
  # Combine multiple single-image results into one data.frame
  combined <- do.call(rbind, results_list)
  combined$image_id <- basename(input)
  return(combined)
}

#' Combine results from multiple videos  
#' @noRd
combine_video_results <- function(results_list, input) {
  # Combine multiple single-video results into one data.frame
  combined <- do.call(rbind, results_list)
  combined$video_id <- input
  return(combined)
}

#' Combine VAD dimension results into final output
#' @noRd
combine_vad_results <- function(results_list, input, input_type) {
  
  # Determine number of inputs and create input IDs
  if (is.list(input) && input_type == "text") {
    # Transformer scores with named list
    input_ids <- names(input)
    if (is.null(input_ids)) {
      input_ids <- paste0("text_", seq_along(input))
    }
    n_inputs <- length(input)
  } else if (is.character(input) && length(input) > 1) {
    # Multiple inputs (images, videos, or texts)
    if (input_type == "text") {
      input_ids <- input  # Use text content as ID
    } else {
      input_ids <- basename(input)  # Use filename as ID
    }
    n_inputs <- length(input)
  } else {
    # Single input
    if (input_type == "text") {
      input_ids <- as.character(input)
    } else {
      input_ids <- basename(input)
    }
    n_inputs <- 1
  }
  
  # Create result data frame
  result_df <- data.frame(
    input_id = input_ids,
    stringsAsFactors = FALSE
  )
  
  # Add each dimension's scores
  for (dim in names(results_list)) {
    scores <- results_list[[dim]]
    
    # Ensure scores match number of inputs
    if (length(scores) != n_inputs) {
      if (length(scores) == 1 && n_inputs > 1) {
        # Replicate single score for multiple inputs (shouldn't happen normally)
        scores <- rep(scores, n_inputs)
      } else {
        stop("Mismatch between number of inputs (", n_inputs, 
             ") and scores for dimension ", dim, " (", length(scores), ")", call. = FALSE)
      }
    }
    
    result_df[[dim]] <- scores
  }
  
  return(result_df)
}
