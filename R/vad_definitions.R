#' VAD Definitional Labels
#'
#' @description
#' This file contains the definitional label pairs for each VAD dimension.
#' These labels provide rich, educational descriptions that help transformer
#' models understand the psychological concepts behind each dimension.
#'
#' @details
#' The definitional approach uses the format:
#' "Dimension name, which refers to [definition], such as [examples]"
#' 
#' This approach is more informative than simple polar labels because it:
#' 1. Educates the model about the psychological theory
#' 2. Provides clear definitions with multiple examples
#' 3. Reduces ambiguity in classification
#' 4. Works robustly across different domains and contexts
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @noRd

#' Get default VAD definitional labels
#' @noRd
get_vad_definitions <- function() {
  list(
    valence = list(
      positive = "Positive emotional experience: pleasant, joyful, happy, satisfied, hopeful",
      negative = "Negative emotional experience: unpleasant, sad, angry, fearful, distressed"
    ),
    
    arousal = list(
      high = "High energy emotional state: intense, excited, energetic, stimulating, alert", 
      low = "Low energy emotional state: calm, relaxed, peaceful, subdued, tranquil"
    ),
    
    dominance = list(
      high = "Feeling in control: powerful, confident, dominant, assertive, commanding",
      low = "Feeling controlled by others: powerless, submissive, helpless, vulnerable, weak"
    )
  )
}

#' Get simple VAD labels (for speed/compatibility)
#' @noRd
get_vad_simple_labels <- function() {
  list(
    valence = list(
      positive = "positive",
      negative = "negative"
    ),
    
    arousal = list(
      high = "excited",
      low = "calm"
    ),
    
    dominance = list(
      high = "dominant", 
      low = "submissive"
    )
  )
}

#' Validate custom VAD labels
#' @noRd
validate_vad_labels <- function(custom_labels) {
  required_structure <- c("valence", "arousal", "dominance")
  required_poles <- c("positive", "negative", "high", "low", "high", "low")
  
  if (!is.list(custom_labels)) {
    stop("custom_labels must be a list", call. = FALSE)
  }
  
  # Check top-level structure
  if (!all(names(custom_labels) %in% required_structure)) {
    stop("custom_labels must have elements named: ", 
         paste(required_structure, collapse = ", "), call. = FALSE)
  }
  
  # Check each dimension
  for (dim in names(custom_labels)) {
    dim_labels <- custom_labels[[dim]]
    
    if (!is.list(dim_labels)) {
      stop("Each dimension in custom_labels must be a list", call. = FALSE)
    }
    
    expected_poles <- switch(dim,
      "valence" = c("positive", "negative"),
      "arousal" = c("high", "low"), 
      "dominance" = c("high", "low")
    )
    
    if (!all(expected_poles %in% names(dim_labels))) {
      stop("Dimension '", dim, "' must have poles: ", 
           paste(expected_poles, collapse = ", "), call. = FALSE)
    }
    
    # Check that labels are character strings
    for (pole in expected_poles) {
      if (!is.character(dim_labels[[pole]]) || length(dim_labels[[pole]]) != 1) {
        stop("Label for ", dim, "$", pole, " must be a single character string", 
             call. = FALSE)
      }
      
      if (nchar(dim_labels[[pole]]) < 5) {
        warning("Label for ", dim, "$", pole, " is very short. Consider using more descriptive labels.", 
                call. = FALSE)
      }
    }
  }
  
  return(TRUE)
}

#' Get VAD labels based on type
#' @noRd
get_vad_labels <- function(label_type = "definitional", custom_labels = NULL) {
  
  if (label_type == "custom") {
    if (is.null(custom_labels)) {
      stop("custom_labels must be provided when label_type = 'custom'", call. = FALSE)
    }
    validate_vad_labels(custom_labels)
    return(custom_labels)
  }
  
  if (label_type == "definitional") {
    return(get_vad_definitions())
  } else if (label_type == "simple") {
    return(get_vad_simple_labels())
  } else {
    stop("label_type must be 'definitional', 'simple', or 'custom'", call. = FALSE)
  }
}

#' Format labels for classification
#' @noRd
format_labels_for_classification <- function(dimension_labels) {
  # Extract the two poles and return as a character vector
  return(c(dimension_labels$high %||% dimension_labels$positive, 
           dimension_labels$low %||% dimension_labels$negative))
}

#' Helper function for null coalescing
#' @noRd
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}