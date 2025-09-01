#' Map Discrete Emotions to VAD (Valence-Arousal-Dominance) Framework
#'
#' @description
#' Maps discrete emotion classifications from image_scores(), transformer_scores(),
#' or video_scores() functions to the Valence-Arousal-Dominance (VAD) framework
#' using published lexicons. Automatically downloads the NRC VAD lexicon via
#' textdata package on first use.
#'
#' @param results Output from image_scores(), transformer_scores(), or video_scores().
#'   Can be a data.frame (from image/video functions) or a list (from transformer functions).
#' @param mapping Character. Which VAD mapping to use. Currently supports:
#'   \itemize{
#'     \item \code{"nrc_vad"}: Uses NRC VAD lexicon (Mohammad, 2018)
#'   }
#' @param weighted Logical. Whether to compute weighted averages based on confidence
#'   scores (default: TRUE). If FALSE, performs simple lookup of the highest-scoring emotion.
#' @param cache_lexicon Logical. Whether to cache the VAD lexicon for repeated use
#'   (default: TRUE).
#' @param vad_lexicon Optional data.frame. Pre-loaded VAD lexicon data to use instead
#'   of loading from textdata. Must have columns for word, valence, arousal, dominance
#'   (accepts both lowercase and capitalized versions, e.g., Word/word, Valence/valence).
#'   If provided, the function will use this data directly.
#'
#' @return
#' A data.frame with columns:
#' \itemize{
#'   \item \code{valence}: Valence score (positive vs negative emotion)
#'   \item \code{arousal}: Arousal score (excitement vs calmness)
#'   \item \code{dominance}: Dominance score (control vs submissiveness)
#' }
#' For transformer_scores() input, includes additional identifier columns.
#' For image/video_scores() input, returns one row per input row.
#'
#' @details
#' This function bridges discrete emotion classification outputs with the continuous
#' VAD emotion framework. The VAD model represents emotions in a three-dimensional
#' space where:
#' \itemize{
#'   \item **Valence**: Pleasantness (positive/negative)
#'   \item **Arousal**: Activation level (excited/calm)
#'   \item **Dominance**: Control (dominant/submissive)
#' }
#'
#' **Input Type Detection:**
#' The function automatically detects the input type:
#' \itemize{
#'   \item **data.frame**: Assumes output from image_scores() or video_scores()
#'   \item **list**: Assumes output from transformer_scores()
#' }
#'
#' **Weighting Methods:**
#' \itemize{
#'   \item **weighted = TRUE**: Computes weighted average VAD scores based on
#'     classification confidence scores
#'   \item **weighted = FALSE**: Uses VAD values for the highest-scoring emotion only
#' }
#'
#' **VAD Mappings:**
#' Currently supports the NRC VAD lexicon which provides VAD ratings for emotion
#' words based on crowdsourced annotations. The lexicon must be downloaded first
#' using `textdata::lexicon_nrc_vad()` in an interactive session.
#'
#' **Setup Required:**
#' Before using this function, download the NRC VAD lexicon by running:
#' `textdata::lexicon_nrc_vad()` in an interactive R session and accepting the license.
#'
#' @examples
#' \dontrun{
#' # Method 1: Auto-load from textdata (requires prior download)
#' textdata::lexicon_nrc_vad()  # Run once to download
#' 
#' # With image scores
#' image_path <- system.file("extdata", "boris-1.png", package = "transforEmotion")
#' emotions <- c("joy", "sadness", "anger", "fear", "surprise", "disgust")
#' img_results <- image_scores(image_path, emotions)
#' vad_results <- map_discrete_to_vad(img_results)
#'
#' # Method 2: Download once and pass as argument (recommended)
#' nrc_vad <- textdata::lexicon_nrc_vad()  # Download once
#' 
#' # Use with different emotion results
#' vad_results1 <- map_discrete_to_vad(img_results, vad_lexicon = nrc_vad)
#' 
#' text <- "I am so happy today!"
#' trans_results <- transformer_scores(text, emotions)
#' vad_results2 <- map_discrete_to_vad(trans_results, vad_lexicon = nrc_vad)
#'
#' # Simple lookup (no weighting)
#' vad_simple <- map_discrete_to_vad(img_results, weighted = FALSE, vad_lexicon = nrc_vad)
#' }
#'
#' @references
#' Mohammad, S. M. (2018). Obtaining reliable human ratings of valence, arousal, and
#' dominance for 20,000 English words. Proceedings of the 56th Annual Meeting of the
#' Association for Computational Linguistics (Volume 1: Long Papers), 174-184.
#'
#' @section Data Privacy:
#'   VAD lexicon is downloaded once and cached locally. No data is sent to external servers.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @export
#' @importFrom textdata lexicon_nrc_vad
map_discrete_to_vad <- function(results,
                                mapping = "nrc_vad",
                                weighted = TRUE,
                                cache_lexicon = TRUE,
                                vad_lexicon = NULL) {

  # Validate inputs
  if (missing(results)) {
    stop("results argument is required", call. = FALSE)
  }

  if (!mapping %in% c("nrc_vad")) {
    stop("mapping must be 'nrc_vad'", call. = FALSE)
  }

  # Get or load VAD lexicon
  if (!is.null(vad_lexicon)) {
    # Use provided lexicon data
    # Handle both lowercase and capitalized column names
    possible_word_cols <- c("word", "Word", "WORD")
    possible_val_cols <- c("valence", "Valence", "VALENCE")
    possible_aro_cols <- c("arousal", "Arousal", "AROUSAL") 
    possible_dom_cols <- c("dominance", "Dominance", "DOMINANCE")
    
    word_col <- intersect(possible_word_cols, names(vad_lexicon))[1]
    val_col <- intersect(possible_val_cols, names(vad_lexicon))[1]
    aro_col <- intersect(possible_aro_cols, names(vad_lexicon))[1]
    dom_col <- intersect(possible_dom_cols, names(vad_lexicon))[1]
    
    if (any(is.na(c(word_col, val_col, aro_col, dom_col)))) {
      stop("Provided vad_lexicon missing required columns. Expected: word/Word, valence/Valence, arousal/Arousal, dominance/Dominance\n",
           "Found columns: ", paste(names(vad_lexicon), collapse = ", "),
           call. = FALSE)
    }
    
    # Standardize column names and data
    vad_data <- data.frame(
      word = tolower(trimws(as.character(vad_lexicon[[word_col]]))),
      valence = as.numeric(vad_lexicon[[val_col]]),
      arousal = as.numeric(vad_lexicon[[aro_col]]),
      dominance = as.numeric(vad_lexicon[[dom_col]]),
      stringsAsFactors = FALSE
    )
    vad_data <- vad_data[complete.cases(vad_data), ]
  } else {
    # Load from textdata
    vad_data <- get_vad_lexicon(mapping, cache_lexicon)
  }

  # Detect input type and process accordingly
  if (is.data.frame(results)) {
    # Handle image_scores() or video_scores() output
    return(process_dataframe_input(results, vad_data, weighted))
  } else if (is.list(results)) {
    # Handle transformer_scores() output
    return(process_list_input(results, vad_data, weighted))
  } else {
    stop("results must be a data.frame (from image/video_scores) or list (from transformer_scores)",
         call. = FALSE)
  }
}

#' Get VAD lexicon data
#' @noRd
get_vad_lexicon <- function(mapping, cache_lexicon) {

  # Check if lexicon is already cached in package environment
  pkg_env <- asNamespace("transforEmotion")
  cache_name <- paste0("cached_", mapping)

  if (cache_lexicon && exists(cache_name, envir = pkg_env)) {
    return(get(cache_name, envir = pkg_env))
  }

  # Load lexicon based on mapping type
  if (mapping == "nrc_vad") {
    message("Loading NRC VAD lexicon...")

    # Try to load the lexicon (assumes user has already downloaded it)
    vad_raw <- try({
      textdata::lexicon_nrc_vad()
    }, silent = TRUE)

    if (inherits(vad_raw, "try-error")) {
      error_msg <- attr(vad_raw, "condition")$message
      stop("Failed to load NRC VAD lexicon.\n",
           "Please download it first by running in an interactive R session:\n",
           "textdata::lexicon_nrc_vad()\n",
           "Error details: ", error_msg,
           call. = FALSE)
    }

    # Process the lexicon - ensure it has the expected columns
    expected_cols <- c("word", "valence", "arousal", "dominance")
    if (!all(expected_cols %in% names(vad_raw))) {
      stop("NRC VAD lexicon missing expected columns: ",
           paste(setdiff(expected_cols, names(vad_raw)), collapse = ", "),
           call. = FALSE)
    }

    # Clean and prepare the data
    vad_data <- vad_raw[, expected_cols]
    vad_data$word <- tolower(trimws(vad_data$word))

    # Remove any rows with missing values
    vad_data <- vad_data[complete.cases(vad_data), ]

    # Cache the lexicon if requested
    if (cache_lexicon) {
      assign(cache_name, vad_data, envir = pkg_env)
      message("VAD lexicon cached for future use")
    }

    return(vad_data)
  }

  stop("Unsupported mapping: ", mapping, call. = FALSE)
}

#' Process data.frame input (from image_scores or video_scores)
#' @noRd
process_dataframe_input <- function(results, vad_data, weighted) {

  # Get emotion column names (all columns should be emotion classes)
  emotion_cols <- names(results)

  # Validate that we have emotion data
  if (length(emotion_cols) == 0) {
    stop("No emotion columns found in results", call. = FALSE)
  }

  # Check that all columns are numeric
  if (!all(sapply(results[emotion_cols], is.numeric))) {
    stop("All emotion columns must be numeric", call. = FALSE)
  }

  # Process each row
  vad_results <- data.frame(
    valence = numeric(nrow(results)),
    arousal = numeric(nrow(results)),
    dominance = numeric(nrow(results)),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(results))) {
    row_scores <- as.numeric(results[i, emotion_cols])
    names(row_scores) <- emotion_cols

    # Remove any NA scores
    valid_scores <- row_scores[!is.na(row_scores)]

    if (length(valid_scores) == 0) {
      # If no valid scores, return NA
      vad_results[i, ] <- c(NA, NA, NA)
      next
    }

    if (weighted) {
      # Weighted average based on scores
      vad_results[i, ] <- compute_weighted_vad(valid_scores, vad_data)
    } else {
      # Use highest scoring emotion only
      top_emotion <- names(valid_scores)[which.max(valid_scores)]
      vad_results[i, ] <- lookup_emotion_vad(top_emotion, vad_data)
    }
  }

  return(vad_results)
}

#' Process list input (from transformer_scores)
#' @noRd
process_list_input <- function(results, vad_data, weighted) {

  # Get text identifiers (names of the list)
  text_ids <- names(results)
  if (is.null(text_ids)) {
    text_ids <- paste0("text_", seq_along(results))
  }

  # Process each text
  vad_results <- data.frame(
    text_id = text_ids,
    valence = numeric(length(results)),
    arousal = numeric(length(results)),
    dominance = numeric(length(results)),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(results)) {
    scores <- results[[i]]

    # Validate scores
    if (!is.numeric(scores) || is.null(names(scores))) {
      warning("Invalid scores for text ", i, ": skipping", call. = FALSE)
      vad_results[i, c("valence", "arousal", "dominance")] <- c(NA, NA, NA)
      next
    }

    # Remove any NA scores
    valid_scores <- scores[!is.na(scores)]

    if (length(valid_scores) == 0) {
      # If no valid scores, return NA
      vad_results[i, c("valence", "arousal", "dominance")] <- c(NA, NA, NA)
      next
    }

    if (weighted) {
      # Weighted average based on scores
      vad_vals <- compute_weighted_vad(valid_scores, vad_data)
    } else {
      # Use highest scoring emotion only
      top_emotion <- names(valid_scores)[which.max(valid_scores)]
      vad_vals <- lookup_emotion_vad(top_emotion, vad_data)
    }

    vad_results[i, c("valence", "arousal", "dominance")] <- vad_vals
  }

  return(vad_results)
}

#' Compute weighted VAD scores
#' @noRd
compute_weighted_vad <- function(scores, vad_data) {

  emotions <- names(scores)

  # Look up VAD values for each emotion
  vad_values <- matrix(NA, nrow = length(emotions), ncol = 3)
  colnames(vad_values) <- c("valence", "arousal", "dominance")

  for (i in seq_along(emotions)) {
    emotion_vad <- lookup_emotion_vad(emotions[i], vad_data)
    vad_values[i, ] <- emotion_vad
  }

  # Remove emotions that couldn't be mapped
  valid_rows <- complete.cases(vad_values)
  if (!any(valid_rows)) {
    warning("No emotions could be mapped to VAD values", call. = FALSE)
    return(c(valence = NA, arousal = NA, dominance = NA))
  }

  # Filter to valid mappings
  valid_vad <- vad_values[valid_rows, , drop = FALSE]
  valid_scores <- scores[valid_rows]

  # Normalize scores to sum to 1
  normalized_scores <- valid_scores / sum(valid_scores)

  # Compute weighted average
  weighted_vad <- colSums(valid_vad * normalized_scores)

  return(weighted_vad)
}

#' Look up VAD values for a single emotion
#' @noRd
lookup_emotion_vad <- function(emotion, vad_data) {

  emotion_clean <- tolower(trimws(emotion))

  # Direct lookup
  match_idx <- which(vad_data$word == emotion_clean)

  if (length(match_idx) > 0) {
    # Use first match if multiple
    idx <- match_idx[1]
    return(c(
      valence = vad_data$valence[idx],
      arousal = vad_data$arousal[idx],
      dominance = vad_data$dominance[idx]
    ))
  }

  # If no exact match, try partial matching
  partial_matches <- grep(emotion_clean, vad_data$word, value = FALSE)

  if (length(partial_matches) > 0) {
    # Use first partial match
    idx <- partial_matches[1]
    return(c(
      valence = vad_data$valence[idx],
      arousal = vad_data$arousal[idx],
      dominance = vad_data$dominance[idx]
    ))
  }

  # If still no match, warn and return NA
  warning("Could not find VAD values for emotion: ", emotion, call. = FALSE)
  return(c(valence = NA, arousal = NA, dominance = NA))
}
