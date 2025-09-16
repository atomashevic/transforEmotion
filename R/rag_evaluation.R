#' Validate RAG Emotion/Sentiment Predictions
#'
#' @description
#' Evaluates emotion/sentiment predictions from rag() or rag_sentemo()
#' against ground truth labels using the same metrics pipeline as
#' evaluate_emotions(). Supports table or JSON structured outputs.
#'
#' @param rag_output Output from rag() or rag_sentemo() with structured outputs
#'   (data.frame with columns like `doc_id`, `label`, `confidence`; or JSON
#'   string with these fields). Global schema outputs with `labels`/`confidences`
#'   are also handled by reducing to the top label.
#' @param ground_truth Character vector of ground truth labels matching the
#'   number of predictions (or length of provided ids).
#' @param id_col Optional identifier. If `rag_output` is a data.frame and
#'   `id_col` is a character scalar naming a column present in it, that column
#'   is used as the prediction id. Alternatively, `id_col` can be a vector of
#'   ids (same length as `ground_truth`) used to align ground truth to the
#'   predictions by merge.
#' @param task Task type: one of `"emotion"` or `"sentiment"` (used for
#'   metadata and optional label set enforcement).
#' @param labels_set Optional character vector of allowed labels for
#'   validation. If provided, predictions will be lowercased and filtered to
#'   this set where possible.
#' @param metrics Metrics to compute, forwarded to evaluate_emotions()
#'   (e.g., `c("accuracy","f1_macro","confusion_matrix")`).
#' @param return_plot Logical; whether to include plotting helpers.
#'
#' @return A list of evaluation results in the same format as
#'   evaluate_emotions(), augmented with `$rag_metadata` summarizing
#'   RAG-specific context (documents, transformer, task).
#'
#' @examples
#' \dontrun{
#' texts <- c(
#'   "I feel so happy and grateful today!",
#'   "This is frustrating and makes me angry.",
#'   "I'm not sure how I feel about this."
#' )
#'
#' # Get predictions (structured per-document output)
#' rag_results <- rag_sentemo(
#'   texts,
#'   task = "emotion",
#'   output = "table",
#'   transformer = "Gemma3-1B"
#' )
#'
#' # Ground truth labels
#' ground_truth <- c("joy", "anger", "neutral")
#'
#' # Validate predictions
#' validation_results <- validate_rag_predictions(
#'   rag_output = rag_results,
#'   ground_truth = ground_truth,
#'   task = "emotion",
#'   metrics = c("accuracy", "f1_macro", "confusion_matrix"),
#'   return_plot = TRUE
#' )
#' }
#'
#' @export
#' @importFrom jsonlite fromJSON
validate_rag_predictions <- function(rag_output, ground_truth,
                                     id_col = NULL,
                                     task = c("emotion", "sentiment"),
                                     labels_set = NULL,
                                     metrics = c("accuracy", "f1_macro",
                                                 "confusion_matrix"),
                                     return_plot = FALSE)
{
  task <- match.arg(task)

  # Extract predictions from rag_output based on format
  predictions <- extract_predictions_from_rag(rag_output, id_col)

  # Optional label set enforcement (lowercase normalization)
  if (!is.null(labels_set) && length(predictions$pred) > 0) {
    allowed <- tolower(labels_set)
    pred_low <- tolower(predictions$pred)
    keep <- pred_low %in% allowed
    if (any(!keep)) {
      # Keep rows matching allowed set; drop others with a warning
      n_drop <- sum(!keep)
      if (n_drop > 0) {
        warning(sprintf("Filtered %d predictions not in labels_set", n_drop), call. = FALSE)
      }
      predictions <- predictions[keep, , drop = FALSE]
    }
    predictions$pred <- pred_low[keep]
  }

  # Create evaluation dataset
  eval_data <- create_evaluation_dataset(predictions, ground_truth, id_col)

  # Forward to evaluate_emotions (probability columns not available here)
  results <- evaluate_emotions(
    data = eval_data,
    truth_col = "truth",
    pred_col = "pred",
    probs_cols = NULL,
    classes = labels_set,
    metrics = metrics,
    return_plot = return_plot
  )

  # Add RAG-specific metadata
  results$rag_metadata <- list(
    n_documents = tryCatch({
      if (is.data.frame(rag_output) && nrow(rag_output) > 0) {
        nrow(rag_output)
      } else if (is.character(rag_output) && length(rag_output) == 1) {
        parsed <- try(jsonlite::fromJSON(rag_output), silent = TRUE)
        if (!inherits(parsed, "try-error")) {
          if (is.data.frame(parsed)) nrow(parsed) else if (is.list(parsed) && !is.null(parsed$labels)) length(parsed$labels) else NA_integer_
        } else NA_integer_
      } else if (is.list(rag_output) && !is.null(rag_output$content)) {
        tryCatch(nrow(rag_output$content), error = function(e) NA_integer_)
      } else {
        NA_integer_
      }
    }, error = function(e) NA_integer_),
    transformer = tryCatch(attr(rag_output, "transformer"), error = function(e) NULL),
    task = task
  )

  return(results)
}

#' Helper: Extract predictions from RAG output
#' @noRd
extract_predictions_from_rag <- function(rag_output, id_col = NULL)
{
  # Case 1: Already a data.frame (e.g., rag_sentemo(..., output = "table"))
  if (is.data.frame(rag_output)) {
    # Choose id column: explicit name in data, else doc_id if present, else sequence
    id_vec <- if (!is.null(id_col) && is.character(id_col) && length(id_col) == 1 && id_col %in% names(rag_output)) {
      rag_output[[id_col]]
    } else if ("doc_id" %in% names(rag_output)) {
      rag_output[["doc_id"]]
    } else {
      seq_len(nrow(rag_output))
    }

    # Choose label/confidence columns if present
    lab_col <- if ("label" %in% names(rag_output)) "label" else if ("pred" %in% names(rag_output)) "pred" else NULL
    conf_col <- if ("confidence" %in% names(rag_output)) "confidence" else NULL

    if (is.null(lab_col)) {
      stop("Data frame rag_output must contain a 'label' (or 'pred') column", call. = FALSE)
    }

    return(data.frame(
      id = id_vec,
      pred = rag_output[[lab_col]],
      confidence = if (!is.null(conf_col)) suppressWarnings(as.numeric(rag_output[[conf_col]])) else NA_real_,
      stringsAsFactors = FALSE
    ))
  }

  # Case 2: JSON string
  if (is.character(rag_output) && length(rag_output) == 1L) {
    parsed <- try(jsonlite::fromJSON(rag_output), silent = TRUE)
    if (inherits(parsed, "try-error")) {
      stop("Failed to parse JSON rag_output", call. = FALSE)
    }

    # Per-document array of objects: data.frame with label/confidence
    if (is.data.frame(parsed) && ("label" %in% names(parsed) || "pred" %in% names(parsed))) {
      lab_col <- if ("label" %in% names(parsed)) "label" else "pred"
      conf_col <- if ("confidence" %in% names(parsed)) "confidence" else NULL
      id_vec <- if (!is.null(id_col) && is.character(id_col) && length(id_col) == 1 && id_col %in% names(parsed)) {
        parsed[[id_col]]
      } else if ("doc_id" %in% names(parsed)) {
        parsed[["doc_id"]]
      } else {
        seq_len(nrow(parsed))
      }
      return(data.frame(
        id = id_vec,
        pred = parsed[[lab_col]],
        confidence = if (!is.null(conf_col)) suppressWarnings(as.numeric(parsed[[conf_col]])) else NA_real_,
        stringsAsFactors = FALSE
      ))
    }

    # Global schema: {labels:[...], confidences:[...]}; reduce to argmax
    if (is.list(parsed) && !is.null(parsed$labels)) {
      lbls <- as.character(parsed$labels)
      confs <- tryCatch(suppressWarnings(as.numeric(parsed$confidences)), error = function(e) rep(NA_real_, length(lbls)))
      idx <- if (length(confs) == length(lbls) && any(is.finite(confs))) which.max(confs) else 1L
      return(data.frame(
        id = 1L,
        pred = lbls[idx],
        confidence = if (length(confs) >= idx && is.finite(confs[idx])) confs[idx] else NA_real_,
        stringsAsFactors = FALSE
      ))
    }

    stop("Unsupported JSON structure in rag_output", call. = FALSE)
  }

  stop("Unsupported rag_output format. Provide data.frame or JSON string from rag(..., output=\"table\"/\"json\").", call. = FALSE)
}

#' Helper: Create evaluation dataset (align predictions and ground truth)
#' @noRd
create_evaluation_dataset <- function(predictions, ground_truth, id_col = NULL)
{
  # If id_col is a vector of ids for the ground truth, merge on id
  if (!is.null(id_col) && !is.character(id_col) && length(id_col) == length(ground_truth)) {
    truth_df <- data.frame(
      id = id_col,
      truth = ground_truth,
      stringsAsFactors = FALSE
    )
    eval_data <- merge(truth_df, predictions, by = "id", all = FALSE)
    if (nrow(eval_data) == 0) {
      stop("No overlap between provided ids and prediction ids", call. = FALSE)
    }
    return(eval_data[, c("id", "truth", "pred", "confidence")])
  }

  # Else assume sequential alignment
  if (length(ground_truth) != nrow(predictions)) {
    warning(sprintf(
      "Length mismatch: ground_truth (%d) vs predictions (%d); truncating to min length",
      length(ground_truth), nrow(predictions)
    ), call. = FALSE)
  }
  n <- min(length(ground_truth), nrow(predictions))
  data.frame(
    id = if (!is.null(predictions$id)) predictions$id[seq_len(n)] else seq_len(n),
    truth = ground_truth[seq_len(n)],
    pred = predictions$pred[seq_len(n)],
    confidence = if (!is.null(predictions$confidence)) predictions$confidence[seq_len(n)] else NA_real_,
    stringsAsFactors = FALSE
  )
}

