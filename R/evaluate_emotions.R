#' Evaluate Emotion Classification Performance
#'
#' @description
#' Comprehensive evaluation function for discrete emotion classification tasks.
#' Computes standard classification metrics including accuracy, F1-scores, 
#' AUROC, calibration metrics, and inter-rater reliability measures.
#'
#' @param data A data frame or file path to CSV containing evaluation data.
#'   Must include columns for identifiers, ground truth, predictions, and 
#'   optionally class probabilities.
#' @param id_col Character. Name of column containing unique identifiers 
#'   (default: "id").
#' @param truth_col Character. Name of column containing ground truth labels 
#'   (default: "truth").
#' @param pred_col Character. Name of column containing predicted labels 
#'   (default: "pred").
#' @param probs_cols Character vector. Names of columns containing class 
#'   probabilities. If NULL, probabilistic metrics will be skipped.
#' @param classes Character vector. Emotion classes to evaluate. If NULL, 
#'   will be inferred from the data.
#' @param metrics Character vector. Metrics to compute. Options include:
#'   "accuracy", "precision", "recall", "f1_macro", "f1_micro", "auroc", 
#'   "ece", "krippendorff", "confusion_matrix" (default: all metrics).
#' @param return_plot Logical. Whether to return plotting helpers 
#'   (default: FALSE).
#' @param na_rm Logical. Whether to remove missing values (default: TRUE).
#'
#' @return
#' A list containing:
#' \itemize{
#'   \item \code{metrics}: Data frame with computed evaluation metrics
#'   \item \code{confusion_matrix}: Confusion matrix (if requested)
#'   \item \code{per_class}: Per-class metrics breakdown
#'   \item \code{summary}: Overall performance summary
#'   \item \code{plot_data}: Data prepared for plotting (if return_plot = TRUE)
#' }
#'
#' @details
#' This function implements a comprehensive evaluation pipeline for discrete
#' emotion classification following best practices from the literature.
#' 
#' **Metrics computed:**
#' \itemize{
#'   \item **Accuracy**: Overall classification accuracy
#'   \item **Precision/Recall/F1**: Per-class and macro/micro averages
#'   \item **AUROC**: Area under ROC curve (requires probability scores)
#'   \item **ECE**: Expected Calibration Error for probability calibration
#'   \item **Krippendorff's α**: Inter-rater reliability between human and model
#' }
#' 
#' **Input format:**
#' The input data should contain at minimum:
#' \itemize{
#'   \item ID column: Unique identifier for each instance
#'   \item Truth column: Ground truth emotion labels
#'   \item Prediction column: Model predicted emotion labels
#'   \item Probability columns (optional): Class probabilities for each emotion
#' }
#'
#' @examples
#' \dontrun{
#' # Basic evaluation with predicted labels only
#' results <- evaluate_emotions(
#'   data = evaluation_data,
#'   truth_col = "human_label",
#'   pred_col = "model_prediction"
#' )
#' 
#' # Full evaluation with probabilities
#' results <- evaluate_emotions(
#'   data = evaluation_data,
#'   truth_col = "ground_truth",
#'   pred_col = "predicted_class",
#'   probs_cols = c("prob_anger", "prob_joy", "prob_sadness"),
#'   return_plot = TRUE
#' )
#' 
#' # Custom metrics selection
#' results <- evaluate_emotions(
#'   data = evaluation_data,
#'   metrics = c("accuracy", "f1_macro", "confusion_matrix")
#' )
#' }
#'
#' @references
#' Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class 
#' classification: an overview. arXiv preprint arXiv:2008.05756.
#' 
#' Krippendorff, K. (2011). Computing Krippendorff's alpha-reliability. 
#' Scholarly commons, 25.
#' 
#' Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well 
#' calibrated probabilities using bayesian binning. In AAAI (pp. 2901-2907).
#'
#' @seealso
#' \code{\link{transformer_scores}}, \code{\link{nlp_scores}}, 
#' \code{\link{emoxicon_scores}} for emotion prediction functions.
#'
#' @export
evaluate_emotions <- function(data,
                            id_col = "id",
                            truth_col = "truth", 
                            pred_col = "pred",
                            probs_cols = NULL,
                            classes = NULL,
                            metrics = c("accuracy", "precision", "recall", 
                                      "f1_macro", "f1_micro", "auroc", 
                                      "ece", "krippendorff", "confusion_matrix"),
                            return_plot = FALSE,
                            na_rm = TRUE) {
  
  # Input validation
  eval_data <- .validate_evaluation_input(
    data = data,
    id_col = id_col,
    truth_col = truth_col,
    pred_col = pred_col,
    probs_cols = probs_cols,
    na_rm = na_rm
  )
  
  # Infer classes if not provided
  if (is.null(classes)) {
    classes <- .infer_emotion_classes(eval_data, truth_col, pred_col)
  }
  
  # Initialize results list
  results <- list()
  
  # Compute requested metrics
  if ("accuracy" %in% metrics) {
    results$accuracy <- .compute_accuracy(eval_data, truth_col, pred_col)
  }
  
  if (any(c("precision", "recall", "f1_macro", "f1_micro") %in% metrics)) {
    classification_metrics <- .compute_classification_metrics(
      eval_data, truth_col, pred_col, classes, metrics
    )
    results <- c(results, classification_metrics)
  }
  
  if ("confusion_matrix" %in% metrics) {
    results$confusion_matrix <- .compute_confusion_matrix(
      eval_data, truth_col, pred_col, classes
    )
  }
  
  if ("auroc" %in% metrics && !is.null(probs_cols)) {
    results$auroc <- .compute_auroc(eval_data, truth_col, probs_cols, classes)
  }
  
  if ("ece" %in% metrics && !is.null(probs_cols)) {
    results$ece <- .compute_ece(eval_data, truth_col, pred_col, probs_cols)
  }
  
  if ("krippendorff" %in% metrics) {
    results$krippendorff_alpha <- .compute_krippendorff_alpha(
      eval_data, truth_col, pred_col, classes
    )
  }
  
  # Create summary metrics table
  results$metrics <- .create_metrics_table(results, metrics)
  
  # Per-class breakdown
  results$per_class <- .create_per_class_breakdown(
    eval_data, truth_col, pred_col, classes
  )
  
  # Summary statistics
  results$summary <- .create_evaluation_summary(results, eval_data)
  
  # Plotting data if requested
  if (return_plot) {
    results$plot_data <- .prepare_plot_data(results, eval_data)
  }
  
  # Set class for method dispatch
  class(results) <- c("emotion_evaluation", "list")
  
  return(results)
}

#' Print method for emotion evaluation results
#' @param x An emotion_evaluation object
#' @param ... Additional arguments (unused)
#' @export
print.emotion_evaluation <- function(x, ...) {
  cat("Emotion Classification Evaluation Results\n")
  cat("========================================\n\n")
  
  if (!is.null(x$summary)) {
    cat("Summary:\n")
    cat(sprintf("  Total instances: %d\n", x$summary$n_instances))
    cat(sprintf("  Classes: %d (%s)\n", 
                x$summary$n_classes, 
                paste(x$summary$classes, collapse = ", ")))
    cat(sprintf("  Overall accuracy: %.3f\n", x$summary$accuracy))
    if (!is.null(x$f1_macro)) {
      cat(sprintf("  Macro F1: %.3f\n", x$f1_macro))
    }
    cat("\n")
  }
  
  if (!is.null(x$metrics)) {
    cat("Metrics:\n")
    print(x$metrics)
  }
  
  invisible(x)
}

# Helper functions for metrics computation
# ======================================

#' Validate evaluation input data
#' @noRd
.validate_evaluation_input <- function(data, id_col, truth_col, pred_col, 
                                     probs_cols, na_rm) {
  
  # Load data if file path provided
  if (is.character(data) && length(data) == 1) {
    if (!file.exists(data)) {
      stop("Data file not found: ", data, call. = FALSE)
    }
    data <- utils::read.csv(data, stringsAsFactors = FALSE)
  }
  
  # Validate data frame
  if (!is.data.frame(data)) {
    stop("Data must be a data frame or path to CSV file", call. = FALSE)
  }
  
  # Check required columns
  required_cols <- c(id_col, truth_col, pred_col)
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("Missing required columns: ", paste(missing_cols, collapse = ", "), 
         call. = FALSE)
  }
  
  # Check probability columns if provided
  if (!is.null(probs_cols)) {
    missing_probs <- setdiff(probs_cols, names(data))
    if (length(missing_probs) > 0) {
      warning("Missing probability columns: ", 
              paste(missing_probs, collapse = ", "), 
              call. = FALSE)
      probs_cols <- intersect(probs_cols, names(data))
    }
  }
  
  # Remove missing values if requested
  if (na_rm) {
    complete_cases <- complete.cases(data[, required_cols, drop = FALSE])
    if (sum(!complete_cases) > 0) {
      warning("Removed ", sum(!complete_cases), " rows with missing values", 
              call. = FALSE)
      data <- data[complete_cases, , drop = FALSE]
    }
  }
  
  # Check for empty data
  if (nrow(data) == 0) {
    stop("No valid data rows after preprocessing", call. = FALSE)
  }
  
  return(data)
}

#' Infer emotion classes from data
#' @noRd
.infer_emotion_classes <- function(data, truth_col, pred_col) {
  truth_classes <- unique(data[[truth_col]])
  pred_classes <- unique(data[[pred_col]])
  all_classes <- sort(unique(c(truth_classes, pred_classes)))
  return(all_classes)
}

#' Compute accuracy
#' @noRd
.compute_accuracy <- function(data, truth_col, pred_col) {
  correct <- data[[truth_col]] == data[[pred_col]]
  accuracy <- mean(correct, na.rm = TRUE)
  return(accuracy)
}

#' Compute classification metrics (precision, recall, F1)
#' @noRd
.compute_classification_metrics <- function(data, truth_col, pred_col, classes, metrics) {
  
  # Create confusion matrix
  truth <- factor(data[[truth_col]], levels = classes)
  pred <- factor(data[[pred_col]], levels = classes)
  
  # Compute per-class metrics
  per_class_metrics <- data.frame(
    class = classes,
    precision = numeric(length(classes)),
    recall = numeric(length(classes)),
    f1 = numeric(length(classes)),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(classes)) {
    class_name <- classes[i]
    
    # True positives, false positives, false negatives
    tp <- sum(truth == class_name & pred == class_name)
    fp <- sum(truth != class_name & pred == class_name)
    fn <- sum(truth == class_name & pred != class_name)
    
    # Precision
    precision <- if (tp + fp > 0) tp / (tp + fp) else 0
    per_class_metrics$precision[i] <- precision
    
    # Recall
    recall <- if (tp + fn > 0) tp / (tp + fn) else 0
    per_class_metrics$recall[i] <- recall
    
    # F1
    f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
    per_class_metrics$f1[i] <- f1
  }
  
  # Aggregate metrics
  results <- list(per_class_metrics = per_class_metrics)
  
  if ("precision" %in% metrics) {
    results$precision_macro <- mean(per_class_metrics$precision)
    results$precision_micro <- .compute_micro_precision(truth, pred, classes)
  }
  
  if ("recall" %in% metrics) {
    results$recall_macro <- mean(per_class_metrics$recall)
    results$recall_micro <- .compute_micro_recall(truth, pred, classes)
  }
  
  if ("f1_macro" %in% metrics) {
    results$f1_macro <- mean(per_class_metrics$f1)
  }
  
  if ("f1_micro" %in% metrics) {
    precision_micro <- results$precision_micro %||% .compute_micro_precision(truth, pred, classes)
    recall_micro <- results$recall_micro %||% .compute_micro_recall(truth, pred, classes)
    results$f1_micro <- if (precision_micro + recall_micro > 0) {
      2 * precision_micro * recall_micro / (precision_micro + recall_micro)
    } else 0
  }
  
  return(results)
}

#' Compute micro-averaged precision
#' @noRd
.compute_micro_precision <- function(truth, pred, classes) {
  tp_total <- sum(truth == pred)
  fp_total <- sum(truth != pred)
  
  if (tp_total + fp_total > 0) {
    return(tp_total / (tp_total + fp_total))
  } else {
    return(0)
  }
}

#' Compute micro-averaged recall  
#' @noRd
.compute_micro_recall <- function(truth, pred, classes) {
  # For multi-class, micro recall equals accuracy
  return(mean(truth == pred))
}

#' Compute confusion matrix
#' @noRd
.compute_confusion_matrix <- function(data, truth_col, pred_col, classes) {
  truth <- factor(data[[truth_col]], levels = classes)
  pred <- factor(data[[pred_col]], levels = classes)
  
  cm <- table(Predicted = pred, Actual = truth)
  
  # Convert to matrix and add row/column names
  cm_matrix <- as.matrix(cm)
  
  # Add marginals
  cm_with_margins <- addmargins(cm_matrix)
  
  return(cm_with_margins)
}

#' Compute AUROC for each class
#' @noRd
.compute_auroc <- function(data, truth_col, probs_cols, classes) {
  
  if (length(probs_cols) != length(classes)) {
    warning("Number of probability columns does not match number of classes", 
            call. = FALSE)
    return(NA)
  }
  
  auroc_results <- data.frame(
    class = classes,
    auroc = numeric(length(classes)),
    stringsAsFactors = FALSE
  )
  
  # Compute AUROC for each class
  for (i in seq_along(classes)) {
    class_name <- classes[i]
    prob_col <- probs_cols[i]
    
    if (prob_col %in% names(data)) {
      # Binary classification for this class vs all others
      binary_truth <- as.numeric(data[[truth_col]] == class_name)
      probs <- data[[prob_col]]
      
      # Remove missing values
      valid_idx <- !is.na(binary_truth) & !is.na(probs)
      binary_truth <- binary_truth[valid_idx]
      probs <- probs[valid_idx]
      
      if (length(unique(binary_truth)) > 1) {
        auroc_results$auroc[i] <- .compute_binary_auroc(binary_truth, probs)
      } else {
        auroc_results$auroc[i] <- NA
      }
    } else {
      auroc_results$auroc[i] <- NA
    }
  }
  
  # Macro-averaged AUROC
  macro_auroc <- mean(auroc_results$auroc, na.rm = TRUE)
  
  return(list(
    per_class = auroc_results,
    macro = macro_auroc
  ))
}

#' Compute binary AUROC using trapezoidal rule
#' @noRd
.compute_binary_auroc <- function(y_true, y_scores) {
  # Sort by scores in descending order
  order_idx <- order(y_scores, decreasing = TRUE)
  y_true_sorted <- y_true[order_idx]
  
  # Compute TPR and FPR at each threshold
  n_pos <- sum(y_true)
  n_neg <- length(y_true) - n_pos
  
  if (n_pos == 0 || n_neg == 0) {
    return(NA)
  }
  
  tp <- cumsum(y_true_sorted)
  fp <- cumsum(1 - y_true_sorted)
  
  tpr <- tp / n_pos
  fpr <- fp / n_neg
  
  # Add origin point
  tpr <- c(0, tpr)
  fpr <- c(0, fpr)
  
  # Compute AUC using trapezoidal rule
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  
  return(auc)
}

#' Compute Expected Calibration Error (ECE)
#' @noRd
.compute_ece <- function(data, truth_col, pred_col, probs_cols, n_bins = 10) {
  
  # Get predicted probabilities and actual predictions
  max_probs <- apply(data[, probs_cols, drop = FALSE], 1, max, na.rm = TRUE)
  predicted_classes <- data[[pred_col]]
  actual_classes <- data[[truth_col]]
  
  # Check if predictions match max probability class
  prob_class_indices <- apply(data[, probs_cols, drop = FALSE], 1, which.max)
  prob_classes <- probs_cols[prob_class_indices]
  
  # Create bins
  bin_boundaries <- seq(0, 1, length.out = n_bins + 1)
  bin_lowers <- bin_boundaries[-length(bin_boundaries)]
  bin_uppers <- bin_boundaries[-1]
  
  ece <- 0
  total_samples <- length(max_probs)
  
  for (i in seq_along(bin_lowers)) {
    # Find samples in this bin
    in_bin <- max_probs > bin_lowers[i] & max_probs <= bin_uppers[i]
    
    if (sum(in_bin) > 0) {
      # Compute accuracy and confidence for this bin
      bin_accuracy <- mean(predicted_classes[in_bin] == actual_classes[in_bin])
      bin_confidence <- mean(max_probs[in_bin])
      bin_size <- sum(in_bin)
      
      # Add weighted difference to ECE
      ece <- ece + (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)
    }
  }
  
  return(ece)
}

#' Compute Krippendorff's Alpha for inter-rater reliability
#' @noRd
.compute_krippendorff_alpha <- function(data, truth_col, pred_col, classes) {
  
  # Create agreement matrix (2 raters: human vs model)
  truth <- data[[truth_col]]
  pred <- data[[pred_col]]
  
  # Convert to numeric codes
  class_to_num <- setNames(seq_along(classes), classes)
  truth_num <- class_to_num[truth]
  pred_num <- class_to_num[pred]
  
  # Remove missing values
  valid_idx <- !is.na(truth_num) & !is.na(pred_num)
  truth_num <- truth_num[valid_idx]
  pred_num <- pred_num[valid_idx]
  
  if (length(truth_num) < 2) {
    return(NA)
  }
  
  # Compute observed and expected disagreement
  n <- length(truth_num)
  
  # Observed disagreement
  observed_disagreement <- mean(truth_num != pred_num)
  
  # Expected disagreement (marginal distributions)
  truth_counts <- table(truth_num)
  pred_counts <- table(pred_num)
  total_counts <- truth_counts + pred_counts[names(truth_counts)]
  
  # Expected disagreement under independence
  expected_disagreement <- 1 - sum((total_counts / (2 * n))^2)
  
  # Krippendorff's alpha
  if (expected_disagreement == 0) {
    return(1)  # Perfect agreement
  } else {
    alpha <- 1 - (observed_disagreement / expected_disagreement)
    return(alpha)
  }
}

#' Create metrics summary table
#' @noRd
.create_metrics_table <- function(results, metrics) {
  
  metrics_df <- data.frame(
    metric = character(),
    value = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Add computed metrics to table
  if ("accuracy" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "accuracy", value = results$accuracy))
  }
  
  if ("precision_macro" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "precision_macro", value = results$precision_macro))
  }
  
  if ("recall_macro" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "recall_macro", value = results$recall_macro))
  }
  
  if ("f1_macro" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "f1_macro", value = results$f1_macro))
  }
  
  if ("f1_micro" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "f1_micro", value = results$f1_micro))
  }
  
  if ("auroc" %in% names(results) && !is.na(results$auroc$macro)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "auroc_macro", value = results$auroc$macro))
  }
  
  if ("ece" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "ece", value = results$ece))
  }
  
  if ("krippendorff_alpha" %in% names(results)) {
    metrics_df <- rbind(metrics_df, 
                       data.frame(metric = "krippendorff_alpha", value = results$krippendorff_alpha))
  }
  
  return(metrics_df)
}

#' Create per-class breakdown
#' @noRd
.create_per_class_breakdown <- function(data, truth_col, pred_col, classes) {
  
  if ("per_class_metrics" %in% names(data)) {
    return(data$per_class_metrics)
  }
  
  # Compute basic per-class metrics
  truth <- factor(data[[truth_col]], levels = classes)
  pred <- factor(data[[pred_col]], levels = classes)
  
  per_class <- data.frame(
    class = classes,
    support = as.numeric(table(truth)[classes]),
    stringsAsFactors = FALSE
  )
  
  return(per_class)
}

#' Create evaluation summary
#' @noRd
.create_evaluation_summary <- function(results, data) {
  
  summary_list <- list(
    n_instances = nrow(data),
    n_classes = length(unique(c(data[[2]], data[[3]]))),  # truth and pred cols
    classes = sort(unique(c(data[[2]], data[[3]]))),
    accuracy = results$accuracy %||% NA
  )
  
  return(summary_list)
}

#' Prepare data for plotting
#' @noRd
.prepare_plot_data <- function(results, data) {
  
  plot_data <- list()
  
  # Confusion matrix data for heatmap
  if (!is.null(results$confusion_matrix)) {
    cm <- results$confusion_matrix
    if (!is.null(dim(cm)) && nrow(cm) > 1 && ncol(cm) > 1) {
      # Remove marginal sums for plotting
      cm_clean <- cm[-nrow(cm), -ncol(cm)]
      
      # Convert to long format
      cm_long <- expand.grid(
        Predicted = rownames(cm_clean),
        Actual = colnames(cm_clean),
        stringsAsFactors = FALSE
      )
      cm_long$Count <- as.vector(cm_clean)
      
      plot_data$confusion_matrix <- cm_long
    }
  }
  
  # Per-class metrics for bar plot
  if (!is.null(results$per_class_metrics)) {
    plot_data$per_class_metrics <- results$per_class_metrics
  }
  
  return(plot_data)
}

# Utility operator for NULL coalescing
`%||%` <- function(x, y) if (is.null(x)) y else x

#' Plot Evaluation Results
#'
#' @description
#' Creates visualizations for emotion evaluation results including confusion
#' matrix heatmaps and per-class metrics bar plots.
#'
#' @param x An emotion_evaluation object from evaluate_emotions()
#' @param type Character. Type of plot: "confusion_matrix", "metrics", or "both"
#' @param ... Additional arguments passed to plotting functions
#'
#' @return A ggplot object or list of ggplot objects
#' @export
plot.emotion_evaluation <- function(x, type = "both", ...) {
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required for plotting. Please install it.", 
         call. = FALSE)
  }
  
  plots <- list()
  
  # Confusion matrix heatmap
  if (type %in% c("confusion_matrix", "both") && !is.null(x$plot_data$confusion_matrix)) {
    cm_data <- x$plot_data$confusion_matrix
    
    plots$confusion_matrix <- ggplot2::ggplot(cm_data, ggplot2::aes(x = Actual, y = Predicted, fill = Count)) +
      ggplot2::geom_tile(color = "white") +
      ggplot2::geom_text(ggplot2::aes(label = Count), vjust = 0.5) +
      ggplot2::scale_fill_gradient(low = "white", high = "steelblue") +
      ggplot2::labs(
        title = "Confusion Matrix",
        x = "Actual Class",
        y = "Predicted Class"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        plot.title = ggplot2::element_text(hjust = 0.5)
      )
  }
  
  # Per-class metrics bar plot
  if (type %in% c("metrics", "both") && !is.null(x$per_class_metrics)) {
    metrics_long <- reshape2::melt(
      x$per_class_metrics[, c("class", "precision", "recall", "f1")],
      id.vars = "class",
      variable.name = "metric",
      value.name = "value"
    )
    
    plots$metrics <- ggplot2::ggplot(metrics_long, ggplot2::aes(x = class, y = value, fill = metric)) +
      ggplot2::geom_bar(stat = "identity", position = "dodge") +
      ggplot2::scale_y_continuous(limits = c(0, 1)) +
      ggplot2::labs(
        title = "Per-Class Metrics",
        x = "Emotion Class",
        y = "Metric Value",
        fill = "Metric"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
        plot.title = ggplot2::element_text(hjust = 0.5)
      )
  }
  
  # Return single plot or list
  if (length(plots) == 1) {
    return(plots[[1]])
  } else if (length(plots) > 1) {
    return(plots)
  } else {
    warning("No plots could be generated with available data", call. = FALSE)
    return(NULL)
  }
}

#' Summary method for emotion evaluation results
#' @param object An emotion_evaluation object
#' @param ... Additional arguments (unused)
#' @export
summary.emotion_evaluation <- function(object, ...) {
  
  cat("Emotion Classification Evaluation Summary\n")
  cat("=======================================\n\n")
  
  # Dataset summary
  if (!is.null(object$summary)) {
    cat("Dataset Information:\n")
    cat(sprintf("  • Total instances: %d\n", object$summary$n_instances))
    cat(sprintf("  • Number of classes: %d\n", object$summary$n_classes))
    cat(sprintf("  • Classes: %s\n", paste(object$summary$classes, collapse = ", ")))
    cat("\n")
  }
  
  # Overall performance
  cat("Overall Performance:\n")
  if (!is.null(object$accuracy)) {
    cat(sprintf("  • Accuracy: %.3f\n", object$accuracy))
  }
  if (!is.null(object$f1_macro)) {
    cat(sprintf("  • Macro F1: %.3f\n", object$f1_macro))
  }
  if (!is.null(object$f1_micro)) {
    cat(sprintf("  • Micro F1: %.3f\n", object$f1_micro))
  }
  if (!is.null(object$auroc) && !is.na(object$auroc$macro)) {
    cat(sprintf("  • Macro AUROC: %.3f\n", object$auroc$macro))
  }
  if (!is.null(object$ece)) {
    cat(sprintf("  • Expected Calibration Error: %.3f\n", object$ece))
  }
  if (!is.null(object$krippendorff_alpha)) {
    cat(sprintf("  • Krippendorff's α: %.3f\n", object$krippendorff_alpha))
  }
  cat("\n")
  
  # Per-class breakdown
  if (!is.null(object$per_class_metrics)) {
    cat("Per-Class Metrics:\n")
    print(object$per_class_metrics, row.names = FALSE)
    cat("\n")
  }
  
  # Confusion matrix summary
  if (!is.null(object$confusion_matrix)) {
    cat("Confusion Matrix:\n")
    print(object$confusion_matrix)
  }
  
  invisible(object)
}