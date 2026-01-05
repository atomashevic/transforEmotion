## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.height = 5
)

## ----eval=FALSE---------------------------------------------------------------
# # Install transforEmotion if not already installed
# # devtools::install_github("your-repo/transforEmotion")
# library(transforEmotion)

## -----------------------------------------------------------------------------
library(transforEmotion)

## -----------------------------------------------------------------------------
# Create synthetic evaluation data
set.seed(42)
n_samples <- 200

# Generate ground truth labels
emotions <- c("anger", "joy", "sadness", "fear", "surprise")
eval_data <- data.frame(
  id = 1:n_samples,
  truth = sample(emotions, n_samples, replace = TRUE, 
                prob = c(0.2, 0.3, 0.2, 0.15, 0.15)),
  stringsAsFactors = FALSE
)

# Generate realistic predictions (correlated with truth but with some errors)
eval_data$pred <- eval_data$truth
# Introduce some classification errors
error_indices <- sample(1:n_samples, size = 0.25 * n_samples)
eval_data$pred[error_indices] <- sample(emotions, length(error_indices), replace = TRUE)

# Generate probability scores
for (emotion in emotions) {
  # Higher probability for correct class, lower for others
  eval_data[[paste0("prob_", emotion)]] <- ifelse(
    eval_data$truth == emotion,
    runif(n_samples, 0.6, 0.95),  # Higher prob for correct class
    runif(n_samples, 0.01, 0.4)   # Lower prob for incorrect classes
  )
}

# Normalize probabilities to sum to 1
prob_cols <- paste0("prob_", emotions)
prob_sums <- rowSums(eval_data[, prob_cols])
eval_data[, prob_cols] <- eval_data[, prob_cols] / prob_sums

# Display sample data
head(eval_data)

## -----------------------------------------------------------------------------
# Basic evaluation with default metrics
results <- evaluate_emotions(
  data = eval_data,
  truth_col = "truth",
  pred_col = "pred"
)

# Print results
print(results)

## -----------------------------------------------------------------------------
# Full evaluation with probability scores
results_full <- evaluate_emotions(
  data = eval_data,
  truth_col = "truth",
  pred_col = "pred",
  probs_cols = prob_cols,
  classes = emotions,
  return_plot = TRUE
)

# Display summary
summary(results_full)

## -----------------------------------------------------------------------------
# Access per-class metrics
results_full$per_class_metrics

## -----------------------------------------------------------------------------
# AUROC results
results_full$auroc

# Calibration error
cat("Expected Calibration Error:", round(results_full$ece, 3))

## -----------------------------------------------------------------------------
cat("Krippendorff's α:", round(results_full$krippendorff_alpha, 3))

## ----eval=FALSE---------------------------------------------------------------
# # Plot confusion matrix and metrics (requires ggplot2)
# if (requireNamespace("ggplot2", quietly = TRUE)) {
#   plots <- plot(results_full)
# 
#   # Display confusion matrix
#   print(plots$confusion_matrix)
# 
#   # Display per-class metrics
#   print(plots$metrics)
# }

## ----eval=FALSE---------------------------------------------------------------
# # Step 1: Get emotion predictions using transforEmotion
# text_data <- c(
#   "I am so happy today!",
#   "This makes me really angry.",
#   "I feel very sad about this news."
# )
# 
# # Get transformer-based predictions
# predictions <- transformer_scores(
#   x = text_data,
#   classes = c("anger", "joy", "sadness"),
#   return_prob = TRUE
# )
# 
# # Step 2: Prepare evaluation data (assuming you have ground truth)
# ground_truth <- c("joy", "anger", "sadness")  # Your ground truth labels
# 
# eval_df <- data.frame(
#   id = 1:length(text_data),
#   truth = ground_truth,
#   pred = predictions$predicted_class,
#   prob_anger = predictions$prob_anger,
#   prob_joy = predictions$prob_joy,
#   prob_sadness = predictions$prob_sadness,
#   stringsAsFactors = FALSE
# )
# 
# # Step 3: Evaluate performance
# evaluation <- evaluate_emotions(
#   data = eval_df,
#   probs_cols = c("prob_anger", "prob_joy", "prob_sadness")
# )
# 
# print(evaluation)

## ----eval=FALSE---------------------------------------------------------------
# # Save evaluation data to CSV
# write.csv(eval_data, "model_evaluation.csv", row.names = FALSE)
# 
# # Load and evaluate from CSV
# csv_results <- evaluate_emotions(
#   data = "model_evaluation.csv",
#   probs_cols = prob_cols
# )

## -----------------------------------------------------------------------------
# Evaluate only accuracy and F1 scores
quick_eval <- evaluate_emotions(
  data = eval_data,
  metrics = c("accuracy", "f1_macro", "f1_micro"),
  return_plot = FALSE
)

print(quick_eval$metrics)

## -----------------------------------------------------------------------------
# Create data with missing values
eval_data_missing <- eval_data
eval_data_missing$truth[1:5] <- NA
eval_data_missing$pred[6:10] <- NA

# Evaluate with automatic missing value removal
results_clean <- evaluate_emotions(
  data = eval_data_missing,
  na_rm = TRUE  # Default behavior
)

cat("Original samples:", nrow(eval_data_missing), "\n")
cat("Samples after cleaning:", results_clean$summary$n_instances, "\n")

## -----------------------------------------------------------------------------
# Rename columns in your data
custom_data <- eval_data
names(custom_data)[names(custom_data) == "truth"] <- "ground_truth"
names(custom_data)[names(custom_data) == "pred"] <- "model_prediction"

# Evaluate with custom column names
custom_results <- evaluate_emotions(
  data = custom_data,
  truth_col = "ground_truth",
  pred_col = "model_prediction",
  metrics = c("accuracy", "f1_macro")
)

print(custom_results)

## ----eval=FALSE---------------------------------------------------------------
# # Good: Include probabilities for calibration analysis
# results_with_probs <- evaluate_emotions(
#   data = eval_data,
#   probs_cols = prob_cols
# )

## -----------------------------------------------------------------------------
# Check class distribution
table(eval_data$truth)
table(eval_data$pred)

# Check for missing values
sum(is.na(eval_data$truth))
sum(is.na(eval_data$pred))

## -----------------------------------------------------------------------------
# Get comprehensive evaluation
comprehensive_eval <- evaluate_emotions(
  data = eval_data,
  probs_cols = prob_cols,
  metrics = c("accuracy", "precision", "recall", "f1_macro", "f1_micro", 
             "auroc", "ece", "krippendorff", "confusion_matrix")
)

# Report key metrics
key_metrics <- comprehensive_eval$metrics[
  comprehensive_eval$metrics$metric %in% c("accuracy", "f1_macro", "f1_micro"),
]
print(key_metrics)

