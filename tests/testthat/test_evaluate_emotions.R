test_that("evaluate_emotions basic functionality works", {
  
  # Create synthetic test data
  test_data <- data.frame(
    id = 1:100,
    truth = sample(c("anger", "joy", "sadness"), 100, replace = TRUE),
    pred = sample(c("anger", "joy", "sadness"), 100, replace = TRUE),
    prob_anger = runif(100, 0, 1),
    prob_joy = runif(100, 0, 1),  
    prob_sadness = runif(100, 0, 1),
    stringsAsFactors = FALSE
  )
  
  # Normalize probabilities to sum to 1
  prob_cols <- c("prob_anger", "prob_joy", "prob_sadness")
  prob_sums <- rowSums(test_data[, prob_cols])
  test_data[, prob_cols] <- test_data[, prob_cols] / prob_sums
  
  # Test basic evaluation
  results <- evaluate_emotions(
    data = test_data,
    metrics = c("accuracy", "f1_macro", "confusion_matrix")
  )
  
  expect_type(results, "list")
  expect_s3_class(results, "emotion_evaluation") 
  expect_true("accuracy" %in% names(results))
  expect_true("f1_macro" %in% names(results))
  expect_true("confusion_matrix" %in% names(results))
  expect_true("metrics" %in% names(results))
  expect_true("summary" %in% names(results))
  
  # Check metric values are reasonable
  expect_gte(results$accuracy, 0)
  expect_lte(results$accuracy, 1)
  expect_gte(results$f1_macro, 0)
  expect_lte(results$f1_macro, 1)
})

test_that("evaluate_emotions works with probabilities", {
  
  # Create test data with probabilities
  test_data <- data.frame(
    id = 1:50,
    truth = rep(c("anger", "joy"), 25),
    pred = rep(c("anger", "joy"), 25),
    prob_anger = c(rep(0.8, 25), rep(0.2, 25)),
    prob_joy = c(rep(0.2, 25), rep(0.8, 25)),
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(
    data = test_data,
    probs_cols = c("prob_anger", "prob_joy"),
    metrics = c("accuracy", "auroc", "ece")
  )
  
  expect_true("auroc" %in% names(results))
  expect_true("ece" %in% names(results))
  expect_type(results$auroc, "list")
  expect_true("per_class" %in% names(results$auroc))
  expect_true("macro" %in% names(results$auroc))
  expect_gte(results$ece, 0)
  expect_lte(results$ece, 1)
})

test_that("evaluate_emotions input validation works", {
  
  # Test missing columns
  bad_data <- data.frame(
    id = 1:10,
    truth = rep("anger", 10)
    # Missing pred column
  )
  
  expect_error(
    evaluate_emotions(bad_data),
    "Missing required columns"
  )
  
  # Test non-data.frame input
  expect_error(
    evaluate_emotions("not_a_dataframe"),
    "Data file not found"
  )
  
  # Test empty data
  empty_data <- data.frame(
    id = integer(0),
    truth = character(0),
    pred = character(0)
  )
  
  expect_error(
    evaluate_emotions(empty_data),
    "No valid data rows"
  )
})

test_that("evaluate_emotions handles missing values", {
  
  # Create data with missing values
  test_data <- data.frame(
    id = 1:20,
    truth = c(rep("anger", 5), rep("joy", 5), rep(NA, 5), rep("sadness", 5)),
    pred = c(rep("anger", 5), rep("joy", 5), rep("sadness", 5), rep(NA, 5)),
    stringsAsFactors = FALSE
  )
  
  # Test with na_rm = TRUE (default)
  expect_warning(
    results <- evaluate_emotions(test_data, na_rm = TRUE),
    "Removed .* rows with missing values"
  )
  
  expect_equal(results$summary$n_instances, 10)  # Only complete cases
  
  # Test with na_rm = FALSE
  expect_error(
    evaluate_emotions(test_data, na_rm = FALSE),
    # Should fail due to NA values in computations
    class = "error"
  )
})

test_that("evaluate_emotions perfect classification", {
  
  # Create perfect classification data
  test_data <- data.frame(
    id = 1:30,
    truth = rep(c("anger", "joy", "sadness"), 10),
    pred = rep(c("anger", "joy", "sadness"), 10),  # Perfect match
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(test_data)
  
  expect_equal(results$accuracy, 1.0)
  expect_equal(results$f1_macro, 1.0)
  expect_equal(results$f1_micro, 1.0)
  
  # Check confusion matrix is diagonal
  cm <- results$confusion_matrix
  diag_sum <- sum(diag(cm[-nrow(cm), -ncol(cm)]))  # Remove margin sums
  total_sum <- cm[nrow(cm), ncol(cm)]  # Total from margins
  expect_equal(diag_sum, total_sum)
})

test_that("evaluate_emotions custom column names", {
  
  test_data <- data.frame(
    instance_id = 1:20,
    ground_truth = sample(c("happy", "sad"), 20, replace = TRUE),
    model_pred = sample(c("happy", "sad"), 20, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(
    data = test_data,
    id_col = "instance_id",
    truth_col = "ground_truth", 
    pred_col = "model_pred"
  )
  
  expect_type(results, "list")
  expect_s3_class(results, "emotion_evaluation")
  expect_true(all(c("happy", "sad") %in% results$summary$classes))
})

test_that("evaluate_emotions single class edge case", {
  
  # Test with only one class in truth (edge case)
  test_data <- data.frame(
    id = 1:10,
    truth = rep("anger", 10),
    pred = sample(c("anger", "joy"), 10, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(test_data)
  
  expect_type(results, "list")
  expect_s3_class(results, "emotion_evaluation")
  # Some metrics might be NA or 0 for single class
  expect_true(is.numeric(results$accuracy))
})

test_that("evaluate_emotions Krippendorff's alpha", {
  
  # Create data for inter-rater reliability testing
  test_data <- data.frame(
    id = 1:40,
    truth = sample(c("anger", "joy", "sadness", "fear"), 40, replace = TRUE),
    pred = sample(c("anger", "joy", "sadness", "fear"), 40, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(
    data = test_data,
    metrics = "krippendorff"
  )
  
  expect_true("krippendorff_alpha" %in% names(results))
  expect_type(results$krippendorff_alpha, "double")
  
  # Alpha should be between -1 and 1 (though negative values are rare)
  expect_gte(results$krippendorff_alpha, -1)
  expect_lte(results$krippendorff_alpha, 1)
})

test_that("evaluate_emotions plotting data", {
  
  test_data <- data.frame(
    id = 1:30,
    truth = sample(c("anger", "joy", "sadness"), 30, replace = TRUE),
    pred = sample(c("anger", "joy", "sadness"), 30, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(
    data = test_data,
    return_plot = TRUE
  )
  
  expect_true("plot_data" %in% names(results))
  expect_type(results$plot_data, "list")
  
  if (!is.null(results$plot_data$confusion_matrix)) {
    plot_cm <- results$plot_data$confusion_matrix
    expect_true(all(c("Predicted", "Actual", "Count") %in% names(plot_cm)))
  }
})

test_that("evaluate_emotions print method", {
  
  test_data <- data.frame(
    id = 1:20,
    truth = sample(c("anger", "joy"), 20, replace = TRUE),
    pred = sample(c("anger", "joy"), 20, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  results <- evaluate_emotions(test_data)
  
  # Test that print method works without error
  expect_output(print(results), "Emotion Classification Evaluation Results")
  expect_output(print(results), "Summary:")
  expect_output(print(results), "Total instances:")
})

test_that("evaluate_emotions CSV file input", {
  
  skip_on_cran()
  
  # Create temporary CSV file
  temp_file <- tempfile(fileext = ".csv")
  
  test_data <- data.frame(
    id = 1:15,
    truth = sample(c("anger", "joy"), 15, replace = TRUE),
    pred = sample(c("anger", "joy"), 15, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  write.csv(test_data, temp_file, row.names = FALSE)
  
  # Test reading from CSV
  results <- evaluate_emotions(temp_file)
  
  expect_type(results, "list")
  expect_s3_class(results, "emotion_evaluation")
  expect_equal(results$summary$n_instances, 15)
  
  # Clean up
  unlink(temp_file)
})

test_that("evaluate_emotions metrics selection", {
  
  test_data <- data.frame(
    id = 1:25,
    truth = sample(c("anger", "joy", "sadness"), 25, replace = TRUE),
    pred = sample(c("anger", "joy", "sadness"), 25, replace = TRUE),
    stringsAsFactors = FALSE
  )
  
  # Test selecting only specific metrics
  results <- evaluate_emotions(
    data = test_data,
    metrics = c("accuracy", "f1_macro")
  )
  
  expect_true("accuracy" %in% names(results))
  expect_true("f1_macro" %in% names(results))
  expect_false("precision_macro" %in% names(results))
  expect_false("auroc" %in% names(results))
  
  # Check metrics table only contains requested metrics
  metrics_table <- results$metrics
  expect_true(all(metrics_table$metric %in% c("accuracy", "f1_macro")))
})

test_that("evaluate_emotions with package data", {
  
  skip_if_not_installed("transforEmotion")
  skip_on_cran()
  
  # Test with package datasets if available
  tryCatch({
    data("emotions", package = "transforEmotion", envir = environment())
    
    if (exists("emotions") && is.data.frame(emotions)) {
      # Create mock predictions for testing
      if (nrow(emotions) > 0) {
        test_emotions <- emotions[1:min(50, nrow(emotions)), ]
        test_emotions$pred <- sample(test_emotions$emotion, nrow(test_emotions))
        test_emotions$id <- seq_len(nrow(test_emotions))
        
        results <- evaluate_emotions(
          data = test_emotions,
          truth_col = "emotion",
          pred_col = "pred"
        )
        
        expect_type(results, "list")
        expect_s3_class(results, "emotion_evaluation")
      }
    }
  }, error = function(e) {
    skip("Package data not available")
  })
})