skip_on_cran()

test_that("download_findingemo_data input validation works", {
  
  # Test missing target_dir
  expect_error(
    download_findingemo_data(),
    "argument.*target_dir.*is missing"
  )
  
  # Test invalid target_dir
  expect_error(
    download_findingemo_data(c("dir1", "dir2")),
    "target_dir must be a single character string"
  )
  
  expect_error(
    download_findingemo_data(123),
    "target_dir must be a single character string"
  )
  
  # Test invalid max_images
  expect_error(
    download_findingemo_data("./test", max_images = -1),
    "max_images must be a positive integer"
  )
  
  expect_error(
    download_findingemo_data("./test", max_images = "invalid"),
    "max_images must be a positive integer"
  )
  
  # Test invalid randomize
  expect_error(
    download_findingemo_data("./test", randomize = "invalid"),
    "randomize must be a single logical value"
  )
  
  expect_error(
    download_findingemo_data("./test", randomize = c(TRUE, FALSE)),
    "randomize must be a single logical value"
  )
})

test_that("download_findingemo_data finds Python script", {
  
  skip_on_cran()
  
  # Check that Python script exists
  script_path <- system.file("python", "download_findingemo.py", 
                            package = "transforEmotion")
  expect_true(file.exists(script_path))
  
  # Test with non-existent directory (should create it)
  temp_dir <- tempfile()
  
  # Mock the script execution to avoid actual download
  with_mocked_bindings(
    system2 = function(...) 0,  # Mock successful execution
    `jsonlite::fromJSON` = function(...) list(
      success = TRUE,
      message = "Mock download success",
      target_dir = temp_dir
    ),
    {
      result <- download_findingemo_data(temp_dir)
      expect_type(result, "list")
      expect_true("success" %in% names(result))
      expect_true("message" %in% names(result))
    }
  )
})

test_that("load_findingemo_annotations input validation works", {
  
  # Test missing data_dir
  expect_error(
    load_findingemo_annotations(),
    "argument.*data_dir.*is missing"
  )
  
  # Test invalid data_dir
  expect_error(
    load_findingemo_annotations(c("dir1", "dir2")),
    "data_dir must be a single character string"
  )
  
  expect_error(
    load_findingemo_annotations(123),
    "data_dir must be a single character string"
  )
  
  # Test non-existent directory
  expect_error(
    load_findingemo_annotations("/non/existent/directory"),
    "Data directory does not exist"
  )
  
  # Test invalid output_format
  expect_error(
    load_findingemo_annotations(tempdir(), output_format = "invalid"),
    "'arg' should be one of"
  )
})

test_that("load_findingemo_annotations finds Python script", {
  
  skip_on_cran()
  
  # Check that Python script exists
  script_path <- system.file("python", "load_findingemo_annotations.py", 
                            package = "transforEmotion")
  expect_true(file.exists(script_path))
})

test_that("prepare_findingemo_evaluation input validation works", {
  
  # Test invalid inputs
  expect_error(
    prepare_findingemo_evaluation("not_a_dataframe", data.frame()),
    "annotations must be a data.frame"
  )
  
  expect_error(
    prepare_findingemo_evaluation(data.frame(), "not_a_dataframe"),
    "predictions must be a data.frame"
  )
  
  # Test missing columns
  annotations <- data.frame(other_col = 1:5)
  predictions <- data.frame(image_id = 1:5, predicted_emotion = letters[1:5])
  
  expect_error(
    prepare_findingemo_evaluation(annotations, predictions),
    "ID column.*not found in annotations"
  )
  
  annotations <- data.frame(image_id = 1:5, emotion_label = letters[1:5])
  predictions <- data.frame(other_col = 1:5)
  
  expect_error(
    prepare_findingemo_evaluation(annotations, predictions),
    "ID column.*not found in predictions"
  )
  
  predictions <- data.frame(image_id = 1:5, other_col = letters[1:5])
  
  expect_error(
    prepare_findingemo_evaluation(annotations, predictions),
    "Prediction column.*not found in predictions"
  )
})

test_that("prepare_findingemo_evaluation works with valid data", {
  
  # Create sample data
  annotations <- data.frame(
    image_id = 1:10,
    emotion_label = sample(c("happy", "sad", "angry"), 10, replace = TRUE),
    valence = runif(10, -1, 1),
    arousal = runif(10, -1, 1),
    stringsAsFactors = FALSE
  )
  
  predictions <- data.frame(
    image_id = 1:8,  # Partial overlap to test merging
    predicted_emotion = sample(c("happy", "sad", "angry"), 8, replace = TRUE),
    prob_happy = runif(8),
    prob_sad = runif(8),
    prob_angry = runif(8),
    stringsAsFactors = FALSE
  )
  
  # Test basic functionality
  expect_message(
    result <- prepare_findingemo_evaluation(annotations, predictions),
    "Prepared evaluation data"
  )
  
  expect_s3_class(result, "data.frame")
  expect_true("id" %in% names(result))
  expect_true("truth" %in% names(result))
  expect_true("pred" %in% names(result))
  expect_equal(nrow(result), 8)  # Should match predictions (inner join)
  
  # Check probability columns are preserved
  prob_cols <- grep("^prob_", names(result), value = TRUE)
  expect_length(prob_cols, 3)
  
  # Test with include_va = FALSE
  result_no_va <- prepare_findingemo_evaluation(
    annotations, predictions, include_va = FALSE
  )
  expect_false("valence" %in% names(result_no_va))
  expect_false("arousal" %in% names(result_no_va))
})

test_that("prepare_findingemo_evaluation handles missing truth column", {
  
  # Annotations without standard emotion_label column
  annotations <- data.frame(
    image_id = 1:5,
    some_emotion_col = letters[1:5],
    valence = runif(5),
    stringsAsFactors = FALSE
  )
  
  predictions <- data.frame(
    image_id = 1:5,
    predicted_emotion = letters[1:5],
    stringsAsFactors = FALSE
  )
  
  # Should find and use the emotion column
  expect_message(
    result <- prepare_findingemo_evaluation(annotations, predictions),
    "Using.*as truth column"
  )
  
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 5)
})

test_that("prepare_findingemo_evaluation handles missing values", {
  
  # Create data with missing values
  annotations <- data.frame(
    image_id = 1:6,
    emotion_label = c("happy", "sad", NA, "angry", "happy", "sad"),
    valence = c(0.5, -0.5, 0.2, -0.8, NA, 0.3),
    stringsAsFactors = FALSE
  )
  
  predictions <- data.frame(
    image_id = 1:6,
    predicted_emotion = c("happy", NA, "angry", "angry", "happy", "sad"),
    prob_happy = runif(6),
    stringsAsFactors = FALSE
  )
  
  # Should remove rows with missing truth/prediction
  expect_message(
    result <- prepare_findingemo_evaluation(annotations, predictions),
    "Removed.*rows with missing"
  )
  
  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 4)  # Should remove rows 2 and 3
  expect_true(all(!is.na(result$truth)))
  expect_true(all(!is.na(result$pred)))
})

test_that("prepare_findingemo_evaluation handles no matching records", {
  
  annotations <- data.frame(
    image_id = 1:5,
    emotion_label = letters[1:5],
    stringsAsFactors = FALSE
  )
  
  predictions <- data.frame(
    image_id = 6:10,  # No overlap
    predicted_emotion = letters[6:10],
    stringsAsFactors = FALSE
  )
  
  expect_error(
    prepare_findingemo_evaluation(annotations, predictions),
    "No matching records found"
  )
})

test_that("prepare_findingemo_evaluation custom column names work", {
  
  annotations <- data.frame(
    img_id = 1:5,
    ground_truth = letters[1:5],
    valence = runif(5),
    stringsAsFactors = FALSE
  )
  
  predictions <- data.frame(
    img_id = 1:5,
    model_pred = letters[1:5],
    stringsAsFactors = FALSE
  )
  
  result <- prepare_findingemo_evaluation(
    annotations = annotations,
    predictions = predictions,
    id_col = "img_id",
    truth_col = "ground_truth",
    pred_col = "model_pred"
  )
  
  expect_s3_class(result, "data.frame")
  expect_true("id" %in% names(result))
  expect_true("truth" %in% names(result))
  expect_true("pred" %in% names(result))
  expect_equal(nrow(result), 5)
})

test_that("FindingEmo integration with evaluate_emotions works", {
  
  skip_on_cran()
  skip_if_not_installed("transforEmotion")
  
  # Create synthetic FindingEmo-style data
  annotations <- data.frame(
    image_id = 1:50,
    emotion_label = sample(c("happy", "sad", "angry", "neutral"), 50, replace = TRUE),
    valence = runif(50, -1, 1),
    arousal = runif(50, -1, 1),
    stringsAsFactors = FALSE
  )
  
  # Create realistic predictions with some correlation to truth
  predictions <- data.frame(
    image_id = 1:50,
    predicted_emotion = annotations$emotion_label,  # Start with truth
    prob_happy = ifelse(annotations$emotion_label == "happy", 
                       runif(50, 0.6, 0.9), runif(50, 0.1, 0.4)),
    prob_sad = ifelse(annotations$emotion_label == "sad", 
                     runif(50, 0.6, 0.9), runif(50, 0.1, 0.4)),
    prob_angry = ifelse(annotations$emotion_label == "angry", 
                       runif(50, 0.6, 0.9), runif(50, 0.1, 0.4)),
    prob_neutral = ifelse(annotations$emotion_label == "neutral", 
                         runif(50, 0.6, 0.9), runif(50, 0.1, 0.4)),
    stringsAsFactors = FALSE
  )
  
  # Add some prediction errors
  error_indices <- sample(1:50, 10)
  predictions$predicted_emotion[error_indices] <- sample(
    c("happy", "sad", "angry", "neutral"), 10, replace = TRUE
  )
  
  # Normalize probabilities
  prob_cols <- c("prob_happy", "prob_sad", "prob_angry", "prob_neutral")
  prob_sums <- rowSums(predictions[, prob_cols])
  predictions[, prob_cols] <- predictions[, prob_cols] / prob_sums
  
  # Prepare evaluation data
  eval_data <- prepare_findingemo_evaluation(annotations, predictions)
  
  # Test with evaluate_emotions
  results <- evaluate_emotions(
    data = eval_data,
    probs_cols = prob_cols,
    metrics = c("accuracy", "f1_macro", "confusion_matrix")
  )
  
  expect_type(results, "list")
  expect_s3_class(results, "emotion_evaluation")
  expect_true("accuracy" %in% names(results))
  expect_true("f1_macro" %in% names(results))
  expect_true("confusion_matrix" %in% names(results))
  
  # Check that results are reasonable
  expect_gte(results$accuracy, 0.5)  # Should be better than random
  expect_lte(results$accuracy, 1.0)
  expect_gte(results$f1_macro, 0.0)
  expect_lte(results$f1_macro, 1.0)
})