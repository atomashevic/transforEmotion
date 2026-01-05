skip_on_cran()

test_that("video_scores works with local_model_path", {
  skip_on_cran()
  skip_on_ci()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  skip("This test requires a local model directory")
  
  # This test is skipped by default as it requires a locally downloaded model
  # To run it, you need to:
  # 1. Download a model locally (e.g., using transformers-cli or git clone from HuggingFace)
  # 2. Set the local_model_path to that directory
  # 3. Remove the skip() line above
  
  # Use a short YouTube video or a local video file for testing
  video_url <- "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example YouTube URL
  labels <- c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  local_model_path <- "/path/to/local/model" # Replace with actual path when testing
  
  # Create a temporary directory for frames
  temp_dir <- tempdir()
  
  result <- video_scores(
    video = video_url,
    classes = labels,
    nframes = 5, # Use a small number of frames for faster testing
    face_selection = "largest",
    save_dir = temp_dir,
    model = "custom-model", # This can be any string as the local_model_path is used
    local_model_path = local_model_path
  )
  
  expect_s3_class(result, "data.frame")
  expect_equal(ncol(result), length(labels))
  expect_equal(names(result), labels)
  
  # Clean up
  unlink(temp_dir, recursive = TRUE)
})