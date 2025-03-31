# test_that("image_scores returns the correct output for a single face image", {
#   # Test with example values
#   trump = 'https://s.abcnews.com/images/US/trump-mugshot-main-ht-jt-230824_1692924861331_hpMain_16x9_1600.jpg'
#   labels =  c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
#   result = image_scores(trump, labels)
#   expect_equal(length(labels), ncol(result))
# })

# test_that("image_scores returns the correct output for a multiple face image", {
#   # Test with example values
#   image = 'https://cloudfront-us-east-1.images.arcpublishing.com/bostonglobe/BDI4NHFBKEYSK3GB34Q62HZJ4U.jpg'
#   labels =  c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
#   result = image_scores(image, labels, face_selection = "largest")
#   expect_equal(length(labels), ncol(result))
# })

# test_that("selecting left and right face produces different results",
# {
#   image = 'https://cloudfront-us-east-1.images.arcpublishing.com/bostonglobe/BDI4NHFBKEYSK3GB34Q62HZJ4U.jpg'
#   labels =  c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
#   result_left = image_scores(image, labels, face_selection = "left")
#   result_right = image_scores(image, labels, face_selection = "right")
#   expect_equal(sum(result_left ==  result_right), 0 )
# })

test_that("image_scores works with local_model_path", {
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
  
  image_path <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  labels <- c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  local_model_path <- "/path/to/local/model" # Replace with actual path when testing
  
  result <- image_scores(
    image = image_path,
    classes = labels,
    model = "custom-model", # This can be any string as the local_model_path is used
    local_model_path = local_model_path
  )
  
  expect_s3_class(result, "data.frame")
  expect_equal(ncol(result), length(labels))
  expect_equal(names(result), labels)
})