test_that("image_scores works with all default models", {
  skip_on_cran()
  skip_on_ci()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  
  image_path <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  labels <- c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  
  # Test with oai-base model (default)
  result_base <- image_scores(
    image = image_path,
    classes = labels,
    model = "oai-base"
  )
  
  expect_s3_class(result_base, "data.frame")
  expect_equal(ncol(result_base), length(labels))
  expect_equal(names(result_base), labels)
  
  # Test with oai-large model
  result_large <- image_scores(
    image = image_path,
    classes = labels,
    model = "oai-large"
  )
  
  expect_s3_class(result_large, "data.frame")
  expect_equal(ncol(result_large), length(labels))
  expect_equal(names(result_large), labels)
  
  # Test with eva-8B model
  result_eva <- image_scores(
    image = image_path,
    classes = labels,
    model = "eva-8B"
  )
  
  expect_s3_class(result_eva, "data.frame")
  expect_equal(ncol(result_eva), length(labels))
  expect_equal(names(result_eva), labels)
  
  # Test with jina-v2 model
  result_jina <- image_scores(
    image = image_path,
    classes = labels,
    model = "jina-v2"
  )
  
  expect_s3_class(result_jina, "data.frame")
  expect_equal(ncol(result_jina), length(labels))
  expect_equal(names(result_jina), labels)
  
  # Verify that different models produce different results
  # (This is a simple check to ensure we're actually using different models)
  expect_false(identical(result_base, result_large))
  expect_false(identical(result_base, result_eva))
  expect_false(identical(result_base, result_jina))
})

test_that("image_scores works with local_model_path", {
  skip_on_cran()
  skip_on_ci()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  # skip("This test requires a local model directory")
  
  # This test is skipped by default as it requires a locally downloaded model
  # To run it, you need to:
  # 1. Download a model locally (e.g., using transformers-cli or git clone from HuggingFace)
  # 2. Set the local_model_path to that directory
  # 3. Remove the skip() line above
  
  image_path <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  labels <- c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")
  local_model_path <- "/home/aleksandar/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/" # Replace with actual path when testing
  
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