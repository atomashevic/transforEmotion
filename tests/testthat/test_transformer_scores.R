test_that("transformer_scores works with default settings", {
  # skip_on_cran()
  skip("This is the basic functionality test")

  text <- "So even though we face the difficulties
          of today and tomorrow, I still have a dream.
          It is a dream deeply rooted in the American dream.
          I have a dream that one day this nation will rise
          up and live out the true meaning of its creed:
          We hold these truths to be self-evident,
          that all men are created equal."

  # Define custom emotion categories
  # Researchers can customize these based on their needs
  emotions <- c("anger", "fear", "joy", "sadness",
                "optimism", "hope", "surprise", "disgust")

  # Run analysis with default DistilRoBERTa
  # Fast and efficient model
  results <- transformer_scores(
    text = text,         # Text to analyze
    classes = emotions   # List of emotions to detect
  )

  expect_type(result, "list")
  expect_named(result, test_text)
  expect_equal(names(result[[1]]), test_classes)
  expect_true(all(sapply(result[[1]], function(x) x >= 0 && x <= 1)))
})

test_that("transformer_scores works with local_model_path", {
  skip_on_cran()
  skip_on_ci()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  skip("This test is for manual testing only")
  #skip("This test requires a local model directory")

  # This test is skipped by default as it requires a locally downloaded model
  # To run it, you need to:
  # 1. Download a model locally (e.g., using transformers-cli or git clone from HuggingFace)
  # 2. Set the local_model_path to that directory
  # 3. Remove the skip() line above

  test_text <- "With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion classification"
  test_classes <- c("technical", "informative", "promotional", "educational")
  local_model_path <- "/home/aleksandar/.cache/huggingface/hub/models--cross-encoder--nli-distilroberta-base/snapshots/b5b020e8117e1ddc6a0c7ed0fd22c0e679edf0fa/" # Replace with actual path when testing

  result <- transformer_scores(
    text = test_text,
    classes = test_classes,
    transformer = "custom-model", # This can be any string as the local_model_path is used
    local_model_path = local_model_path
  )

  expect_type(result, "list")
  expect_named(result, test_text)
  expect_equal(names(result[[1]]), test_classes)
  expect_true(all(sapply(result[[1]], function(x) x >= 0 && x <= 1)))
})
