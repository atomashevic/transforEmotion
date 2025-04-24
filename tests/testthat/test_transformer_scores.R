test_that("transformer_scores works with default settings", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  skip("This test only for manual testing")

  test_text <- "With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion
        classification of text, image, and video in R, *all without the need for a GPU,
        subscriptions, paid services, or using Python. Implements sentiment analysis
        using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines.
        The default pipeline for text is
        [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base)
        trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/snli) (SNLI) and
        [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/multi_nli) (MultiNLI) datasets.
        Using similar models, zero-shot classification transformers have demonstrated
        superior performance relative to other natural language processing models
        (Yin, Hay, & Roth, [2019](https://arxiv.org/abs/1909.00161)).
        All other zero-shot classification model pipelines can be implemented using their model name
        from https://huggingface.co/models?pipeline_tag=zero-shot-classification."

  test_classes <- c("technical", "informative", "promotional", "educational")

  result <- transformer_scores(
    text = test_text,
    classes = test_classes,
    transformer = "cross-encoder-distilroberta"
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
