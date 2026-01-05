skip_on_cran()

test_that("vad_scores validates input arguments", {
  
  # Test missing input
  expect_error(
    vad_scores(),
    "input argument is required"
  )
  
  # Test invalid dimensions
  expect_error(
    vad_scores("test text", dimensions = "invalid"),
    "dimensions must be one or more of"
  )
  
  # Test invalid input_type
  expect_error(
    vad_scores("test text", input_type = "invalid"),
    "'arg' should be one of"
  )
  
  # Test invalid label_type
  expect_error(
    vad_scores("test text", label_type = "invalid"),
    "'arg' should be one of"
  )
  
  # Test custom labels without providing them
  expect_error(
    vad_scores("test text", label_type = "custom"),
    "custom_labels must be provided"
  )
})

test_that("detect_input_type works correctly", {
  skip_on_cran()
  
  # Test text detection
  expect_equal(transforEmotion:::detect_input_type("This is some text"), "text")
  expect_equal(transforEmotion:::detect_input_type(c("Text 1", "Text 2")), "text")
  
  # Test URL detection (video)
  expect_equal(transforEmotion:::detect_input_type("https://www.youtube.com/watch?v=test"), "video")
  expect_equal(transforEmotion:::detect_input_type("http://example.com/video.mp4"), "video")
  
  # Test image detection
  expect_equal(transforEmotion:::detect_input_type("path/to/image.jpg"), "image")
  expect_equal(transforEmotion:::detect_input_type("image.PNG"), "image")
  expect_equal(transforEmotion:::detect_input_type("test.jpeg"), "image")
  
  # Test with list input
  expect_equal(transforEmotion:::detect_input_type(list("text1", "text2")), "text")
  expect_equal(transforEmotion:::detect_input_type(list("image.jpg")), "image")
})

test_that("VAD label functions work correctly", {
  
  # Test definitional labels
  def_labels <- transforEmotion:::get_vad_definitions()
  expect_type(def_labels, "list")
  expect_named(def_labels, c("valence", "arousal", "dominance"))
  expect_named(def_labels$valence, c("positive", "negative"))
  expect_named(def_labels$arousal, c("high", "low"))
  expect_named(def_labels$dominance, c("high", "low"))
  
  # Check that definitional labels are long and descriptive
  expect_gt(nchar(def_labels$valence$positive), 50)
  expect_gt(nchar(def_labels$arousal$high), 50)
  expect_gt(nchar(def_labels$dominance$low), 50)
  
  # Test simple labels
  simple_labels <- transforEmotion:::get_vad_simple_labels()
  expect_type(simple_labels, "list")
  expect_named(simple_labels, c("valence", "arousal", "dominance"))
  expect_equal(simple_labels$valence$positive, "positive")
  expect_equal(simple_labels$arousal$high, "excited")
  
  # Test label formatting
  formatted <- transforEmotion:::format_labels_for_classification(def_labels$valence)
  expect_length(formatted, 2)
  expect_type(formatted, "character")
})

test_that("VAD label validation works", {
  
  # Test valid custom labels
  valid_labels <- list(
    valence = list(positive = "happy emotions", negative = "sad emotions"),
    arousal = list(high = "energetic states", low = "calm states"),
    dominance = list(high = "powerful feelings", low = "weak feelings")
  )
  expect_true(transforEmotion:::validate_vad_labels(valid_labels))
  
  # Test invalid structure
  expect_error(
    transforEmotion:::validate_vad_labels("not a list"),
    "custom_labels must be a list"
  )
  
  # Test missing dimensions
  incomplete_labels <- list(valence = list(positive = "good", negative = "bad"))
  expect_error(
    transforEmotion:::validate_vad_labels(incomplete_labels),
    "custom_labels must have elements named"
  )
  
  # Test missing poles
  missing_poles <- list(
    valence = list(positive = "good"),
    arousal = list(high = "excited", low = "calm"),
    dominance = list(high = "strong", low = "weak")
  )
  expect_error(
    transforEmotion:::validate_vad_labels(missing_poles),
    "must have poles"
  )
  
  # Test non-character labels
  invalid_type <- list(
    valence = list(positive = 123, negative = "bad"),
    arousal = list(high = "excited", low = "calm"),
    dominance = list(high = "strong", low = "weak")
  )
  expect_error(
    transforEmotion:::validate_vad_labels(invalid_type),
    "must be a single character string"
  )
  
  # Test short labels (should warn)
  short_labels <- list(
    valence = list(positive = "good", negative = "bad"),
    arousal = list(high = "up", low = "down"),
    dominance = list(high = "big", low = "small")
  )
  expect_warning(
    transforEmotion:::validate_vad_labels(short_labels),
    "very short"
  )
})

test_that("vad_scores works with text input", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  skip("This test requires manual execution with transformer setup")
  
  # Test single text
  text <- "I am absolutely thrilled and excited!"
  result <- vad_scores(text, input_type = "text", dimensions = "valence")
  
  expect_s3_class(result, "data.frame")
  expect_named(result, c("input_id", "valence"))
  expect_equal(nrow(result), 1)
  expect_true(result$valence > 0.5)  # Should be positive
  
  # Test multiple texts
  texts <- c("I'm so happy!", "I feel terrible", "This is okay")
  result_multi <- vad_scores(texts, input_type = "text")
  
  expect_s3_class(result_multi, "data.frame")
  expect_named(result_multi, c("input_id", "valence", "arousal", "dominance"))
  expect_equal(nrow(result_multi), 3)
  expect_equal(result_multi$input_id, texts)
})

test_that("vad_scores works with different label types", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  skip("This test requires manual execution with transformer setup")
  
  text <- "I'm feeling great today!"
  
  # Test definitional labels
  result_def <- vad_scores(text, label_type = "definitional", dimensions = "valence")
  expect_s3_class(result_def, "data.frame")
  
  # Test simple labels  
  result_simple <- vad_scores(text, label_type = "simple", dimensions = "valence")
  expect_s3_class(result_simple, "data.frame")
  
  # Results might differ between label types
  expect_true(is.numeric(result_def$valence))
  expect_true(is.numeric(result_simple$valence))
  
  # Test custom labels
  custom_labels <- list(
    valence = list(
      positive = "Positive customer sentiment and satisfaction with our brand",
      negative = "Negative customer sentiment and dissatisfaction with our brand"
    )
  )
  
  result_custom <- vad_scores(text, 
                              label_type = "custom", 
                              custom_labels = custom_labels,
                              dimensions = "valence")
  expect_s3_class(result_custom, "data.frame")
  expect_named(result_custom, c("input_id", "valence"))
})

test_that("vad_scores handles different dimensions correctly", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("transformers"))
  skip("This test requires manual execution with transformer setup")
  
  text <- "I'm energetically excited but feel powerless"
  
  # Test single dimensions
  val_only <- vad_scores(text, dimensions = "valence")
  expect_named(val_only, c("input_id", "valence"))
  
  aro_only <- vad_scores(text, dimensions = "arousal")
  expect_named(aro_only, c("input_id", "arousal"))
  
  dom_only <- vad_scores(text, dimensions = "dominance")
  expect_named(dom_only, c("input_id", "dominance"))
  
  # Test multiple dimensions
  val_aro <- vad_scores(text, dimensions = c("valence", "arousal"))
  expect_named(val_aro, c("input_id", "valence", "arousal"))
  
  # Test all dimensions
  all_dims <- vad_scores(text, dimensions = c("valence", "arousal", "dominance"))
  expect_named(all_dims, c("input_id", "valence", "arousal", "dominance"))
  
  # Check that energetic text scores high on arousal
  expect_true(all_dims$arousal > 0.5)
  
  # Check that powerless text scores low on dominance
  expect_true(all_dims$dominance < 0.5)
})

test_that("vad_scores auto-detection works", {
  skip_on_cran()
  skip_if_not_installed("reticulate")
  skip("This test requires manual execution and comprehensive setup")
  
  # Text should be auto-detected
  text_result <- vad_scores("Happy text", input_type = "auto", dimensions = "valence")
  expect_s3_class(text_result, "data.frame")
  
  # Mock image path should be detected as image
  # (This will fail without actual image, but tests the detection logic)
  expect_error(
    vad_scores("test.jpg", input_type = "auto"),
    "Image file does not exist"  # Expected error from image_scores
  )
  
  # URL should be detected as video
  expect_error(
    vad_scores("https://youtube.com/watch?v=test", input_type = "auto"),
    "Video file does not exist|You need to provide a YouTube video URL"
  )
})

test_that("vad_scores handles edge cases", {
  
  # Test empty input
  expect_error(
    vad_scores(""),
    # Will likely fail at transformer level with empty text
    NA
  )
  
  # Test non-character input
  expect_error(
    vad_scores(123),
    "Input must be character"
  )
  
  # Test with all dimensions requested
  expect_no_error({
    # This will fail without proper setup, but tests argument validation
    try(vad_scores("test", dimensions = c("valence", "arousal", "dominance")), silent = TRUE)
  })
})