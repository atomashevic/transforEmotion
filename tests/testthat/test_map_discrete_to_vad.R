library(transforEmotion)
library(testthat)

test_that("map_discrete_to_vad works with data.frame input (image/video scores)", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  # Create mock image/video scores data.frame
  mock_results <- data.frame(
    joy = c(0.8, 0.2, 0.1),
    sadness = c(0.1, 0.7, 0.2),
    anger = c(0.1, 0.1, 0.7),
    stringsAsFactors = FALSE
  )

  # Test weighted averaging (default)
  vad_weighted <- map_discrete_to_vad(mock_results, weighted = TRUE)

  expect_s3_class(vad_weighted, "data.frame")
  expect_equal(nrow(vad_weighted), 3)
  expect_named(vad_weighted, c("valence", "arousal", "dominance"))
  expect_true(all(sapply(vad_weighted, is.numeric)))

  # Test simple lookup (highest scoring emotion)
  vad_simple <- map_discrete_to_vad(mock_results, weighted = FALSE)

  expect_s3_class(vad_simple, "data.frame")
  expect_equal(nrow(vad_simple), 3)
  expect_named(vad_simple, c("valence", "arousal", "dominance"))
  expect_true(all(sapply(vad_simple, is.numeric)))

  # Results should be different between weighted and simple
  expect_false(identical(vad_weighted, vad_simple))
})

test_that("map_discrete_to_vad works with list input (transformer scores)", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  # Create mock transformer scores list
  mock_results <- list(
    "I am happy today" = c(joy = 0.8, sadness = 0.1, anger = 0.1),
    "This is terrible" = c(joy = 0.1, sadness = 0.2, anger = 0.7),
    "I feel neutral" = c(joy = 0.3, sadness = 0.3, anger = 0.4)
  )

  # Test weighted averaging
  vad_weighted <- map_discrete_to_vad(mock_results, weighted = TRUE)

  expect_s3_class(vad_weighted, "data.frame")
  expect_equal(nrow(vad_weighted), 3)
  expect_named(vad_weighted, c("text_id", "valence", "arousal", "dominance"))
  expect_equal(vad_weighted$text_id, names(mock_results))

  # Test simple lookup
  vad_simple <- map_discrete_to_vad(mock_results, weighted = FALSE)

  expect_s3_class(vad_simple, "data.frame")
  expect_equal(nrow(vad_simple), 3)
  expect_named(vad_simple, c("text_id", "valence", "arousal", "dominance"))
})

test_that("map_discrete_to_vad handles missing values gracefully", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  # Create data with missing values
  mock_results <- data.frame(
    joy = c(0.8, NA, 0.1),
    sadness = c(NA, 0.7, 0.2),
    anger = c(0.1, 0.1, NA),
    stringsAsFactors = FALSE
  )

  vad_results <- map_discrete_to_vad(mock_results)

  expect_s3_class(vad_results, "data.frame")
  expect_equal(nrow(vad_results), 3)

  # Should handle missing values without error
  expect_true(all(is.numeric(vad_results$valence)))
  expect_true(all(is.numeric(vad_results$arousal)))
  expect_true(all(is.numeric(vad_results$dominance)))
})

test_that("map_discrete_to_vad handles unknown emotions", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  # Create data with unknown emotion labels
  mock_results <- data.frame(
    unknown_emotion1 = c(0.8, 0.2),
    unknown_emotion2 = c(0.2, 0.8),
    stringsAsFactors = FALSE
  )

  expect_warning(
    vad_results <- map_discrete_to_vad(mock_results),
    "Could not find VAD values"
  )

  expect_s3_class(vad_results, "data.frame")
  expect_equal(nrow(vad_results), 2)

  # Should return NA for unknown emotions
  expect_true(all(is.na(vad_results$valence)))
  expect_true(all(is.na(vad_results$arousal)))
  expect_true(all(is.na(vad_results$dominance)))
})

test_that("map_discrete_to_vad validates input arguments", {

  # Test missing results
  expect_error(
    map_discrete_to_vad(),
    "results argument is required"
  )

  # Test invalid mapping
  mock_results <- data.frame(joy = 0.8, sadness = 0.2)
  expect_error(
    map_discrete_to_vad(mock_results, mapping = "invalid"),
    "mapping must be 'nrc_vad'"
  )

  # Test invalid input type - expect error about lexicon download since textdata won't be available
  expect_error(
    map_discrete_to_vad("invalid_input"),
    "Failed to load NRC VAD lexicon"
  )
})

test_that("map_discrete_to_vad requires pre-downloaded lexicon", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires NRC VAD lexicon to be pre-downloaded")
  
  mock_results <- data.frame(joy = 0.8, sadness = 0.2)
  
  # Test that function works when lexicon is available
  expect_no_error(
    vad_result <- map_discrete_to_vad(mock_results)
  )
  
  expect_s3_class(vad_result, "data.frame")
  expect_named(vad_result, c("valence", "arousal", "dominance"))
})

test_that("map_discrete_to_vad caching works", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  mock_results <- data.frame(joy = 0.8, sadness = 0.2)

  # First call should cache the lexicon
  vad1 <- map_discrete_to_vad(mock_results, cache_lexicon = TRUE)

  # Second call should use cached lexicon (faster)
  vad2 <- map_discrete_to_vad(mock_results, cache_lexicon = TRUE)

  # Results should be identical
  expect_identical(vad1, vad2)

  # Test without caching
  vad3 <- map_discrete_to_vad(mock_results, cache_lexicon = FALSE)
  expect_identical(vad1, vad3)
})

test_that("map_discrete_to_vad handles edge cases", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  # Test with single emotion
  single_emotion <- data.frame(joy = 1.0)
  vad_single <- map_discrete_to_vad(single_emotion)

  expect_s3_class(vad_single, "data.frame")
  expect_equal(nrow(vad_single), 1)

  # Test with all zeros
  zero_scores <- data.frame(joy = 0, sadness = 0, anger = 0)
  vad_zero <- map_discrete_to_vad(zero_scores)

  expect_s3_class(vad_zero, "data.frame")
  expect_equal(nrow(vad_zero), 1)

  # Test with empty list
  empty_list <- list()
  expect_error(
    map_discrete_to_vad(empty_list),
    NA  # Should handle gracefully, not error
  )
})

test_that("weighted vs non-weighted produces different results", {
  skip_on_cran()
  skip_if_not_installed("textdata")
  skip("This test requires manual execution and textdata setup")

  # Create data where weighted average would differ from max
  mock_results <- data.frame(
    joy = 0.4,      # Moderate joy
    sadness = 0.35, # High sadness
    anger = 0.25,   # Moderate anger
    stringsAsFactors = FALSE
  )

  vad_weighted <- map_discrete_to_vad(mock_results, weighted = TRUE)
  vad_simple <- map_discrete_to_vad(mock_results, weighted = FALSE)

  # Should produce different results since weighted considers all emotions
  # while simple only uses the highest (joy in this case)
  expect_false(identical(vad_weighted, vad_simple))
})
