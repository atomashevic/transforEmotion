skip_on_cran()

library(transforEmotion)
library(testthat)

# A small stand-in for the NRC VAD lexicon. These tests check the mapping,
# not the downloaded lexicon.
local_vad <- data.frame(
  word = c("joy", "sadness", "anger", "happy", "sad", "fear"),
  valence = c(0.95, 0.10, 0.15, 0.90, 0.20, 0.15),
  arousal = c(0.70, 0.30, 0.85, 0.55, 0.40, 0.80),
  dominance = c(0.65, 0.25, 0.55, 0.60, 0.30, 0.25),
  stringsAsFactors = FALSE
)

clear_vad_cache <- function() {
  cache <- transforEmotion:::.te_vad_cache
  rm(list = ls(envir = cache), envir = cache)
}

test_that("map_discrete_to_vad works with data.frame input (image/video scores)", {
  # Create mock image/video scores data.frame
  mock_results <- data.frame(
    joy = c(0.8, 0.2, 0.1),
    sadness = c(0.1, 0.7, 0.2),
    anger = c(0.1, 0.1, 0.7),
    stringsAsFactors = FALSE
  )

  # Test weighted averaging (default)
  vad_weighted <- map_discrete_to_vad(mock_results, weighted = TRUE, vad_lexicon = local_vad)

  expect_s3_class(vad_weighted, "data.frame")
  expect_equal(nrow(vad_weighted), 3)
  expect_named(vad_weighted, c("valence", "arousal", "dominance"))
  expect_true(all(sapply(vad_weighted, is.numeric)))

  # Test simple lookup (highest scoring emotion)
  vad_simple <- map_discrete_to_vad(mock_results, weighted = FALSE, vad_lexicon = local_vad)

  expect_s3_class(vad_simple, "data.frame")
  expect_equal(nrow(vad_simple), 3)
  expect_named(vad_simple, c("valence", "arousal", "dominance"))
  expect_true(all(sapply(vad_simple, is.numeric)))

  # Results should be different between weighted and simple
  expect_false(identical(vad_weighted, vad_simple))
})

test_that("map_discrete_to_vad works with list input (transformer scores)", {

  # Create mock transformer scores list
  mock_results <- list(
    "I am happy today" = c(joy = 0.8, sadness = 0.1, anger = 0.1),
    "This is terrible" = c(joy = 0.1, sadness = 0.2, anger = 0.7),
    "I feel neutral" = c(joy = 0.3, sadness = 0.3, anger = 0.4)
  )

  # Test weighted averaging
  vad_weighted <- map_discrete_to_vad(mock_results, weighted = TRUE, vad_lexicon = local_vad)

  expect_s3_class(vad_weighted, "data.frame")
  expect_equal(nrow(vad_weighted), 3)
  expect_named(vad_weighted, c("text_id", "valence", "arousal", "dominance"))
  expect_equal(vad_weighted$text_id, names(mock_results))

  # Test simple lookup
  vad_simple <- map_discrete_to_vad(mock_results, weighted = FALSE, vad_lexicon = local_vad)

  expect_s3_class(vad_simple, "data.frame")
  expect_equal(nrow(vad_simple), 3)
  expect_named(vad_simple, c("text_id", "valence", "arousal", "dominance"))
})

test_that("map_discrete_to_vad handles missing values gracefully", {

  # Create data with missing values
  mock_results <- data.frame(
    joy = c(0.8, NA, 0.1),
    sadness = c(NA, 0.7, 0.2),
    anger = c(0.1, 0.1, NA),
    stringsAsFactors = FALSE
  )

  vad_results <- map_discrete_to_vad(mock_results, vad_lexicon = local_vad)

  expect_s3_class(vad_results, "data.frame")
  expect_equal(nrow(vad_results), 3)

  # Should handle missing values without error
  expect_true(all(is.numeric(vad_results$valence)))
  expect_true(all(is.numeric(vad_results$arousal)))
  expect_true(all(is.numeric(vad_results$dominance)))
})

test_that("map_discrete_to_vad handles unknown emotions", {
  # Create data with unknown emotion labels
  mock_results <- data.frame(
    unknown_emotion1 = c(0.8, 0.2),
    unknown_emotion2 = c(0.2, 0.8),
    stringsAsFactors = FALSE
  )

  # Each unknown label warns once per row, then the row warns that nothing mapped
  warned <- capture_warnings(
    vad_results <- map_discrete_to_vad(mock_results, vad_lexicon = local_vad)
  )
  expect_equal(sum(grepl("Could not find VAD values", warned)), 4)
  expect_equal(sum(grepl("No emotions could be mapped", warned)), 2)

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

  # A string is neither a data frame nor a list. Pass a lexicon so this
  # does not try to download NRC VAD.
  expect_error(
    map_discrete_to_vad("invalid_input", vad_lexicon = local_vad),
    "results must be a data.frame"
  )
})

test_that("map_discrete_to_vad loads and caches a lexicon from textdata", {
  skip_if_not_installed("textdata")
  clear_vad_cache()
  on.exit(clear_vad_cache(), add = TRUE)

  local_mocked_bindings(
    lexicon_nrc_vad = function() local_vad,
    .package = "textdata"
  )
  mock_results <- data.frame(joy = 0.8, sadness = 0.2)

  # First call loads the lexicon and stores it. Second call reuses the store
  # and does not ask textdata again.
  expect_message(
    vad1 <- map_discrete_to_vad(mock_results, cache_lexicon = TRUE),
    "VAD lexicon cached"
  )
  expect_no_message(
    vad2 <- map_discrete_to_vad(mock_results, cache_lexicon = TRUE)
  )
  expect_identical(vad1, vad2)

  # Without caching, the lexicon is loaded again and the result matches
  expect_message(
    vad3 <- map_discrete_to_vad(mock_results, cache_lexicon = FALSE),
    "Loading NRC VAD lexicon"
  )
  expect_identical(vad1, vad3)
  expect_named(vad1, c("valence", "arousal", "dominance"))
})

test_that("map_discrete_to_vad handles edge cases", {
  # Test with single emotion
  single_emotion <- data.frame(joy = 1.0)
  vad_single <- map_discrete_to_vad(single_emotion, vad_lexicon = local_vad)

  expect_s3_class(vad_single, "data.frame")
  expect_equal(nrow(vad_single), 1)

  # Test with all zeros: every emotion is known, so the row is still numeric
  zero_scores <- data.frame(joy = 0, sadness = 0, anger = 0)
  vad_zero <- map_discrete_to_vad(zero_scores, vad_lexicon = local_vad)

  expect_s3_class(vad_zero, "data.frame")
  expect_equal(nrow(vad_zero), 1)
  expect_true(all(is.numeric(unlist(vad_zero))))

  # Test with empty list
  expect_no_error(
    vad_empty <- map_discrete_to_vad(list(), vad_lexicon = local_vad)
  )
  expect_equal(nrow(vad_empty), 0)
})

test_that("weighted vs non-weighted produces different results", {

  # Create data where weighted average would differ from max
  mock_results <- data.frame(
    joy = 0.4,      # Moderate joy
    sadness = 0.35, # High sadness
    anger = 0.25,   # Moderate anger
    stringsAsFactors = FALSE
  )

  vad_weighted <- map_discrete_to_vad(mock_results, weighted = TRUE, vad_lexicon = local_vad)
  vad_simple <- map_discrete_to_vad(mock_results, weighted = FALSE, vad_lexicon = local_vad)

  # Should produce different results since weighted considers all emotions
  # while simple only uses the highest (joy in this case)
  expect_false(identical(vad_weighted, vad_simple))
})
