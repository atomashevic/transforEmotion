# vad_scores() with transformer_scores() mocked, so these tests use no Python
# and no network.

# Records each call that vad_scores() makes. Every text scores 0.8 for the
# first (high) pole, unless score() gives other values for the classes.
mock_transformer_scores <- function(score = function(classes) c(0.8, 0.2),
                                    env = parent.frame()) {
  seen <- new.env()
  seen$transformer <- character()
  seen$classes <- list()
  local_mocked_bindings(
    transformer_scores = function(text, classes, transformer, ...) {
      seen$transformer <- c(seen$transformer, transformer)
      seen$classes <- c(seen$classes, list(classes))
      lapply(text, function(x) stats::setNames(score(classes), classes))
    },
    .env = env
  )
  seen
}

test_that("vad_scores() uses cross-encoder-distilroberta for text when model = 'auto'", {
  seen <- mock_transformer_scores()
  out <- vad_scores("I am happy", input_type = "text", dimensions = "valence")
  expect_equal(seen$transformer, "cross-encoder-distilroberta")
  expect_equal(unname(out$valence), 0.8)
})

test_that("vad_scores() gives a Hugging Face model ID unchanged to transformer_scores()", {
  seen <- mock_transformer_scores()
  vad_scores("Estoy muy contento", input_type = "text",
             dimensions = c("valence", "arousal"),
             model = "joeddav/xlm-roberta-large-xnli")
  expect_equal(seen$transformer, rep("joeddav/xlm-roberta-large-xnli", 2))
})

test_that("vad_scores() rejects 'transformer' or 'classes' in ... and a model that is not one string", {
  seen <- mock_transformer_scores()
  expect_error(
    vad_scores("x", input_type = "text", transformer = "facebook-bart"),
    "vad_scores() sets 'transformer' itself", fixed = TRUE
  )
  expect_error(
    vad_scores("x", input_type = "text", dimensions = "valence", classes = c("a", "b")),
    "vad_scores() sets 'classes' itself", fixed = TRUE
  )
  # An abbreviation of 'classes' matches it by partial matching
  expect_error(
    vad_scores("x", input_type = "text", dimensions = "valence", class = c("a", "b")),
    "vad_scores() sets 'classes' itself", fixed = TRUE
  )
  expect_error(vad_scores("x", input_type = "text", model = NULL), "one character string")
  expect_error(vad_scores("x", input_type = "text", model = c("a", "b")), "one character string")
  expect_error(vad_scores("x", input_type = "text", model = NA_character_), "one character string")
  expect_length(seen$transformer, 0)
})

test_that("the custom-label example on the help page runs with one dimension", {
  seen <- mock_transformer_scores()
  # As in the Examples of ?vad_scores
  texts <- c("I'm absolutely thrilled!", "I feel so helpless and sad", "This is boring")
  custom_labels <- list(
    valence = list(
      positive = "Customer satisfaction and positive brand sentiment",
      negative = "Customer complaints and negative brand sentiment"
    )
  )
  brand_vad <- vad_scores(texts, dimensions = "valence",
                          label_type = "custom", custom_labels = custom_labels)
  expect_named(brand_vad, c("input_id", "valence"))
  expect_equal(brand_vad$input_id, texts)
  expect_equal(brand_vad$valence, rep(0.8, 3))
  expect_equal(seen$classes, list(c(
    "Customer satisfaction and positive brand sentiment",
    "Customer complaints and negative brand sentiment"
  )))

  # A requested dimension without labels still stops
  expect_error(
    vad_scores(texts, dimensions = c("valence", "arousal"),
               label_type = "custom", custom_labels = custom_labels),
    "custom_labels must have elements named: valence, arousal"
  )
  expect_length(seen$classes, 1)
})

test_that("validate_vad_labels() checks only the requested dimensions", {
  valence_only <- list(valence = list(positive = "happy emotions", negative = "sad emotions"))
  expect_true(transforEmotion:::validate_vad_labels(valence_only, "valence"))
  expect_error(
    transforEmotion:::validate_vad_labels(valence_only, c("valence", "dominance")),
    "custom_labels must have elements named: valence, dominance"
  )
  # A requested dimension with a missing pole stops
  expect_error(
    transforEmotion:::validate_vad_labels(list(arousal = list(high = "energetic states")), "arousal"),
    "Dimension 'arousal' must have poles: high, low"
  )
})

test_that("only definitional labels fall back to simple labels when all scores are NA", {
  # NA for every class except the simple valence labels
  na_unless_simple <- function(classes) {
    if (identical(classes, c("positive", "negative"))) c(0.8, 0.2) else c(NA_real_, NA_real_)
  }

  seen <- mock_transformer_scores(na_unless_simple)
  expect_warning(
    out <- vad_scores("x", input_type = "text", dimensions = "valence"),
    "Definitional labels failed for dimension 'valence'"
  )
  expect_equal(out$valence, 0.8)
  expect_equal(seen$classes[[2]], c("positive", "negative"))

  # Simple labels: no warning and no second call with the same labels
  seen <- mock_transformer_scores(function(classes) c(NA_real_, NA_real_))
  expect_no_warning(
    out <- vad_scores("x", input_type = "text", dimensions = "valence", label_type = "simple")
  )
  expect_true(is.na(out$valence))
  expect_length(seen$classes, 1)

  # Custom labels: no warning, and the simple labels never replace them
  seen <- mock_transformer_scores(na_unless_simple)
  custom_labels <- list(valence = list(positive = "satisfied customer", negative = "angry customer"))
  expect_no_warning(
    out <- vad_scores("x", input_type = "text", dimensions = "valence",
                      label_type = "custom", custom_labels = custom_labels)
  )
  expect_true(is.na(out$valence))
  expect_equal(seen$classes, list(c("satisfied customer", "angry customer")))
})
