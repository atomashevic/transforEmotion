skip_on_cran()

test_that("validate_rag_json passes on valid structure", {
  skip_on_cran()
  skip_on_ci()
  x <- list(
    labels = c("joy", "surprise"),
    confidences = c(0.82, 0.47),
    intensity = 0.76,
    evidence_chunks = list(
      list(doc_id = "doc1", span = "some text", score = 0.89),
      list(doc_id = "doc2", span = "other text", score = 0.77)
    )
  )
  expect_true(validate_rag_json(x, error = FALSE))
})

test_that("validate_rag_json catches errors", {
  skip_on_cran()
  skip_on_ci()
  x <- list(
    labels = c("joy"),
    confidences = c(1.2), # out of range
    intensity = -0.1,     # out of range
    evidence_chunks = list(list(doc_id = 1, span = 2, score = "bad"))
  )
  expect_error(validate_rag_json(x))
})

test_that("parse_rag_json parses JSON and normalizes evidence", {
  skip_on_cran()
  skip_on_ci()
  j <- '{"labels":["joy","surprise"],"confidences":[0.8,0.5],"intensity":0.7,"evidence_chunks":[{"doc_id":"d1","span":"a","score":0.9}]}'
  p <- parse_rag_json(j)
  expect_type(p$labels, "character")
  expect_type(p$confidences, "double")
  expect_type(p$intensity, "double")
  expect_s3_class(p$evidence, "data.frame")
  expect_equal(nrow(p$evidence), 1)
})

test_that("as_rag_table produces long-form frame", {
  skip_on_cran()
  skip_on_ci()
  j <- '{"labels":["joy","surprise"],"confidences":[0.8,0.5],"intensity":0.7,"evidence_chunks":[{"doc_id":"d1","span":"a","score":0.9},{"doc_id":"d2","span":"b","score":0.8}]}'
  df <- as_rag_table(j)
  expect_s3_class(df, "data.frame")
  expect_equal(ncol(df), 6)
  expect_true(all(c("label","confidence","intensity","doc_id","span","score") %in% names(df)))
  expect_equal(nrow(df), 4) # 2 labels x 2 evidence
})

