test_that("rag uses modern llama-index import path", {
  rag_path <- normalizePath(testthat::test_path("..", "..", "R", "rag.R"), mustWork = FALSE)
  skip_if_not(file.exists(rag_path))

  src <- paste(readLines(rag_path, warn = FALSE), collapse = "\n")
  expect_false(grepl("llama_index\\.legacy", src))
  expect_true(grepl("te_validate_modern_llama_index", src, fixed = TRUE))
})
