# Model lists and structured-output gating in rag() / rag_sentemo(), without Python

test_that("Qwen 2.5 is a known rag() transformer with structured outputs", {
  choices <- eval(formals(rag)$transformer)
  expect_true(all(c("Qwen2.5-1.5B", "Qwen2.5-3B", "Gemma3-1B", "Gemma3-4B") %in% choices))
  expect_true(all(c("qwen2.5-1.5b", "qwen2.5-3b", "gemma3-1b", "gemma3-4b") %in% .te_structured_llms))
  expect_false("tinyllama" %in% .te_structured_llms)
  expect_true(all(.te_structured_llms %in% tolower(choices)))
  expect_true(all(c("Qwen2.5-1.5B", "Qwen2.5-3B") %in% eval(formals(rag_sentemo)$transformer)))
})

test_that("response_cleanup leaves Qwen 2.5 answers unchanged", {
  expect_identical(response_cleanup("  {\"label\": \"joy\"} ", "qwen2.5-3b"), "{\"label\": \"joy\"}")
  expect_identical(response_cleanup("x", "qwen2.5-1.5b"), "x")
})

test_that("rag_sentemo() warns and falls back to text for a model without structured outputs", {
  # rag() is stubbed so that no Python is needed; the warning is raised before it runs
  local_mocked_bindings(rag = function(...) list(...)$output)
  expect_warning(out <- rag_sentemo("a text", transformer = "TinyLLAMA", output = "table"),
                 "Structured outputs")
  expect_identical(out, "text")
  expect_silent(out <- rag_sentemo("a text", transformer = "Qwen2.5-3B", output = "table"))
  expect_identical(out, "table")
})
