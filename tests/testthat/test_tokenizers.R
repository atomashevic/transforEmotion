# transformers converts a SentencePiece model to a fast tokenizer when a
# repository has no tokenizer.json, and the conversion needs protobuf.
# These tests run only in a prepared Python environment.
skip_without_tokenizer_python <- function() {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON")),
          "TRANSFOREMOTION_PYTHON is not set")
  ensure_te_py_env()
  skip_if_not(reticulate::py_module_available("transformers"), "transformers is not available")
  skip_if_not(reticulate::py_module_available("sentencepiece"), "sentencepiece is not available")
}

test_that("a SentencePiece model converts to a fast tokenizer (protobuf)", {
  skip_without_tokenizer_python()
  expect_true(reticulate::py_module_available("google.protobuf"))

  # A small SentencePiece model trained here, so no download is needed
  dir <- withr::local_tempdir()
  corpus <- file.path(dir, "corpus.txt")
  writeLines(sprintf("Estoy muy contento de verte hoy. Tengo miedo. Siento ira y alegria %d", 1:200),
             corpus)
  spm <- reticulate::import("sentencepiece")
  spm$SentencePieceTrainer$train(input = corpus, model_prefix = file.path(dir, "m"),
                                 vocab_size = 60L, minloglevel = 2L)
  transformers <- reticulate::import("transformers")
  tok <- transformers$XLMRobertaTokenizerFast(vocab_file = file.path(dir, "m.model"))
  expect_true(tok$is_fast)
})

test_that("transformer_scores() scores Spanish text with joeddav/xlm-roberta-large-xnli", {
  skip_without_tokenizer_python()
  hub <- reticulate::import("huggingface_hub")
  # The weights, not only the config: the test must not start a 2.2 GB download
  cached <- hub$try_to_load_from_cache("joeddav/xlm-roberta-large-xnli", "model.safetensors")
  skip_if(!is.character(cached), "joeddav/xlm-roberta-large-xnli is not downloaded")

  s <- transformer_scores(
    text = "Estoy muy contento de verte hoy",
    classes = c("alegr\u00eda", "ira", "miedo"),
    transformer = "joeddav/xlm-roberta-large-xnli",
    device = "cpu", keep_in_env = FALSE
  )
  expect_equal(names(which.max(s[[1]])), "alegr\u00eda")
  expect_gt(s[[1]][["alegr\u00eda"]], 0.9)
})
