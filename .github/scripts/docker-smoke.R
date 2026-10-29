# Smoke test for the image built from docker/Dockerfile. Called by
# .github/workflows/docker.yml inside the image, with no network:
#   docker run --rm --network none -v "$PWD/.github/scripts:/scripts:ro" \
#     -e EXPECT_EXTRAS="" -e EXPECT_GPU=0 IMAGE Rscript /scripts/docker-smoke.R
# EXPECT_EXTRAS and EXPECT_GPU repeat the EXTRAS and GPU build arguments.
# Exits with status 1 if a check fails.

results <- data.frame(check = character(), status = character(), seconds = numeric())

check <- function(name, expr) {
  message("\n==> ", name)
  t0 <- Sys.time()
  ok <- tryCatch({
    value <- force(expr)
    if (!isTRUE(value)) stop("check returned ", deparse(value)[1])
    TRUE
  }, error = function(e) {
    message("FAILED: ", conditionMessage(e))
    FALSE
  })
  secs <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  results[nrow(results) + 1, ] <<- list(name, if (ok) "ok" else "FAILED", secs)
}

extras <- trimws(strsplit(Sys.getenv("EXPECT_EXTRAS"), ",")[[1]])
extras <- extras[nzchar(extras)]
gpu <- identical(Sys.getenv("EXPECT_GPU"), "1")
if (gpu) extras <- unique(c(extras, "gpu"))
python <- Sys.getenv("TRANSFOREMOTION_PYTHON")

check("image environment", {
  message("TRANSFOREMOTION_PYTHON=", python, "; HF_HUB_OFFLINE=", Sys.getenv("HF_HUB_OFFLINE"),
          "; user ", system("id -u", intern = TRUE))
  nzchar(python) && file.exists(python) && identical(Sys.getenv("HF_HUB_OFFLINE"), "1")
})

check("library(transforEmotion)", {
  suppressPackageStartupMessages(library(transforEmotion))
  message("transforEmotion ", packageVersion("transforEmotion"), ", R ", getRversion())
  TRUE
})

check("Python is the fixed environment", {
  transforEmotion:::ensure_te_py_env()
  cfg <- reticulate::py_config()
  message("python ", cfg$python, " (", cfg$version, ")")
  normalizePath(cfg$python) == normalizePath(python)
})

check("Python dependencies are consistent (uv pip check)", {
  out <- system2("uv", c("pip", "check", "--python", python), stdout = TRUE, stderr = TRUE)
  message(paste(out, collapse = "\n"))
  is.null(attr(out, "status"))
})

# The import names of each feature set, as the package checks them
for (feature in c("core", extras)) {
  local({
    check(paste0("Python modules: ", feature), {
      modules <- transforEmotion:::.te_feature_modules[[feature]]
      if (!length(modules)) stop("no module list for '", feature, "'")
      missing <- modules[!vapply(modules, reticulate::py_module_available, logical(1))]
      if (length(missing)) stop("not importable: ", paste(missing, collapse = ", "))
      TRUE
    })
  })
}

check("protobuf (issue #46)", {
  pb <- reticulate::import("google.protobuf")
  message("protobuf ", pb$`__version__`)
  package_version(pb$`__version__`) >= "3.20"
})

# transformers converts a SentencePiece model to a fast tokenizer when a
# repository has no tokenizer.json; the conversion needs protobuf. The model
# is trained here, so no download is needed.
check("SentencePiece model converts to a fast tokenizer", {
  dir <- tempfile("spm")
  dir.create(dir)
  corpus <- file.path(dir, "corpus.txt")
  writeLines(sprintf("Estoy muy contento de verte hoy. Tengo miedo. Siento ira y alegria %d", 1:200),
             corpus)
  spm <- reticulate::import("sentencepiece")
  spm$SentencePieceTrainer$train(input = corpus, model_prefix = file.path(dir, "m"),
                                 vocab_size = 60L, minloglevel = 2L)
  tok <- reticulate::import("transformers")$XLMRobertaTokenizerFast(
    vocab_file = file.path(dir, "m.model"))
  isTRUE(tok$is_fast)
})

check("torch build", {
  torch <- reticulate::import("torch")
  message("torch ", torch$`__version__`, "; CUDA runtime ", format(torch$version$cuda),
          "; CUDA device ", torch$cuda$is_available())
  if (gpu) grepl("+cu", torch$`__version__`, fixed = TRUE) else grepl("+cpu", torch$`__version__`, fixed = TRUE)
})

check("transformer_scores (default model, offline)", {
  s <- transformer_scores(text = "I am so happy to see you today!",
                          classes = c("joy", "anger", "fear"), device = "cpu")
  print(s)
  names(which.max(s[[1]])) == "joy"
})

check("sentence_similarity (default model, offline)", {
  s <- sentence_similarity(
    text = "The cat sat on the mat",
    comparison_text = c("A cat is sitting on a rug", "Stock markets fell sharply today")
  )
  print(s)
  s[1, 1] > s[1, 2]
})

check("image_scores (default model, offline)", {
  image <- system.file("extdata", "boris-1.png", package = "transforEmotion")
  s <- image_scores(image, classes = c("happiness", "sadness", "anger", "fear", "surprise", "neutral"))
  print(round(s, 3))
  nrow(s) >= 1
})

message("\n")
print(results, row.names = FALSE)
if (any(results$status != "ok")) quit(status = 1)
