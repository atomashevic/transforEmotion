# Standalone demo: RAG on MLK excerpt with Gemma3-1B
# Compares vector (semantic) vs BM25 (lexical) retrieval across tasks

suppressPackageStartupMessages(library(transforEmotion))

# --- Configuration ---------------------------------------------------------

# Gemma 3 access: ensure you have an HF token set, e.g.:
# Sys.setenv(HF_TOKEN = "hf_...")

# MLK excerpt (as used in the manuscript)
mlk_text <- paste(
  "So even though we face the difficulties of today and tomorrow,",
  "I still have a dream. It is a dream deeply rooted in the American dream.",
  "I have a dream that one day this nation will rise up and live out the true",
  "meaning of its creed: 'We hold these truths to be self-evident, that all men",
  "are created equal.'",
  sep = " \n"
)

# Retrieval settings
SIM_TOPK <- 5

# --- Helper to run a single case ------------------------------------------

run_case <- function(task = c("sentiment","emotion","general"),
                     retriever = c("vector","bm25")) {
  task <- match.arg(task)
  retriever <- match.arg(retriever)

  query <- switch(task,
    sentiment = "Determine the overall sentiment as positive, neutral, or negative.",
    emotion   = "Extract the predominant emotion in the document (Emo8).",
    general   = "Identify salient emotional themes and supporting evidence."
  )

  out_mode <- if (task %in% c("sentiment","emotion")) "table" else "table"

  cat(sprintf("\n=== Task: %s | Retriever: %s ===\n", task, retriever))
  t0 <- Sys.time()
  res <- try(
    rag(
      text = mlk_text,
      transformer = "Gemma3-1B",
      query = query,
      response_mode = "compact",
      similarity_top_k = SIM_TOPK,
      retriever = retriever,
      output = out_mode,
      task = task,
      progress = TRUE
    ), silent = TRUE
  )
  dt <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)

  if (inherits(res, "try-error")) {
    cat("[ERROR] ", as.character(res), "\n", sep = "")
    return(invisible(NULL))
  }

  cat(sprintf("... completed in %ss\n", dt))

  if (is.data.frame(res)) {
    print(res)
  } else if (is.character(res) && length(res) == 1) {
    cat(res, "\n")
  } else if (inherits(res, "rag")) {
    print(res)
  } else {
    str(res)
  }

  invisible(res)
}

# --- Run all comparisons ---------------------------------------------------

cat("\ntransforEmotion RAG demo: MLK excerpt with Gemma3-1B\n")
cat("If Gemma 3 download fails, set HF_TOKEN and accept model terms on Hugging Face.\n\n")

# Sentiment
run_case("sentiment", "vector")
run_case("sentiment", "bm25")

# Emotion (Emo8 default labels)
run_case("emotion", "vector")
run_case("emotion", "bm25")

# General themes
run_case("general", "vector")
run_case("general", "bm25")

cat("\nDone.\n")

