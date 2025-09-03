#' Structured Emotion/Sentiment via RAG (Small LLMs)
#'
#' @description
#' Convenience wrapper around \code{rag()} that keeps vector retrieval but
#' simplifies getting structured outputs for emotion or sentiment analysis
#' using small local LLMs (1–4B) with sensible defaults.
#'
#' @param text Character vector or list. Text to analyze. 
#'   One entry per document.
#' @param path Character. Optional directory with files to index (e.g., PDFs).
#'             If provided, overrides \code{text}.
#' @param task Character. One of \code{"emotion"} or \code{"sentiment"}.
#' @param labels_set Character vector of allowed labels. 
#'   If \code{NULL}, defaults
#'   to Emo8 for \code{task = "emotion"} and c("positive","neutral","negative")
#'   for \code{task = "sentiment"}.
#' @param max_labels Integer. Max number of labels to return.
#' @param transformer Character. Small local LLM to use. One of:
#'   \itemize{
#'     \item \code{"TinyLLAMA"} (default)
#'     \item \code{"Gemma3-1B"}
#'     \item \code{"Gemma3-4B"}
#'     \item \code{"Qwen3-0.6B"}
#'     \item \code{"Qwen3-1.7B"}
#'     \item \code{"Ministral-3B"}
#'   }
#' @param similarity_top_k Integer. Retrieval depth per query. Default 5.
#' @param response_mode Character. LlamaIndex response mode. 
#'   Default \code{"compact"}.
#' @param output Character. \code{"table"} (default) or \code{"json"}.
#' @param global_analysis Logical. If TRUE, analyze all documents 
#'   jointly. Default FALSE.
#' @param ... Additional arguments passed to \code{rag()} 
#'   (e.g., \code{device}, \code{keep_in_env}).
#'
#' @return
#' For Gemma3-1B/4B and \code{output = "table"}/\code{"csv"}, 
#'   a data.frame with columns\cr
#' \code{doc_id, text, label, confidence}.\cr
#' For Gemma3-1B/4B and \code{output = "json"}, a JSON array 
#'   of per-doc objects with those fields.\cr
#' For other models, structured outputs are not supported; 
#'   the function falls back to \code{output = "text"} and 
#'   returns a free-text \code{"rag"} object.
#'
#' @examples
#' \dontrun{
#' texts <- c(
#'   "I feel so happy and grateful today!",
#'   "This is frustrating and makes me angry."
#' )
#' rag_sentemo(texts, task = "emotion", output = "table")
#' rag_sentemo(texts, task = "sentiment", output = "json")
#' }
#'
#' @export
rag_sentemo <- function(
  text = NULL,
  path = NULL,
  task = c("emotion", "sentiment"),
  labels_set = NULL,
  max_labels = 5,
  transformer = c(
    "TinyLLAMA", "Gemma3-1B", "Gemma3-4B", "Qwen3-1.7B", "Ministral-3B"
  ),
  similarity_top_k = 5,
  response_mode = c("compact", "refine", "simple_summarize"),
  output = c("table", "json", "csv"),
  global_analysis = FALSE,
  ...
){
  task <- match.arg(task)
  transformer <- match.arg(transformer)
  response_mode <- match.arg(response_mode)
  output <- match.arg(output)

  # Enforce: structured outputs supported only by Gemma3-1B / Gemma3-4B
  t_lower <- tolower(transformer)
  if (!t_lower %in% c("gemma3-1b", "gemma3-4b") && output != "text") {
    stop("Structured outputs (json/table/csv) are supported only for ",
         "Gemma3-1B and Gemma3-4B.", call. = FALSE)
  }

  # Default label sets if not provided
  if (is.null(labels_set)) {
    if (identical(task, "emotion")) {
      labels_set <- c("joy", "trust", "fear", "surprise", 
                       "sadness", "disgust", "anger", "anticipation")
    } else {
      labels_set <- c("positive", "neutral", "negative")
    }
  }

  # Short, structured-oriented query per task
  query <- if (identical(task, "emotion")) {
    "Extract the predominant emotions in each document."
  } else {
    "Extract the overall sentiment (positive, neutral, negative) \n    in each document."
  }

  rag(
    text = text,
    path = path,
    transformer = transformer,
    query = query,
    response_mode = response_mode,
    similarity_top_k = similarity_top_k,
    output = output,
    task = task,
    labels_set = labels_set,
    max_labels = max_labels,
    global_analysis = global_analysis,
    ...
  )
}
