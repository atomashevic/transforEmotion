#' RAG retriever registry and helpers
#'
#' Provides a simple plugin system to add custom retrievers and built-in
#' handlers for vector and BM25 retrieval. Used internally by `rag()`.
#'
#' @name rag_retrievers
#' @keywords internal
NULL

.te_retriever_registry <- new.env(parent = emptyenv())

#' Register a custom retriever
#'
#' @description
#' Registers a retriever under a name. The handler should construct and return
#' a query engine compatible with llama-index or a fallback with a `query_fn`.
#'
#' @param name Character scalar; retriever name (e.g., "my_retriever").
#' @param handler Function with signature:
#'   function(llama_index, documents, service_context, similarity_top_k,
#'            response_mode, params) -> engine_or_list
#'   where the return value is either a Python query engine with `$query()`
#'   or a list with element `query_fn` taking a single `query` argument
#'   and returning a list with `response` and `source_nodes`.
#'
#' @export
register_retriever <- function(name, handler)
{
  stopifnot(is.character(name), length(name) == 1L, nchar(name) > 0)
  stopifnot(is.function(handler))
  assign(name, handler, envir = .te_retriever_registry)
  invisible(TRUE)
}

#' @noRd
get_registered_retriever <- function(name)
{
  if (exists(name, envir = .te_retriever_registry, inherits = FALSE)) {
    get(name, envir = .te_retriever_registry, inherits = FALSE)
  } else {
    NULL
  }
}

#' @noRd
run_query <- function(engine_or_list, query)
{
  # Fallback: R list with query_fn
  if (is.list(engine_or_list) && !is.null(engine_or_list$query_fn)) {
    return(engine_or_list$query_fn(query))
  }
  # Python engine
  return(engine_or_list$query(query))
}

#' @noRd
vector_retriever_handler <- function(llama_index, documents, service_context,
                                     similarity_top_k, response_mode, params)
{
  index <- llama_index$VectorStoreIndex(
    documents, service_context = service_context,
    show_progress = isTRUE(params$show_progress)
  )
  index$as_query_engine(
    similarity_top_k = as.integer(similarity_top_k),
    response_mode = response_mode
  )
}

#' @noRd
bm25_retriever_handler <- function(llama_index, documents, service_context,
                                   similarity_top_k, response_mode, params)
{
  # Try native BM25 retriever in llama-index
  engine <- try({
    bm25_cls <- NULL
    # common locations across versions
    if (!is.null(llama_index$retrievers) &&
        !is.null(llama_index$retrievers$BM25Retriever)) {
      bm25_cls <- llama_index$retrievers$BM25Retriever
    } else if (!is.null(llama_index$legacy) &&
               !is.null(llama_index$legacy$retrievers) &&
               !is.null(llama_index$legacy$retrievers$BM25Retriever)) {
      bm25_cls <- llama_index$legacy$retrievers$BM25Retriever
    }
    if (!is.null(bm25_cls)) {
      bm25 <- bm25_cls$from_defaults(
        documents = documents,
        similarity_top_k = as.integer(similarity_top_k)
      )
      return(
        llama_index$RetrieverQueryEngine$from_args(
          retriever = bm25,
          service_context = service_context,
          response_mode = response_mode
        )
      )
    }
    NULL
  }, silent = TRUE)

  if (!inherits(engine, "try-error") && !is.null(engine)) return(engine)

  # Fallback: rank_bm25 with manual synthesis using the service LLM
  rank_bm25 <- try(reticulate::import("rank_bm25", delay_load = TRUE), silent = TRUE)
  if (inherits(rank_bm25, "try-error")) {
    warning("rank_bm25 is not available; falling back to vector retriever", call. = FALSE)
    return(vector_retriever_handler(llama_index, documents, service_context,
                                    similarity_top_k, response_mode, params))
  }

  # Extract plain texts per document
  doc_texts <- vapply(documents, function(d) as.character(d$text), character(1))
  # Simple tokenization for BM25
  tokenize <- function(x) {
    tokens <- unlist(strsplit(tolower(x), "[^a-z0-9]+"))
    tokens[nzchar(tokens)]
  }
  corpus <- lapply(doc_texts, tokenize)
  bm25 <- rank_bm25$BM25Okapi(corpus)

  # Build query function closure
  query_fn <- function(q) {
    q_tokens <- tokenize(q)
    scores <- try(as.numeric(bm25$get_scores(q_tokens)), silent = TRUE)
    if (inherits(scores, "try-error") || length(scores) == 0) {
      scores <- rep(0, length(doc_texts))
    }
    # Select top-k
    k <- min(as.integer(similarity_top_k), length(scores))
    ord <- order(scores, decreasing = TRUE)
    take <- ord[seq_len(k)]
    top_scores <- scores[take]
    # Normalize to [0,1]
    if (length(top_scores) > 0) {
      smin <- min(top_scores); smax <- max(top_scores)
      norm <- if (smax > smin) (top_scores - smin) / (smax - smin) else rep(1, length(top_scores))
    } else {
      norm <- numeric(0)
    }

    # Compose context
    context <- paste(
      vapply(seq_along(take), function(i){
        paste0("[doc_", take[i], "] ", substr(doc_texts[take[i]], 1, 4000))
      }, character(1)), collapse = "\n\n"
    )

    # Call the LLM directly with a simple context wrapper
    llm <- try(service_context$llm, silent = TRUE)
    answer <- NULL
    if (!inherits(llm, "try-error") && !is.null(llm)) {
      prompt <- paste0(
        "Use the following retrieved context to answer the query.\n",
        "Context:\n", context, "\n\n",
        "Query:\n", q, "\n\nAnswer:"
      )
      # Try common methods
      answer <- try(llm$complete(prompt), silent = TRUE)
      if (inherits(answer, "try-error") || is.null(answer)) {
        answer <- try(llm$predict(prompt), silent = TRUE)
      }
      if (inherits(answer, "try-error") || is.null(answer)) {
        # Last resort: coerce to character
        answer <- ""
      }
      if (is.list(answer) && !is.null(answer$text)) answer <- answer$text
      answer <- as.character(answer)
    } else {
      answer <- ""
    }

    # Build source_nodes in llama-index-like shape
    src <- lapply(seq_along(take), function(i){
      list(id_ = paste0("doc_", take[i]), text = doc_texts[take[i]], score = as.numeric(norm[i]))
    })

    list(response = answer, source_nodes = src)
  }

  list(query_fn = query_fn)
}

#' @noRd
resolve_retriever_engine <- function(name, llama_index, documents, service_context,
                                     similarity_top_k, response_mode, params = list())
{
  # Registered override takes precedence
  handler <- get_registered_retriever(name)
  if (is.null(handler)) {
    # Built-ins
    handler <- switch(tolower(name),
      "vector" = vector_retriever_handler,
      "bm25"   = bm25_retriever_handler,
      stop("Unknown retriever: ", name, call. = FALSE)
    )
  }
  handler(llama_index, documents, service_context, similarity_top_k, response_mode, params)
}
