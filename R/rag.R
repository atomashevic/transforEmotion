#' Retrieval-augmented Generation (RAG)
#'
#' @description Performs retrieval-augmented generation \{llama-index\}
#'
#' Supports multiple local LLM backends via HuggingFace and llama-index.
#'
#' @param text Character vector or list.
#' Text in a vector or list data format.
#' \code{path} will override input into \code{text}
#' Defaults to \code{NULL}
#'
#' @param path Character.
#' Path to .pdfs stored locally on your computer.
#' Defaults to \code{NULL}
#'
#' @param transformer Character.
#' Large language model to use for RAG.
#' Available models include:
#'
#' \describe{
#'   \item{"TinyLLAMA"}{Default. TinyLlama 1.1B Chat via HuggingFace.
#'   Fast and light local inference.}
#'   \item{"Gemma3-1B / Gemma3-4B"}{Google's Gemma 3 Instruct via
#'   HuggingFace: \code{google/gemma-3-1b-it},
#'   \code{google/gemma-3-4b-it}.}
#'   \item{"Qwen3-0.6B / Qwen3-1.7B"}{Qwen 3 small Instruct models via
#'   HuggingFace: \code{Qwen/Qwen3-0.6B-Instruct},
#'   \code{Qwen/Qwen3-1.7B-Instruct}.}
#'   \item{"Ministral-3B"}{Mistral's compact 3B Instruct via
#'   HuggingFace: \code{ministral/Ministral-3b-instruct}.}
#' }
#'
#' @param prompt Character (length = 1).
#' Prompt to feed into TinyLLAMA.
#' Defaults to \code{"You are an expert at extracting emotional themes
#' across many texts"}
#'
#' @param query Character.
#' The query you'd like to know from the documents.
#' Defaults to \code{prompt} if not provided
#'
#' @param response_mode Character (length = 1).
#' Different responses generated from the model.
#' See documentation
#' \href{https://docs.llamaindex.ai/en/stable/module_guides/}{here}
#'
#' Defaults to \code{"tree_summarize"}
#'
#' @param similarity_top_k Numeric (length = 1).
#' Retrieves most representative texts given the \code{query}.
#' Larger values will provide a more comprehensive response but at
#' the cost of computational efficiency; small values will provide
#' a more focused response at the cost of comprehensiveness.
#' Defaults to \code{5}.
#'
#' Values will vary based on number of texts but some suggested values might be:
#'
#' \describe{
#'
#' \item{40-60}{Comprehensive search across all texts}
#'
#' \item{20-40}{Exploratory with good trade-off between comprehensive and speed}
#'
#' \item{5-15}{Focused search that should give generally good results}
#'
#' }
#'
#' These values depend on the number and quality of texts. Adjust as necessary
#'
#' @param retriever Character (length = 1).
#' Retrieval backend: one of \code{"vector"} (default, semantic search using
#' embeddings) or \code{"bm25"} (lexical BM25 search). BM25 uses llama-index's
#' retriever when available and falls back to the Python \code{rank_bm25}
#' implementation otherwise. Scores are normalized to [0,1] for consistency.
#'
#' @param retriever_params List.
#' Optional parameters passed to the selected retriever handler. Reserved keys
#' include \code{show_progress}.
#'
#' @param output Character (length = 1).
#' Output format: one of \code{"text"}, \code{"json"}, \code{"table"},
#' or \code{"csv"}.
#' \itemize{
#'   \item \code{"text"} (default): returns a free-text response
#'   with retrieved content.
#'   \item Structured outputs (\code{"json"}/\code{"table"}/\code{"csv"})
#'   are supported ONLY for Gemma3-1B and Gemma3-4B.
#'   For other models, requests for structured outputs
#'   fall back to \code{"text"}.
#'   \item For Gemma3-1B/4B and task = \code{"sentiment"} or
#'   \code{"emotion"}, returns per-document dominant
#'   \code{label} and \code{confidence}.
#'   \item For Gemma3-1B/4B and task = \code{"general"},
#'   returns the prior schema with \code{labels},
#'   \code{confidences}, \code{intensity}, and
#'   \code{evidence_chunks}.
#' }
#'
#' @param task Character (length = 1).
#' Task hint for structured extraction: one of \code{"general"},
#' \code{"emotion"}, or \code{"sentiment"}.
#' When \code{"emotion"} or \code{"sentiment"}, the prompt constrains
#' labels to a set (see \code{labels_set}).
#'
#' @param labels_set Character vector.
#' Allowed labels for classification when \code{task != "general"}.
#' If \code{NULL}, defaults to
#' Emo8 labels for \code{task = "emotion"}
#'   (\code{c("joy","trust","fear","surprise","sadness",
#'   "disgust","anger","anticipation")}) for \code{task = "emotion"} and
#'   \code{c("positive","neutral","negative")} for \code{task = "sentiment"}.
#'
#' @param max_labels Integer (length = 1).
#' Maximum number of labels to return in structured outputs;
#' used to guide the model instruction when
#' \code{output != "text"}.
#'
#' @param global_analysis Boolean (length = 1).
#' Whether to perform analysis across all documents globally
#' (legacy behavior) or per-document (default).
#' When \code{FALSE} (default), each document is analyzed
#' individually then results are aggregated.
#' When \code{TRUE}, all documents are processed together
#' for a single global analysis.
#' Defaults to \code{FALSE}.
#'
#' @param device Character.
#' Whether to use CPU or GPU for inference.
#' Defaults to \code{"auto"} which will use
#' GPU over CPU (if CUDA-capable GPU is setup).
#' Set to \code{"cpu"} to perform over CPU
#'
#' @param temperature Numeric or NULL.
#' Overrides the LLM sampling temperature when using local HF models.
#' Recommended: 0.0–0.2 for structured/classification; 0.3–0.7 for summaries.
#'
#' @param do_sample Logical or NULL.
#' If \code{FALSE}, forces greedy decoding for maximum determinism.
#' Defaults are conservative; set explicitly for reproducibility.
#'
#' @param max_new_tokens Integer or NULL.
#' Maximum new tokens to generate. Suggested: 64–128 for label decisions; 256–512 for summaries.
#'
#' @param top_p Numeric or NULL.
#' Nucleus sampling parameter. Typical: 0.7–0.95. Use with \code{do_sample=TRUE}.
#'
#' @param keep_in_env Boolean (length = 1).
#' Whether the classifier should be kept in your global environment.
#' Defaults to \code{TRUE}.
#' By keeping the classifier in your environment, you can skip
#' re-loading the classifier every time you run this function.
#' \code{TRUE} is recommended
#'
#' @param envir Numeric (length = 1).
#' Environment for the classifier to be saved for repeated use.
#' Defaults to the global environment
#'
#' @param progress Boolean (length = 1).
#' Whether progress should be displayed.
#' Defaults to \code{TRUE}
#'
#' @return
#' For \code{output = "text"}, returns an object of class
#' \code{"rag"} with fields:
#' \code{$response} (character), \code{$content} (data.frame),
#' and \code{$document_embeddings} (matrix).
#' For \code{output = "json"}, returns a JSON \code{character(1)}
#' string matching the enforced schema.
#' For \code{output = "table"}, returns a \code{data.frame}
#' suitable for statistical analysis.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @examples
#' # Load data
#' data(neo_ipip_extraversion)
#'
#' # Example text
#' text <- neo_ipip_extraversion$friendliness[1:5]
#'
#' \dontrun{
#' rag(
#'  text = text,
#'  query = "What themes are prevalent across the text?",
#'  response_mode = "tree_summarize",
#'  similarity_top_k = 5
#')
#'
#' # Structured outputs
#' rag(text = text, query = "Extract emotions", output = "json")
#' rag(text = text, query = "Extract emotions", output = "table")
#'}
#'
#' @section Data Privacy:
#'   All processing is done locally with the downloaded model,
#'   and your text is never sent to any remote server or third-party.
#'
#' @export
#'
# Retrieval-augmented generation
# Updated 17.02.2024
rag <- function(
    text = NULL, path = NULL,
    transformer = c(
      "TinyLLAMA",
      "Gemma3-1B", "Gemma3-4B",
      "Qwen3-1.7B",
      "Ministral-3B"
    ),
    prompt = "You are an expert at extracting themes across many texts",
    query, response_mode = c(
      "accumulate", "compact", "no_text",
      "refine", "simple_summarize", "tree_summarize"
    ), similarity_top_k = 5,
    retriever = c("vector", "bm25"),
    retriever_params = list(),
    output = c("text", "json", "table", "csv"),
    task = c("general", "emotion", "sentiment"),
    labels_set = NULL,
    max_labels = 5,
    global_analysis = FALSE,
    device = c("auto", "cpu", "cuda"),
    temperature = NULL, do_sample = NULL, max_new_tokens = NULL, top_p = NULL,
    keep_in_env = TRUE,
    envir = 1, progress = TRUE
)
{

  # Ensure reticulate uses the transforEmotion conda environment
  ensure_te_py_env()

  # Check that input of 'text' argument is in the appropriate format 
  # for the analysis
  if(!is.null(text)){
    non_text_warning(text) # see utils-transforEmotion.R for function
  }

  # Check that 'text' or 'path' are set
  if(is.null(text) & is.null(path)){
    stop("Argument 'text' or 'path' must be provided.", call. = FALSE)
  }

  # Check for 'transformer'
  if(missing(transformer)){
    transformer <- "tinyllama"
  }else{transformer <- tolower(match.arg(transformer))}

  # If a non-Gemma small model is requested with structured output, fallback to text with warning
  if (!transformer %in% c("gemma3-1b", "gemma3-4b") && 
      !identical(output, "text")) {
    warning("Structured outputs (json/table/csv) are supported only for Gemma3-1B and Gemma3-4B. Falling back to output = 'text'.", call. = FALSE)
    output <- "text"
  }

  # Check for 'query'
  if(missing(query)){ # set 'query' to 'prompt' if missing
    query <- prompt
  }

  # Set default for 'response_mode'
  if(missing(response_mode)){
    response_mode <- "tree_summarize"
  }else{response_mode <- match.arg(response_mode)}

  # Set output mode
  output <- match.arg(output)

  # Set task mode
  task <- match.arg(task)

  # Set retriever mode
  retriever <- match.arg(retriever)

  # Enforce: structured outputs only for Gemma3-1B / Gemma3-4B
  # transformer will be lowercased after match.arg below

  # Default label sets for emotion/sentiment tasks
  if (is.null(labels_set)) {
    if (identical(task, "emotion")) {
      labels_set <- c("joy", "trust", "fear", "surprise", 
                       "sadness", "disgust", "anger", "anticipation")
    } else if (identical(task, "sentiment")) {
      labels_set <- c("positive", "neutral", "negative")
    }
  }

  # Set default for 'device'
  if(missing(device)){
    device <- "auto"
  } else {
    device <- match.arg(device)
  }

  # Run setup for modules
  # setup_modules()

  # Check for llama_index in environment
  if(!exists("llama_index", envir = as.environment(envir))){
    
    # Try to import llama-index
    llama_index <- try(
        if("llama-index-legacy" %in% reticulate::py_list_packages()$package) {
            # {llama-index} >= 0.10.5
            reticulate::import("llama_index.legacy")
        } else {
            # {llama-index} < 0.10.5
            reticulate::import("llama_index")
        }, silent = TRUE
    )
    
    # If import fails, try setting up modules
    if(inherits(llama_index, "try-error")) {
        message("Required Python modules not found. Setting up modules...")
        setup_modules()
        
        # Try import again
        llama_index <- if("llama-index-legacy" %in% 
                        reticulate::py_list_packages()$package) {
            reticulate::import("llama_index.legacy")
        } else {
            reticulate::import("llama_index")
        }
    }
  }

  # Check for service context
  if(exists("service_context", envir = as.environment(envir))){

    # Check for service context LLM
    if(attr(service_context, which = "transformer") != transformer){
      rm(service_context, envir = as.environment(envir)); gc(verbose = FALSE)
    }

  }

  # Get service context
  if(!exists("service_context", envir = as.environment(envir))){

    # Set device (only print when progress is requested)
    device <- auto_device(device, transformer, verbose = isTRUE(progress))

    # If Gemma 3 model requested, ensure HF auth for gated repos
    if (transformer %in% c("gemma3-1b", "gemma3-4b")) {
      repo_id <- switch(
        transformer,
        "gemma3-1b"   = "google/gemma-3-1b-it",
        "gemma3-4b"   = "google/gemma-3-4b-it"
      )
      if (exists("ensure_hf_auth_for_gemma", mode = "function")) {
        ensure_hf_auth_for_gemma(interactive_ok = TRUE, repo_id = repo_id)
      } else {
        warning(
          "ensure_hf_auth_for_gemma() not found; proceeding without ",
        "explicit gated repo auth. Set HF_TOKEN env var if downloads fail.",
          call. = FALSE
        )
      }
    }

    # Set up service context (restricted model set)
    service_context <- switch(
      transformer,
      "tinyllama" = setup_tinyllama(llama_index, prompt, device,
        temperature = temperature, do_sample = do_sample,
        max_new_tokens = max_new_tokens, top_p = top_p
      ),
      # Gemma 3 (HuggingFace Instruct variants)
      "gemma3-1b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "google/gemma-3-1b-it", 
        tokenizer_name = "google/gemma-3-1b-it",
        context_window = 32000L,
        temperature = temperature, do_sample = do_sample,
        max_new_tokens = max_new_tokens, top_p = top_p
      ),
      "gemma3-4b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "google/gemma-3-4b-it", 
        tokenizer_name = "google/gemma-3-4b-it",
        context_window = 128000L,
        temperature = temperature, do_sample = do_sample,
        max_new_tokens = max_new_tokens, top_p = top_p
      ),
      "qwen3-1.7b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "Qwen/Qwen3-1.7B-Instruct", 
        tokenizer_name = "Qwen/Qwen3-1.7B-Instruct",
        context_window = 32000L,
        temperature = temperature, do_sample = do_sample,
        max_new_tokens = max_new_tokens, top_p = top_p
      ),
      # Ministral 3B (HuggingFace Instruct)
      "ministral-3b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "ministral/Ministral-3b-instruct", 
        tokenizer_name = "ministral/Ministral-3b-instruct",
        context_window = 32000L,
        temperature = temperature, do_sample = do_sample,
        max_new_tokens = max_new_tokens, top_p = top_p
      ),
      stop(paste0("'", transformer, "' not found"), call. = FALSE)
    )

  }

  # Add transformer attribute to `service_context`
  attr(service_context, which = "transformer") <- transformer

  # Load into environment
  if(isTRUE(keep_in_env)){

    # Keep llama-index module in environment
    assign(
      x = "llama_index",
      value = llama_index,
      envir = as.environment(envir)
    )

    # Keep service_context in the environment
    assign(
      x = "service_context",
      value = service_context,
      envir = as.environment(envir)
    )

  }

  # Depending on where documents are, load them
  if(!is.null(path)){

    # Set documents
    documents <- llama_index$SimpleDirectoryReader(path)$load_data()

  }else if(!is.null(text)){

    # Set documents
    documents <- lapply(
      text, function(x){
        llama_index$Document(text = x)
      }
    )

  }

  # Force per-document analysis for sentiment/emotion regardless of 
  # global_analysis
  if (task %in% c("emotion", "sentiment")) {
    global_analysis <- FALSE
  }

  # Build query (structured when output != "text")
  built_query <- if (identical(output, "text")) {
    query
  } else {
    # Task-specific minimal JSON requirement for single dominant label
    if (task %in% c("emotion", "sentiment")) {
      allowed <- if (is.null(labels_set)) {
        if (identical(task, "emotion")) {
          c("joy", "trust", "fear", "surprise", "sadness", 
            "disgust", "anger", "anticipation")
        } else {
          c("positive", "neutral", "negative")
        }
      } else labels_set
      paste0(
        query, "\n\n",
        "Classify the text into EXACTLY ONE label from this set ",
        "(lowercase exact match): ",
        "[", paste(tolower(allowed), collapse = ", "), "]. ",
        "Output ONLY a single JSON object with exactly these keys: ",
        '{"label": "<one_of_allowed>", "confidence": 0.0}', 
        " where confidence is numeric in [0,1]. ",
        "No markdown, no extra text."
      )
    } else {
      # General task: keep existing richer JSON contract
      paste0(
        query, "\n\n",
        "Output strictly as a single JSON object with ",
        "EXACTLY these keys: {",
        "\"labels\": [string,...], ",
        "\"confidences\": [number 0..1,...], ",
        "\"intensity\": number 0..1, ",
        "\"evidence_chunks\": [ {\"doc_id\": string, ",
        "\"span\": string, \"score\": number } , ... ]",
        "}. Confidences must be numeric between 0 and 1. ",
        "Return exactly ONE JSON object only - no markdown ",
        "fences, no commentary."
      )
    }
  }

  # Start time
  start <- Sys.time()

  # Process documents based on global_analysis parameter
  if (isTRUE(global_analysis)) {
    
    # Global analysis (legacy behavior) - process all documents together
    if (progress) message("Indexing documents...")
    
    # Build retrieval engine based on selected retriever
    engine <- resolve_retriever_engine(
      name = retriever,
      llama_index = llama_index,
      documents = documents,
      service_context = service_context,
      similarity_top_k = similarity_top_k,
      response_mode = response_mode,
      params = list(show_progress = progress)
    )

    # Send message to user
    if (progress) message("Querying...", appendLF = FALSE)

    # Get query
    extracted_query <- run_query(engine, built_query)

    # Stop time
    if (progress) message(paste0(" elapsed: ", round(Sys.time() - start), "s"))

    # Organize Python output
    result <- list(
      response = if (identical(output, "text")) {
        response_cleanup(extracted_query$response, transformer = transformer)
      } else {
        trimws(extracted_query$response)
      },
      content = content_cleanup(extracted_query$source_nodes),
      document_embeddings = matrix(nrow = 0, ncol = 0)
    )
    
  } else {
    
    # Per-document analysis (default behavior)
    if (progress) message("Analyzing documents individually...")
    
    # Detailed progress (off by default; enable via retriever_params$verbose = TRUE)
    verbose_progress <- isTRUE(retriever_params$verbose)
    
    # Initialize containers for aggregated results
    all_responses <- character()
    all_content <- list()
    all_embeddings <- list()
    
    # Process each document individually
    for (i in seq_along(documents)) {
      
      if (isTRUE(progress) && isTRUE(verbose_progress)) {
        message(paste0("Processing document ", i, "/", 
                     length(documents), "..."))
      }
      
      # Build retrieval engine for this document
      doc_engine <- resolve_retriever_engine(
        name = retriever,
        llama_index = llama_index,
        documents = list(documents[[i]]),
        service_context = service_context,
        similarity_top_k = min(similarity_top_k, 3), 
        response_mode = response_mode,
        params = retriever_params
      )
      
      # Query this document
      doc_query <- try({
        run_query(doc_engine, built_query)
      }, silent = TRUE)

      # If we expect structured output and parsing fails, 
      # retry with stricter JSON-only prompt
      if (!identical(output, "text") && 
          (task %in% c("emotion", "sentiment")) && 
          !inherits(doc_query, "try-error")) {
        parsed_try_single <- try(parse_single_label_json(
                              trimws(doc_query$response)), silent = TRUE)
        if (inherits(parsed_try_single, "try-error")) {
          strict_prompt <- paste0(
            "Classify the text into EXACTLY ONE label from this set (lowercase exact match): ",
            "[", paste(tolower(if (is.null(labels_set)) {
              if (identical(task, "emotion")) 
                c("joy", "trust", "fear", "surprise", 
                  "sadness", "disgust", "anger", "anticipation") 
              else 
                c("positive", "neutral", "negative")
            } else labels_set), collapse = ", "), "]. ",
            'Return ONLY a JSON object: ',
            '{"label":"<one_of_allowed>","confidence":0.0} 
            with confidence in [0,1]. No markdown. Now answer: ', 
            query
          )
          if (isTRUE(progress) && isTRUE(verbose_progress)) message("  Strict JSON retry for document ", i, "...")
          strict_res <- try(run_query(doc_engine, strict_prompt), silent = TRUE)
          if (!inherits(strict_res, "try-error")) {
            doc_query <- strict_res
          }
        }
      }
      
      # Handle errors gracefully
      if (inherits(doc_query, "try-error")) {
        warning(paste0("Failed to process document ", i, ": ", 
                     as.character(doc_query)))
        next
      }
      
      # Store results
      raw_response <- doc_query$response
      
      all_responses[i] <- if (identical(output, "text")) {
        response_cleanup(raw_response, transformer = transformer)
      } else {
        trimws(raw_response)
      }
      
      # Clean up content and add document ID
      doc_content <- content_cleanup(doc_query$source_nodes)
      if (nrow(doc_content) > 0) {
        doc_content$document <- paste0("doc_", i)
        all_content[[i]] <- doc_content
      }
      
      # Store embeddings placeholder (not available for BM25 and current engine abstraction)
      doc_embeddings <- NULL
      
      if (!is.null(doc_embeddings)) {
        all_embeddings[[i]] <- doc_embeddings
      }
    }
    
    # Stop time
    if (progress) message(paste0("Per-document analysis completed in ", round(Sys.time() - start), "s"))
    
    # Aggregate results based on output type
    if (identical(output, "text")) {
      
      # For text output, combine responses with document separators
      combined_response <- paste0(
        paste0("Document ", seq_along(all_responses), ": ", all_responses, collapse = "\n\n"),
        collapse = ""
      )
      
      # Combine content
      combined_content <- if (length(all_content) > 0) {
        do.call(rbind, all_content[!sapply(all_content, is.null)])
      } else {
        data.frame(document = character(0), text = character(0), score = numeric(0))
      }
      
      # Combine embeddings
      combined_embeddings <- if (length(all_embeddings) > 0) {
        do.call(rbind, all_embeddings[!sapply(all_embeddings, is.null)])
      } else {
        matrix(nrow = 0, ncol = 0)
      }
      
      result <- list(
        response = combined_response,
        content = combined_content,
        document_embeddings = combined_embeddings
      )
      
    } else {
      
      # For structured outputs, aggregate to per-document dominant label for sent./emo.
        if (task %in% c("sentiment","emotion")) {
          # New behavior: return ONE dominant label per document with its confidence
          doc_ids <- paste0("doc_", seq_along(documents))
          per_doc <- lapply(seq_along(all_responses), function(i){
            resp <- all_responses[i]
            if (is.na(resp) || !nzchar(resp)) {
              return(list(doc_id = doc_ids[i], text = as.character(documents[[i]]$text), label = NA_character_, confidence = NA_real_))
            }
            # Try single-label parser first, fall back to array parser
            parsed_single <- try(parse_single_label_json(resp), silent = TRUE)
            if (!inherits(parsed_single, "try-error") && is.list(parsed_single)) {
              lbls <- parsed_single$label
              confs <- parsed_single$confidence
            } else {
              parsed <- try(parse_rag_json(resp, validate = TRUE), silent = TRUE)
              if (inherits(parsed, "try-error")) {
                parsed <- NULL
              }
              if (!is.null(parsed)) {
                lbls <- as.character(parsed$labels)
                confs <- as.numeric(parsed$confidences)
              } else {
                lbls <- character(0)
                confs <- numeric(0)
              }
            }
            
            if (length(lbls) > 1 && length(confs) == length(lbls)) {
              ord <- order(confs, decreasing = TRUE)
              lbls <- lbls[ord]
              confs <- confs[ord]
            }
            # Normalize label to allowed set and apply synonyms
            normalized <- normalize_label(lbls[1], task = task, labels_set = labels_set)
            lbls[1] <- normalized
            # Clamp confidence
            if (length(confs) >= 1 && is.finite(confs[1])) {
              confs[1] <- max(0, min(1, as.numeric(confs[1])))
            } else {
              confs[1] <- NA_real_
            }
            if (!is.null(labels_set) && length(lbls) > 0) {
              allowed <- tolower(labels_set)
              if (!tolower(lbls[1]) %in% allowed) {
                # If still out-of-set, mark as NA
                lbls[1] <- NA_character_
              }
            }
            list(
              doc_id = doc_ids[i],
              label = ifelse(length(lbls) >= 1, lbls[1], NA_character_),
              confidence = ifelse(length(confs) >= 1, as.numeric(confs[1]), NA_real_)
            )
          })
          per_doc_df <- do.call(rbind, lapply(per_doc, as.data.frame, stringsAsFactors = FALSE))
          # Output formats
          if (identical(output, "table") || identical(output, "csv")) {
            # Return concise summary and sort by confidence (desc)
            keep <- c("doc_id","label","confidence")
            keep <- keep[keep %in% names(per_doc_df)]
            per_doc_df <- per_doc_df[, keep, drop = FALSE]
            if (nrow(per_doc_df) > 0 && "confidence" %in% names(per_doc_df)) {
              per_doc_df <- per_doc_df[order(-as.numeric(per_doc_df$confidence)), , drop = FALSE]
              rownames(per_doc_df) <- NULL
            }
            result <- per_doc_df
          } else if (identical(output, "json")) {
            result <- jsonlite::toJSON(per_doc_df, auto_unbox = TRUE, digits = NA)
          } else {
            # Should not reach here because output != text in this branch
            result <- per_doc_df
          }
        } else {
          result <- aggregate_structured_results(
            all_responses, all_content, all_embeddings, output,
            task = task, labels_set = labels_set, max_labels = max_labels
          )
        }
      
    }
    
  }

  # If output is text, keep backward-compatible return
  if (identical(output, "text")) {
    class(result) <- "rag"
    return(result)
  }

  # For per-document structured outputs, result is already processed by aggregate_structured_results
  if (!isTRUE(global_analysis)) {
    return(result)
  }

  # For global analysis structured outputs, process the single response
  # Build evidence from retrieved content
  evidence_chunks <- list()
  if (nrow(result$content) > 0) {
    for (i in seq_len(nrow(result$content))) {
      evidence_chunks[[i]] <- list(
        doc_id = as.character(result$content$document[i]),
        span = as.character(result$content$text[i]),
        score = as.numeric(result$content$score[i])
      )
    }
  }

  # Parse model response into structured fields
  # Parse model response into structured fields (first attempt)
  parsed <- try(parse_rag_json(result$response, validate = TRUE), silent = TRUE)

  # If parsing failed, attempt one strict retry asking explicitly for JSON-only
  if (inherits(parsed, "try-error")) {
    strict_prompt <- paste0(
      if (identical(task, "emotion")) "You are extracting emotions from text. " else if (identical(task, "sentiment")) "You are extracting sentiment polarity from text. " else "",
      if (!is.null(labels_set)) paste0("Use only these labels (lowercase exact match), choose up to ", max_labels, ": ", "[", paste(labels_set, collapse = ", "), "]. ") else "",
      "Return ONLY a valid JSON object with EXACTLY these keys: ",
      "{\"labels\":[string,...],\"confidences\":[number 0..1,...],\"intensity\":number 0..1,",
      "\"evidence_chunks\":[{\"doc_id\":string,\"span\":string,\"score\":number},...]}. ",
      "All numbers must be numeric (not strings like '0..1'); return exactly ONE JSON object; no markdown fences, no extra text. Now answer for this question: ", query
    )
    if (progress) message("Retrying with strict JSON prompt...", appendLF = FALSE)
    re <- engine$query(strict_prompt)
    if (progress) message(" done")
    result$response <- trimws(re$response)
    parsed <- try(parse_rag_json(result$response, validate = TRUE), silent = TRUE)
  }

  # If still no JSON, error clearly with guidance (no fallback)
  if (inherits(parsed, "try-error")) {
    snippet <- tryCatch({
      s <- trimws(result$response)
      if (!is.character(s) || length(s) == 0) s <- ""
      n <- nchar(s)
      paste0(substr(s, 1, min(200L, n)), ifelse(n > 200L, "...", ""))
    }, error = function(e) "")
    reason <- tryCatch({
      as.character(parsed)
    }, error = function(e) "")
    stop(paste0(
      "Model did not return a valid JSON object after strict retry. ",
      "Try response_mode=\"compact\", simplify the query, and ensure the model can emit structured JSON.\n",
      if (nzchar(reason)) paste0("Reason: ", reason, "\n") else "",
      "First 200 chars of response: ", snippet
    ), call. = FALSE)
  }

  # Post-process labels/confidences for emotion/sentiment tasks
  lbls <- as.character(parsed$labels)
  conf <- as.numeric(parsed$confidences)
  if (!is.null(labels_set)) {
    allowed <- tolower(labels_set)
    lbls_low <- tolower(lbls)
    keep <- lbls_low %in% allowed
    if (any(keep)) {
      lbls <- lbls[keep]
      conf <- conf[keep]
    }
  }
  # Limit to max_labels if needed
  if (length(lbls) > max_labels) {
    ord <- order(conf, decreasing = TRUE)
    take <- ord[seq_len(max_labels)]
    lbls <- lbls[take]
    conf <- conf[take]
  }
  # Normalize confidences to sum to 1 when multiple labels
  if (length(conf) > 1 && is.finite(sum(conf)) && sum(conf) > 0) {
    conf <- conf / sum(conf)
  }

  # Compose schema-only object to avoid extraneous fields
  schema_out <- list(
    labels = lbls,
    confidences = conf,
    intensity = as.numeric(parsed$intensity),
    evidence_chunks = evidence_chunks
  )

  # Validate final structure
  validate_rag_json(schema_out, error = TRUE)

  if (identical(output, "json")) {
    return(jsonlite::toJSON(schema_out, auto_unbox = TRUE, digits = NA))
  } else if (identical(output, "table")) {
    # Concise summary table: label + confidence, sorted
    df <- data.frame(
      label = as.character(schema_out$labels),
      confidence = as.numeric(schema_out$confidences),
      stringsAsFactors = FALSE
    )
    if (nrow(df) > 0) df <- df[order(-df$confidence), , drop = FALSE]
    rownames(df) <- NULL
    return(df)
  }

  # Fallback (should not reach)
  class(result) <- "rag"
  return(result)

}

# Bug checking ----
# data(neo_ipip_extraversion)
# text = neo_ipip_extraversion$friendliness[1:5]; path = NULL
# transformer = "tinyllama"
# prompt = "You are an expert at extracting themes across many texts"
# query = "Please extract the ten most prevalent emotions across the documents. Be concise and do not repeat emotions"
# response_mode = "no_text"; similarity_top_k = 5
# device = "auto"; keep_in_env = TRUE
# envir = 1; progress = TRUE

#' @exportS3Method
# S3method 'print' ----
# Updated 25.01.2024
print.rag <- function(x, ...)
{
  cat(x$response)
}

#' @exportS3Method
# S3method 'summary' ----
# Updated 25.01.2024
summary.rag <- function(object, ...)
{
  cat(object$response)
}

#' @noRd
# Aggregate structured results from per-document analysis ----
# Updated 02.09.2025
aggregate_structured_results <- function(all_responses, all_content, all_embeddings, output,
                                         task = c("general","emotion","sentiment"),
                                         labels_set = NULL, max_labels = 5)
{
  task <- match.arg(task)
  
  # Remove empty responses
  valid_responses <- all_responses[!is.na(all_responses) & nzchar(all_responses)]
  
  if (length(valid_responses) == 0) {
    # Provide placeholders to avoid downstream aggregation errors.
    if (task == "sentiment") {
      empty_result <- list(
        labels = c("neutral"),
        confidences = c(1),
        intensity = 0,
        evidence_chunks = list()
      )
    } else {
      empty_result <- list(
        labels = character(0),
        confidences = numeric(0),
        intensity = 0,
        evidence_chunks = list()
      )
    }
    if (identical(output, "json")) {
      return(jsonlite::toJSON(empty_result, auto_unbox = TRUE, digits = NA))
    } else if (identical(output, "table")) {
      return(data.frame(label = character(0), confidence = numeric(0), stringsAsFactors = FALSE))
    } else {
      return(empty_result)
    }
  }
  
  # Parse all valid JSON responses
  parsed_results <- list()
  for (i in seq_along(valid_responses)) {
    parsed <- try(parse_rag_json(valid_responses[i], validate = TRUE), silent = TRUE)
    if (!inherits(parsed, "try-error")) {
      parsed_results[[i]] <- parsed
    }
  }
  
  if (length(parsed_results) == 0) {
    stop("No valid JSON responses could be parsed from any document.", call. = FALSE)
  }
  
  # Aggregate labels and confidences across documents
  all_labels <- character()
  all_confidences <- numeric()
  all_intensities <- numeric()
  
  for (result in parsed_results) {
    all_labels <- c(all_labels, result$labels)
    all_confidences <- c(all_confidences, result$confidences)
    all_intensities <- c(all_intensities, result$intensity)
  }
  
  # Merge duplicate labels by summing confidences
  unique_labels <- unique(all_labels)
  merged_confidences <- numeric(length(unique_labels))
  for (i in seq_along(unique_labels)) {
    label <- unique_labels[i]
    label_indices <- which(all_labels == label)
    merged_confidences[i] <- sum(all_confidences[label_indices])
  }

  # Enforce allowed label set if provided: filter, synonym-map, and renormalize
  enforce_allowed <- function(lbls, confs, task, labels_set, max_labels) {
    if (is.null(labels_set) || length(lbls) == 0) return(list(labels = lbls, confs = confs))
    allowed <- tolower(labels_set)
    # Direct keep
    keep <- tolower(lbls) %in% allowed
    if (!any(keep)) {
      # Try synonym mapping
      map_syn <- function(x) {
        lx <- tolower(x)
        if (task == "sentiment") {
          if (lx %in% c("ok","okay","meh","fine","average","so-so","alright","neutral")) return("neutral")
          if (lx %in% c("good","great","excellent","awesome","amazing","positive","happy","love")) return("positive")
          if (lx %in% c("bad","terrible","awful","horrible","negative","sad","angry","disappointed")) return("negative")
          return(NA_character_)
        } else if (task == "emotion") {
          if (lx %in% c("happy","happiness","joyful","delighted")) return("joy")
          if (lx %in% c("trusting","confidence","secure")) return("trust")
          if (lx %in% c("fearful","afraid","scared","anxious","anxiety")) return("fear")
          if (lx %in% c("surprised","astonished","amazed")) return("surprise")
          if (lx %in% c("sad","sorrow","depressed","downcast")) return("sadness")
          if (lx %in% c("disgusted","gross","revolted")) return("disgust")
          if (lx %in% c("angry","rage","mad")) return("anger")
          if (lx %in% c("anticipating","eager","expectant")) return("anticipation")
          return(NA_character_)
        }
        return(NA_character_)
      }
      mapped <- vapply(lbls, map_syn, character(1))
      ok <- !is.na(mapped) & mapped %in% allowed
      if (!any(ok)) return(list(labels = character(0), confs = numeric(0)))
      lbls <- mapped[ok]; confs <- confs[ok]
    } else {
      lbls <- lbls[keep]; confs <- confs[keep]
    }
    # Deduplicate and sum
    if (length(lbls) > 1) {
      u <- unique(lbls)
      summed <- vapply(u, function(l) sum(confs[lbls == l]), numeric(1))
      lbls <- u; confs <- as.numeric(summed)
    }
    # Limit to max_labels
    if (length(lbls) > max_labels) {
      ord <- order(confs, decreasing = TRUE)
      take <- ord[seq_len(max_labels)]
      lbls <- lbls[take]; confs <- confs[take]
    }
    # Renormalize if >1 label
    s <- sum(confs)
    if (length(confs) > 1 && is.finite(s) && s > 0) confs <- confs / s
    list(labels = lbls, confs = confs)
  }
  
  # Normalize confidences to sum to 1
  if (sum(merged_confidences) > 0) {
    merged_confidences <- merged_confidences / sum(merged_confidences)
  }
  
  # Calculate aggregated intensity (weighted average by confidence)
  if (length(all_intensities) > 0 && sum(all_confidences) > 0) {
    aggregated_intensity <- sum(all_intensities * all_confidences) / sum(all_confidences)
  } else {
    aggregated_intensity <- mean(all_intensities, na.rm = TRUE)
    if (is.nan(aggregated_intensity)) aggregated_intensity <- 0
  }
  
  # Combine evidence chunks from all documents
  combined_evidence <- list()
  evidence_counter <- 1
  
  for (i in seq_along(all_content)) {
    if (!is.null(all_content[[i]]) && nrow(all_content[[i]]) > 0) {
      content_df <- all_content[[i]]
      for (j in seq_len(nrow(content_df))) {
        combined_evidence[[evidence_counter]] <- list(
          doc_id = paste0("doc_", i),
          span = as.character(content_df$text[j]),
          score = as.numeric(content_df$score[j])
        )
        evidence_counter <- evidence_counter + 1
      }
    }
  }
  
  # Apply enforcement now
  enforced <- enforce_allowed(unique_labels, merged_confidences, task, labels_set, max_labels)
  out_labels <- enforced$labels
  out_conf <- enforced$confs
  # If all filtered out, keep original merged but limited to top max_labels
  if (length(out_labels) == 0 && length(unique_labels) > 0) {
    ord <- order(merged_confidences, decreasing = TRUE)
    take <- ord[seq_len(min(max_labels, length(ord)))]
    out_labels <- unique_labels[take]
    out_conf <- merged_confidences[take]
    s <- sum(out_conf)
    if (length(out_conf) > 1 && is.finite(s) && s > 0) out_conf <- out_conf / s
  }

  # Create final aggregated result
  aggregated_result <- list(
    labels = out_labels,
    confidences = out_conf,
    intensity = aggregated_intensity,
    evidence_chunks = combined_evidence
  )
  
  # Validate the aggregated structure
  validate_rag_json(aggregated_result, error = TRUE)
  
  # Return in requested format
  if (identical(output, "json")) {
    return(jsonlite::toJSON(aggregated_result, auto_unbox = TRUE, digits = NA))
  } else if (identical(output, "table")) {
    # Concise summary table: labels + confidences, sorted
    df <- data.frame(
      label = as.character(aggregated_result$labels),
      confidence = as.numeric(aggregated_result$confidences),
      stringsAsFactors = FALSE
    )
    if (nrow(df) > 0) df <- df[order(-df$confidence), , drop = FALSE]
    rownames(df) <- NULL
    return(df)
  }
  
  # Fallback (shouldn't reach here)
  return(aggregated_result)
}

#' @noRd
# Clean up response ----
# Updated 29.01.2024
response_cleanup <- function(response, transformer)
{

  # Trim whitespace first!
  response <- trimws(response)

  # Return on switch
  return(
    switch(
      transformer,
      "llama-2" = response,
      "mistral-7b" = gsub(
        "(\\d+)", "\\\n\\1",
        gsub("\n---.*", "", response),
        perl = TRUE
      ),
      "openchat-3.5" = gsub("\n---.*", "", response),
      "orca-2" = response,
      "phi-2" = gsub("\\\n\\\n.*", "", response),
      "tinyllama" = response,
      "gemma3-1b" = response,
      "gemma3-4b" = response,
      "qwen3-1.7b" = response,
      "pleias-rag-350m" = pleias_response_cleanup(response, debug = TRUE),
      "pleias-rag-1b" = pleias_response_cleanup(response, debug = TRUE),
      response
    )
  )

}

#' @noRd
# Clean up content ----
# Updated 28.01.2024
content_cleanup <- function(content)
{

  # Get number of documents
  n_documents <- length(content)

  # Initialize data frame
  content_df <- matrix(
    data = NA, nrow = n_documents, ncol = 3,
    dimnames = list(
      NULL, c("document", "text", "score")
    )
  )

  # Loop over content
  for(i in seq_len(n_documents)){

    # Populate matrix
    content_df[i,] <- c(
      content[[i]]$id_, content[[i]]$text, content[[i]]$score
    )

  }

  # Make it a real data frame
  content_df <- as.data.frame(content_df)

  # Set proper modes
  content_df$score <- as.numeric(content_df$score)

  # Return data frame
  return(content_df)

}

#' @noRd
# Parse a simple single-label JSON: {"label": string, "confidence": number}
parse_single_label_json <- function(x)
{
  if (is.null(x) || !nzchar(x)) stop("Empty response", call. = FALSE)
  txt <- try(strip_code_fence(x), silent = TRUE)
  if (inherits(txt, "try-error")) txt <- x
  obj <- try(jsonlite::fromJSON(txt, simplifyVector = TRUE), silent = TRUE)
  if (inherits(obj, "try-error") || !is.list(obj)) stop("Not JSON", call. = FALSE)
  if (is.null(obj$label) || is.null(obj$confidence)) stop("Missing keys", call. = FALSE)
  lab <- as.character(obj$label)[1]
  conf <- suppressWarnings(as.numeric(obj$confidence)[1])
  if (!is.finite(conf)) conf <- NA_real_
  list(label = lab, confidence = conf)
}

#' @noRd
# Normalize a predicted label to the allowed set using task-specific synonyms
normalize_label <- function(label, task = c("emotion","sentiment"), labels_set = NULL)
{
  if (is.null(label) || !nzchar(label)) return(NA_character_)
  task <- match.arg(task)
  lx <- tolower(trimws(label))
  allowed <- tolower(if (is.null(labels_set)) {
    if (identical(task, "emotion")) c("joy","trust","fear","surprise","sadness","disgust","anger","anticipation") else c("positive","neutral","negative")
  } else labels_set)
  if (lx %in% allowed) return(lx)
  if (task == "sentiment") {
    if (lx %in% c("ok","okay","meh","fine","average","so-so","alright","neutral")) return("neutral")
    if (lx %in% c("good","great","excellent","awesome","amazing","positive","happy","love","like")) return("positive")
    if (lx %in% c("bad","terrible","awful","horrible","negative","sad","angry","disappointed","hate")) return("negative")
    return(NA_character_)
  } else {
    if (lx %in% c("happy","happiness","joyful","delighted")) return("joy")
    if (lx %in% c("trusting","confidence","secure","faith")) return("trust")
    if (lx %in% c("fearful","afraid","scared","anxious","anxiety","worry")) return("fear")
    if (lx %in% c("surprised","astonished","amazed","shock")) return("surprise")
    if (lx %in% c("sad","sorrow","depressed","downcast","unhappy")) return("sadness")
    if (lx %in% c("disgusted","gross","revolted","contempt")) return("disgust")
    if (lx %in% c("angry","rage","mad","annoyed")) return("anger")
    if (lx %in% c("anticipating","eager","expectant","anticipation","hope")) return("anticipation")
    return(NA_character_)
  }
}

#' @noRd
# Setup Ollama-backed models (Gemma3, Ministral)
# Tries to use llama_index llms.Ollama. Fails with a helpful message if unavailable.
setup_ollama_model <- function(llama_index, prompt, device, model, context_window)
{
  # Attempt to access Ollama LLM class
  ollama_llm <- NULL
  # Try common attribute locations in llama_index
  if (!is.null(try(llama_index$llms$Ollama, silent = TRUE))) {
    ollama_llm <- llama_index$llms$Ollama
  } else if (!is.null(try(llama_index$llms$ollama$Ollama, silent = TRUE))) {
    ollama_llm <- llama_index$llms$ollama$Ollama
  }

  if (is.null(ollama_llm)) {
    stop(
      paste0(
        "Ollama backend not available in llama-index. ",
        "Install and run Ollama (https://ollama.ai), and ensure your Python llama-index installation includes the Ollama LLM."
      ), call. = FALSE
    )
  }

  # Build ServiceContext with Ollama LLM
  sc <- llama_index$ServiceContext$from_defaults(
    llm = ollama_llm(
      model = model,
      request_timeout = as.double(120)
      # Note: system prompt support varies by version; we keep wrapper prompt in RAG query
    ),
    context_window = as.integer(context_window),
    embed_model = "local:BAAI/bge-small-en-v1.5"
  )

  return(sc)
}

#' @noRd
# Generic HuggingFace LLM setup (Gemma3/Ministral)
# Uses llama_index$llms$HuggingFaceLLM without Ollama.
setup_hf_llm <- function(llama_index, prompt, device, model_name, tokenizer_name, context_window,
                          temperature = NULL, do_sample = NULL, max_new_tokens = NULL, top_p = NULL)
{
  # Models that don't support token_type_ids in generation
  models_without_token_type_ids <- c(
    "PleIAs/Pleias-RAG-350M",
    "PleIAs/Pleias-RAG-1B",
    "meta-llama/Llama-3.2-1B"
  )
  
  # Configure tokenizer kwargs. Avoid explicit token_type_ids manipulation; prior attempts
  # triggered llama-index to inject an unused `token_type_ids` arg that raised
  # ValueError for decoder-only models (Gemma / PleIAs / Llama 3.2 1B).
  tokenizer_kwargs <- list(trust_remote_code = TRUE)

  # Optimize generation parameters for different model types
  generate_kwargs <- if (grepl("PleIAs/Pleias-RAG", model_name)) {
    # PleIAs RAG models: optimized for factual accuracy with reasoning
    list(
      temperature = as.double(0.2),  # Slightly higher for diverse reasoning
      do_sample = TRUE,
      top_p = as.double(0.9),        # Nucleus sampling for quality
      repetition_penalty = as.double(1.05)  # Light repetition penalty
    )
  } else {
    # Conservative sampling for other structured-ish outputs
    list(
      temperature = as.double(0.1), 
      do_sample = TRUE
    )
  }

  # Apply user overrides if provided
  if (!is.null(temperature)) generate_kwargs$temperature <- as.double(temperature)
  if (!is.null(do_sample))    generate_kwargs$do_sample    <- isTRUE(do_sample)
  if (!is.null(top_p))        generate_kwargs$top_p        <- as.double(top_p)
  if (!is.null(max_new_tokens)) generate_kwargs$max_new_tokens <- as.integer(max_new_tokens)

  # Build ServiceContext with HuggingFace LLM
  sc <- llama_index$ServiceContext$from_defaults(
    llm = llama_index$llms$HuggingFaceLLM(
      model_name = model_name,
      tokenizer_name = tokenizer_name,
      device_map = device,
      generate_kwargs = generate_kwargs,
      # Required for newer architectures like Gemma 3 (remote modeling code)
      tokenizer_kwargs = tokenizer_kwargs,
      model_kwargs = list(trust_remote_code = TRUE)
    ),
    context_window = as.integer(context_window),
    embed_model = "local:BAAI/bge-small-en-v1.5"
  )

  return(sc)
}

#' @noRd
# LLAMA-2 ----
# Updated 06.02.2024
setup_llama2 <- function(llama_index, prompt, device)
{

  # Check for device
  if(grepl("cuda", device)){

    # Try to setup GPU modules
    output <- try(setup_gpu_modules(), silent = TRUE)

    # If error, then switch to "cpu"
    if(is(device, "try-error")){
      device <- "cpu"
    }

  }

  # If GPU possible, try different models
  if(grepl("cuda", device)){

    # Order of models to try
    MODEL <- c("GPTQ", "AWQ")

    # Loop over and try
    for(model in MODEL){

      # Set up model
      model_name <- paste0("TheBloke/Llama-2-7B-Chat-", model)

      # Try to get and load model
      load_model <- try(
        llama_index$ServiceContext$from_defaults(
          llm = llama_index$llms$HuggingFaceLLM(
            model_name = model_name,
            tokenizer_name = model_name,
            query_wrapper_prompt = llama_index$PromptTemplate(
              paste0(
                "<|system|>\n", prompt,
                "</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
              )
            ), device_map = device,
            generate_kwargs = list(
              temperature = as.double(0.1), do_sample = TRUE
            )
          ), context_window = 8192L,
          embed_model = "local:BAAI/bge-small-en-v1.5"
        ), silent = TRUE
      )

      # Check if load model failed
      if(is(load_model, "try-error")){
        delete_transformer(gsub("/", "--",  model_name), TRUE)
      }else{ # Successful load, break out of loop
        break
      }

    }

    # If by the end, still failing, switch to CPU
    if(is(load_model, "try-error")){
      device <- "cpu"
    }

  }

  # Use CPU model
  if(device == "cpu"){
    load_model <- llama_index$ServiceContext$from_defaults(
      llm = llama_index$llms$HuggingFaceLLM(
        model_name = "TheBloke/Llama-2-7B-Chat-fp16",
        tokenizer_name = "TheBloke/Llama-2-7B-Chat-fp16",
        query_wrapper_prompt = llama_index$PromptTemplate(
          paste0(
            "<|system|>\n", prompt,
            "</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
          )
        ), device_map = device,
        generate_kwargs = list(
          temperature = as.double(0.1), do_sample = TRUE
        )
      ), context_window = 8192L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  }

  # Return model
  return(load_model)

}

#' @noRd
# Mistral-7B ----
# Updated 28.01.2024
setup_mistral <- function(llama_index, prompt, device)
{

  # Return model
  return(
    llama_index$ServiceContext$from_defaults(
      llm = llama_index$llms$HuggingFaceLLM(
        model_name = "mistralai/Mistral-7B-v0.1",
        tokenizer_name = "mistralai/Mistral-7B-v0.1",
        device_map = device,
        generate_kwargs = list(
          temperature = as.double(0.1), do_sample = TRUE,
          pad_token_id = 2L, eos_token_id = 2L
        )
      ), context_window = 8192L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  )

}

#' @noRd
# OpenChat-3.5 ----
# Updated 28.01.2024
setup_openchat <- function(llama_index, prompt, device)
{

  # Return model
  return(
    llama_index$ServiceContext$from_defaults(
      llm = llama_index$llms$HuggingFaceLLM(
        model_name = "openchat/openchat_3.5",
        tokenizer_name = "openchat/openchat_3.5",
        device_map = device,
        generate_kwargs = list(
          temperature = as.double(0.1), do_sample = TRUE
        )
      ), context_window = 8192L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  )

}

#' @noRd
# Orca-2 ----
# Updated 28.01.2024
setup_orca2 <- function(llama_index, prompt, device)
{

  # Return model
  return(
    llama_index$ServiceContext$from_defaults(
      llm = llama_index$llms$HuggingFaceLLM(
        model_name = "microsoft/Orca-2-7b",
        tokenizer_name = "microsoft/Orca-2-7b",
        device_map = device,
        generate_kwargs = list(
          temperature = as.double(0.1), do_sample = TRUE
        )
      ), context_window = 4096L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  )

}

#' @noRd
# Phi-2 ----
# Updated 28.01.2024
setup_phi2 <- function(llama_index, prompt, device)
{

  # Return model
  return(
    llama_index$ServiceContext$from_defaults(
      llm = llama_index$llms$HuggingFaceLLM(
        model_name = "microsoft/phi-2",
        tokenizer_name = "microsoft/phi-2",
        device_map = device,
        generate_kwargs = list(
          temperature = as.double(0.1), do_sample = TRUE,
          pad_token_id = 2L, eos_token_id = 2L
        )
      ), context_window = 2048L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  )

}



#' @noRd
# Clean up PleIAs RAG response (extract clean answer with citations)
# Updated 02.01.2025
pleias_response_cleanup <- function(response, debug = FALSE)
{
  # Store original for debugging
  original_response <- response
  
  # Trim whitespace first
  response <- trimws(response)
  
  if (debug && nchar(response) > 0) {
    message("Original Pleias response length: ", nchar(original_response))
    message("First 200 chars: ", substr(original_response, 1, 200), "...")
  }
  
  # If response is empty or very short, return original
  if (nchar(response) < 10) {
    if (debug) message("Response too short, returning original")
    return(original_response)
  }
  
  # PleIAs models generate structured outputs with reasoning traces
  # Be more conservative with cleanup to preserve content
  
  # Look for common patterns in PleIAs output structure, but don't be too aggressive
  if (grepl("\\*\\*Final Answer\\*\\*|\\*\\*Answer\\*\\*", response, ignore.case = TRUE)) {
    # Extract content after "**Final Answer**" or "**Answer**"
    answer_split <- strsplit(response, "\\*\\*(?:Final )?Answer\\*\\*:?", perl = TRUE)[[1]]
    if (length(answer_split) > 1) {
      response <- trimws(answer_split[2])
      if (debug) message("Extracted final answer section")
    }
  }
  
  # Very light cleanup - just normalize whitespace
  response <- gsub("\\n\\s*\\n", "\n\n", response, perl = TRUE)  # Normalize line breaks
  response <- trimws(response)
  
  # If after cleanup we have very little content, return original
  if (nchar(response) < 20) {
    if (debug) message("Cleanup removed too much, returning original")
    return(original_response)
  }
  
  if (debug) {
    message("Cleaned response length: ", nchar(response))
    message("First 200 chars of clean: ", substr(response, 1, 200), "...")
  }
  
  return(response)
}


#' @noRd
# TinyLLAMA ----
# Updated 28.01.2024
setup_tinyllama <- function(llama_index, prompt, device,
                             temperature = NULL, do_sample = NULL,
                             max_new_tokens = NULL, top_p = NULL)
{

  # Return model
  # Build generate kwargs from overrides
  gen <- list()
  if (!is.null(temperature))    gen$temperature    <- as.double(temperature)
  if (!is.null(do_sample))       gen$do_sample      <- isTRUE(do_sample)
  if (!is.null(top_p))           gen$top_p          <- as.double(top_p)
  if (!is.null(max_new_tokens))  gen$max_new_tokens <- as.integer(max_new_tokens)

  return(
    llama_index$ServiceContext$from_defaults(
      llm = llama_index$llms$HuggingFaceLLM(
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        query_wrapper_prompt = llama_index$PromptTemplate(
          paste0(
            "<|system|>\n", prompt,
            "</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
          )
        ), device_map = device,
        generate_kwargs = gen
      ), context_window = 2048L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  )

}
