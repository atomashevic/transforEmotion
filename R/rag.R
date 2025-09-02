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
#'
#' \item{"LLAMA-2"}{The largest model available (13B parameters) but also the most challenging to get up and running for Mac and Windows. Linux operating systems run smooth. The challenge comes with installing the \{llama-cpp-python\} module. Currently, we do not provide support for Mac and Windows users}
#'
#' \item{"Mistral-7B"}{Mistral's 7B parameter model that serves as a high quality but more computationally expensive (more time consuming)}
#'
#' \item{"Orca-2"}{More documentation soon...}
#'
#' \item{"Phi-2"}{More documentation soon...}
#'
#' \item{"TinyLLAMA"}{Default. A smaller, 1B parameter version of LLAMA-2 that offers fast inference with reasonable quality}
#'
#' \item{"Gemma3-270M / Gemma3-1B / Gemma3-4B"}{Google's Gemma 3 family (Instruct) via HuggingFace: \code{google/gemma-3-270m-it}, \code{google/gemma-3-1b-it}, \code{google/gemma-3-4b-it}. 270M/1B support ~32K context; 4B supports up to ~128K.}
#'
#' \item{"Ministral-3B"}{Mistral's compact 3B Instruct model via HuggingFace: \code{ministral/Ministral-3b-instruct} (~32K context)}
#'
#' }
#'
#' @param prompt Character (length = 1).
#' Prompt to feed into TinyLLAMA.
#' Defaults to \code{"You are an expert at extracting emotional themes across many texts"}
#'
#' @param query Character.
#' The query you'd like to know from the documents.
#' Defaults to \code{prompt} if not provided
#'
#' @param response_mode Character (length = 1).
#' Different responses generated from the model.
#' See documentation \href{https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/response_modes.html}{here}
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
#' @param output Character (length = 1).
#' Output format: one of \code{"text"}, \code{"json"}, or \code{"table"}.
#' \itemize{
#'   \item \code{"text"} (default): returns the existing \code{"rag"} object with free-text \code{$response} and retrieval \code{$content}.
#'   \item \code{"json"}: returns a JSON string with fields \code{labels}, \code{confidences}, \code{intensity}, and \code{evidence_chunks[{doc_id, span, score}]}, validated against the enforced schema.
#'   \item \code{"table"}: returns a long-form \code{data.frame} with columns \code{label}, \code{confidence}, \code{intensity}, \code{doc_id}, \code{span}, and \code{score}.
#' }
#'
#' @param task Character (length = 1).
#' Task hint for structured extraction: one of \code{"general"}, \code{"emotion"}, or \code{"sentiment"}.
#' When \code{"emotion"} or \code{"sentiment"}, the prompt constrains labels to a set (see \code{labels_set}).
#'
#' @param labels_set Character vector.
#' Allowed labels for classification when \code{task != "general"}. If \code{NULL}, defaults to
#' Emo8 labels for \code{task = "emotion"} (\code{c("joy","trust","fear","surprise","sadness","disgust","anger","anticipation")})
#' and \code{c("positive","neutral","negative")} for \code{task = "sentiment"}.
#'
#' @param max_labels Integer (length = 1).
#' Maximum number of labels to return in structured outputs; used to guide the model instruction when \code{output != "text"}.
#'
#' @param device Character.
#' Whether to use CPU or GPU for inference.
#' Defaults to \code{"auto"} which will use
#' GPU over CPU (if CUDA-capable GPU is setup).
#' Set to \code{"cpu"} to perform over CPU
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
#' For \code{output = "text"}, returns an object of class \code{"rag"} with fields:
#' \code{$response} (character), \code{$content} (data.frame), and \code{$document_embeddings} (matrix).
#' For \code{output = "json"}, returns a JSON \code{character(1)} string matching the enforced schema.
#' For \code{output = "table"}, returns a \code{data.frame} suitable for statistical analysis.
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
      "LLAMA-2", "Mistral-7B", "OpenChat-3.5",
      "Orca-2", "Phi-2", "TinyLLAMA",
      "Gemma3-270M", "Gemma3-1B", "Gemma3-4B",
      "Ministral-3B"
    ),
    prompt = "You are an expert at extracting themes across many texts",
    query, response_mode = c(
      "accumulate", "compact", "no_text",
      "refine", "simple_summarize", "tree_summarize"
    ), similarity_top_k = 5,
    output = c("text", "json", "table"),
    task = c("general", "emotion", "sentiment"),
    labels_set = NULL,
    max_labels = 5,
    device = c("auto", "cpu", "cuda"), keep_in_env = TRUE,
    envir = 1, progress = TRUE
)
{

  # Ensure reticulate uses the transforEmotion conda environment
  ensure_te_py_env()

  # Check that input of 'text' argument is in the appropriate format for the analysis
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

  # Default label sets for emotion/sentiment tasks
  if (is.null(labels_set)) {
    if (identical(task, "emotion")) {
      labels_set <- c("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
    } else if (identical(task, "sentiment")) {
      labels_set <- c("positive", "neutral", "negative")
    }
  }

  # Set default for 'device'
  if(missing(response_mode)){
    device <- "auto"
  }else{device <- match.arg(device)}

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
        llama_index <- if("llama-index-legacy" %in% reticulate::py_list_packages()$package) {
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

    # Set device
    device <- auto_device(device, transformer)

    # If Gemma 3 model requested, ensure HF auth for gated repos
    if (transformer %in% c("gemma3-270m", "gemma3-1b", "gemma3-4b")) {
      # Map to exact HF repo for validation
      repo_id <- switch(
        transformer,
        "gemma3-270m" = "google/gemma-3-270m-it",
        "gemma3-1b"   = "google/gemma-3-1b-it",
        "gemma3-4b"   = "google/gemma-3-4b-it"
      )
      ensure_hf_auth_for_gemma(interactive_ok = TRUE, repo_id = repo_id)
    }

    # Set up service context
    service_context <- switch(
      transformer,
      "tinyllama" = setup_tinyllama(llama_index, prompt, device),
      "llama-2" = setup_llama2(llama_index, prompt, device),
      "mistral-7b" = setup_mistral(llama_index, prompt, device),
      "openchat-3.5" = setup_openchat(llama_index, prompt, device),
      "orca-2" = setup_orca2(llama_index, prompt, device),
      "phi-2" = setup_phi2(llama_index, prompt, device),
      # Gemma 3 (HuggingFace Instruct variants)
      "gemma3-270m" = setup_hf_llm(llama_index, prompt, device,
        model_name = "google/gemma-3-270m-it", tokenizer_name = "google/gemma-3-270m-it",
        context_window = 32000L
      ),
      "gemma3-1b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "google/gemma-3-1b-it", tokenizer_name = "google/gemma-3-1b-it",
        context_window = 32000L
      ),
      "gemma3-4b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "google/gemma-3-4b-it", tokenizer_name = "google/gemma-3-4b-it",
        context_window = 128000L
      ),
      # Ministral 3B (HuggingFace Instruct)
      "ministral-3b" = setup_hf_llm(llama_index, prompt, device,
        model_name = "ministral/Ministral-3b-instruct", tokenizer_name = "ministral/Ministral-3b-instruct",
        context_window = 32000L
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

  # Send message to user
  message("Indexing documents...")

  # Set indices
  index <- llama_index$VectorStoreIndex(
    documents, service_context = service_context,
    show_progress = progress
  )

  # Set up query engine
  engine <- index$as_query_engine(
    similarity_top_k = similarity_top_k,
    response_mode = response_mode
  )

  # Send message to user
  message("Querying...", appendLF = FALSE)

  # Start time
  start <- Sys.time()

  # Build query (structured when output != "text")
  built_query <- if (identical(output, "text")) {
    query
  } else {
    # Task-specific guidance
    label_guidance <- if (!is.null(labels_set)) {
      paste0(
        "Use only these labels (lowercase exact match), choose up to ", max_labels, ": ",
        "[", paste(labels_set, collapse = ", "), "]. "
      )
    } else { "" }

    paste0(
      query, "\n\n",
      if (identical(task, "emotion")) "You are extracting emotions from text. " else if (identical(task, "sentiment")) "You are extracting sentiment polarity from text. " else "",
      label_guidance,
      "Output strictly as a single JSON object with EXACTLY these keys: ",
      "{",
      "\"labels\": [string,...], ",
      "\"confidences\": [number 0..1,...], ",
      "\"intensity\": number 0..1, ",
      "\"evidence_chunks\": [ {\"doc_id\": string, \"span\": string, \"score\": number } , ... ]",
      "}. ",
      "Confidences must be numeric between 0 and 1 (no strings like '0..1'); if multiple labels are returned, normalize to sum to 1. ",
      "Return exactly ONE JSON object only — no markdown fences, no extra numbers, no commentary."
    )
  }

  # Get query
  extracted_query <- engine$query(built_query)

  # Stop time
  message(paste0(" elapsed: ", round(Sys.time() - start), "s"))

  # Organize Python output
  result <- list(
    response = if (identical(output, "text")) {
      response_cleanup(extracted_query$response, transformer = transformer)
    } else {
      trimws(extracted_query$response)
    },
    content = content_cleanup(extracted_query$source_nodes),
    document_embeddings = do.call(
      rbind, silent_call(index$vector_store$to_dict()$embedding_dict)
    )
  )

  # If output is text, keep backward-compatible return
  if (identical(output, "text")) {
    class(result) <- "rag"
    return(result)
  }

  # For structured outputs, parse and validate
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
    message("Retrying with strict JSON prompt...", appendLF = FALSE)
    re <- engine$query(strict_prompt)
    message(" done")
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
    return(as_rag_table(schema_out, validate = TRUE))
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
      "tinyllama" = response
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
setup_hf_llm <- function(llama_index, prompt, device, model_name, tokenizer_name, context_window)
{
  # Build ServiceContext with HuggingFace LLM
  sc <- llama_index$ServiceContext$from_defaults(
    llm = llama_index$llms$HuggingFaceLLM(
      model_name = model_name,
      tokenizer_name = tokenizer_name,
      device_map = device,
      # Conservative sampling for structured-ish outputs
      generate_kwargs = list(
        temperature = as.double(0.1), do_sample = TRUE
      ),
      # Required for newer architectures like Gemma 3 (remote modeling code)
      tokenizer_kwargs = list(trust_remote_code = TRUE),
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
# TinyLLAMA ----
# Updated 28.01.2024
setup_tinyllama <- function(llama_index, prompt, device)
{

  # Return model
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
        ), device_map = device
      ), context_window = 2048L,
      embed_model = "local:BAAI/bge-small-en-v1.5"
    )
  )

}
