#' Retrieval-augmented Generation (RAG)
#'
#' @description Performs retrieval-augmented generation using {llama-index}
#'
#' Currently limited to the TinyLLAMA model
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
#' Defaults to \code{"TinyLLAMA"}.
#' \code{"LLAMA-2"} is superior in its output but is substantially
#' more difficult to install on Mac and Windows because of the
#' installation of the {llama-cpp-python} module (Linux has no issues).
#' For now, \code{"TinyLLAMA"} is recommended for ease of use
#'
#' @param prompt Character (length = 1).
#' Prompt to feed into TinyLLAMA.
#' Defaults to \code{"You are an expert at extracting emotional themes across many texts"}
#'
#' @param query Character.
#' The query you'd like to know from the documents.
#' Required with no default
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
#' \item{40-60 ---}{Comprehensive search across all texts}
#'
#' \item{20-40 ---}{Exploratory with good trade-off between comprehensive and speed}
#'
#' \item{5-15 ---}{Focused search that should give generally good results}
#'
#' }
#'
#' These values depend on the number and quality of texts. Adjust as necessary
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
#' @return Returns response from TinyLLAMA
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
#'  query = "What themes are prevelant across the text?",
#'  response_mode = "tree_summarize",
#'  similarity_top_k = 5
#')}
#'
#' @export
#'
# Retrieval-augmented generation
# Updated 23.01.2024
rag <- function(
    text = NULL, path = NULL,
    transformer = c("LLAMA-2", "TinyLLAMA"),
    prompt = "You are an expert at extracting themes across many texts",
    query, response_mode = c(
      "accumulate", "compact", "refine",
      "simple_summarize", "tree_summarize"
    ), similarity_top_k = 5,
    keep_in_env = TRUE, envir = 1, progress = TRUE
)
{



  # Check that input of 'text' argument is in the appropriate format for the analysis
  non_text_warning(text) # see utils-transforEmotion.R for function

  # Check that 'text' or 'path' are set
  if(is.null(text) & is.null(path)){
    stop("Argument 'text' or 'path' must be provided.", call. = FALSE)
  }

  # Check for 'transformer'
  if(missing(transformer)){
    transformer <- "tinyllama"
  }else{transformer <- tolower(match.arg(transformer))}

  # Check for 'query'
  if(missing(query)){
    stop("A 'query' must be provided")
  }

  # Set default for 'response_mode'
  if(missing(response_mode)){
    response_mode <- "tree_summarize"
  }else{response_mode <- match.arg(response_mode)}

  # Check for classifiers in environment
  if(!exists("llama_index", envir = as.environment(envir))){

    # Run setup for modules
    setup_modules()

    # Import 'llama-index'
    message("Importing llama-index module...")
    llama_index <- reticulate::import("llama_index")

    # Set service context
    if(transformer == "tinyllama"){

      service_context <- llama_index$ServiceContext$from_defaults(
        llm = llama_index$llms$HuggingFaceLLM(
          model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          query_wrapper_prompt = llama_index$PromptTemplate(
            paste0(
              "<|system|>\n", prompt,
              "</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"
            )
          ), device_map = "auto"
        ),
        embed_model = "local:BAAI/bge-small-en-v1.5"
      )

    }else if(transformer == "llama-2"){

      # Check for {llama-cpp-python} install
      if(!"llama-cpp-python" %in% reticulate::py_list_packages(envname = "transforEmotion")$package){

        # Get operating system
        OS <- system.check$OS

        # Check for operating system
        if(OS == "linux"){

          # Should be good to go...
          reticulate::conda_install(
            envname = "transforEmotion",
            packages = "llama-cpp-python",
            pip = TRUE
          )

        }else{

          # Try it out...
          install_try <- try(
            reticulate::conda_install(
              envname = "transforEmotion",
              packages = "llama-cpp-python",
              pip = TRUE
            ), silent = TRUE
          )

          # Catch the error
          if(is(install_try, "try-error")){

            # Send error on how to install
            if(OS == "windows"){

              stop(
                paste0(
                  "{llama-cpp-python} failed installation. ",
                  "Follow these instructions and try again:\n\n",
                  "https://llama-cpp-python.readthedocs.io/"
                ), call. = FALSE
              )

            }else{ # Mac

              stop(
                paste0(
                  "{llama-cpp-python} failed installation. ",
                  "Follow these instructions and try again:\n\n",
                  "https://llama-cpp-python.readthedocs.io/"
                ), call. = FALSE
              )

            }

          }

        }

      }

      # Set up LLAMA-2
      service_context <- llama_index$ServiceContext$from_defaults(
        embed_model = "local", llm = "local"
      )

    }

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
  if(response_mode == "tree_summarize"){
    engine <- index$as_query_engine(
      similarity_top_k = similarity_top_k,
      response_mode = "tree_summarize",
      service_context = service_context
    )
  }else{
    engine <- index$as_query_engine(
      response_mode = response_mode,
      service_context = service_context
    )
  }

  # Get query
  extracted_query <- engine$query(query)

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

  # Send response
  cat(trimws(extracted_query$response))

}
