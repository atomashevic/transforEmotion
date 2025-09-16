#' Sentiment Analysis Scores
#'
#' @description Uses sentiment analysis pipelines from \href{https://huggingface.co}{huggingface}
#' to compute probabilities that the text corresponds to the specified classes
#'
#' @param text Character vector or list.
#' Text in a vector or list data format
#'
#' @param comparison_text Character vector or list.
#' Text in a vector or list data format
#'
#' @param transformer Character.
#' Specific sentence similarity transformer
#' to be used.
#' Defaults to \code{"all_minilm_l6"} (see \href{https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2}{huggingface})
#'
#' Also allows any sentence similarity models with a pipeline
#' from \href{https://huggingface.co/models?pipeline_tag=sentence-similarity}{huggingface}
#' to be used by using the specified name (e.g., \code{"typeform/distilbert-base-uncased-mnli"}; see Examples)
#'
#' @param device Character.
#' Whether to use CPU or GPU for inference.
#' Defaults to \code{"auto"} which will use
#' GPU over CPU (if CUDA-capable GPU is setup).
#' Set to \code{"cpu"} to perform over CPU
#'
#' @param preprocess Boolean.
#' Should basic preprocessing be applied?
#' Includes making lowercase, keeping only alphanumeric characters,
#' removing escape characters, removing repeated characters,
#' and removing white space.
#' Defaults to \code{FALSE}.
#' Transformers generally are OK without preprocessing and handle
#' many of these functions internally, so setting to \code{TRUE}
#' will not change performance much
#'
#' @param keep_in_env Boolean.
#' Whether the classifier should be kept in your global environment.
#' Defaults to \code{TRUE}.
#' By keeping the classifier in your environment, you can skip
#' re-loading the classifier every time you run this function.
#' \code{TRUE} is recommended
#'
#' @param envir Numeric.
#' Environment for the classifier to be saved for repeated use.
#' Defaults to the global environment
#'
#' @return Returns a \emph{n} x \emph{m} similarity matrix where \emph{n} is length of \code{text} and \emph{m} is the length of \code{comparison_text}
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
#' # Example with defaults
#' sentence_similarity(
#'  text = text, comparison_text = text
#' )
#'
#' # Example with model from 'sentence-transformers'
#' sentence_similarity(
#'  text = text, comparison_text = text,
#'  transformer = "sentence-transformers/all-mpnet-base-v2"
#' )
#'
#'}
#'
#' @export
#'
# Sentence Similarity
# Updated 02.08.2024
sentence_similarity <- function(
    text, comparison_text,
    transformer = c("all_minilm_l6"),
    device = c("auto", "cpu", "cuda"),
    preprocess = FALSE, keep_in_env = TRUE, envir = 1
)
{
  # Ensure reticulate uses the transforEmotion conda environment
  ensure_te_py_env()

  # Check that input of 'text' argument is in the
  # appropriate format for the analysis
  non_text_warning(text) # see utils-transforEmotion.R for function

  # Check for comparison text
  if(missing(comparison_text)){
    stop(
      "Comparison text to compute similarity must be specified using the 'comparison_text' argument (e.g., `comparison_text = c(\"a similar sentence\", \a random sentence\")`)\n",
      call. = FALSE
    )
  }

  # Check for transformer
  if(missing(transformer)){
    transformer <- "all_minilm_l6"
  }

  # Check for multiple transformers
  if(length(transformer) > 1){
    stop("Only one transformer model can be used at a time.\n\nSelect one of the default models or select a model from huggingface: <https://huggingface.co/models?pipeline_tag=sentence-similarity>\n")
  }

  # Set device
  if(missing(device)){
    device <- "auto"
  }else{device <- tolower(match.arg(device))}

  # Check for classifiers in environment
  if(exists(transformer, envir = as.environment(envir))){
    classifier <- get(transformer, envir = as.environment(envir))
  }else{
    
    # Try to import sentence-transformers
    sentence_transformers <- try(
      reticulate::import("sentence_transformers"), 
      silent = TRUE
    )
    
    # If import fails, try setting up modules
    if(inherits(sentence_transformers, "try-error")) {
      message("Required Python modules not found. Setting up modules...")
      setup_modules()
      sentence_transformers <- reticulate::import("sentence_transformers")
    }

    # Check for custom transformer
    if(transformer %in% c("all_minilm_l6")){

      # Load pipeline
      classifier <- sentence_transformers$SentenceTransformer(
        switch(
          transformer,
          "all_minilm_l6" = "sentence-transformers/all-MiniLM-L6-v2",
          device = device
        )
      )

    }else{

      # Custom pipeline from huggingface
      # Try to catch non-existing pipelines
      pipeline_catch <- try(
        classifier <- sentence_transformers$SentenceTransformer(
          transformer, device = device
        ), silent = TRUE
      )

      # Errors
      if(is(pipeline_catch, "try-error")){

        # Model exists but no pipeline
        if(isTRUE(grepl("Tokenizer class", pipeline_catch))){

          stop(
            paste(
              "Transformer model '",
              transformer,
              "' exists but does not have a working pipeline yet.\n\nTry a default model or select a model from huggingface: <https://huggingface.co/models?pipeline_tag=zero-shot-classification>\n",
              sep = ""
            ), call. = FALSE
          )

        }else if(isTRUE(grepl("device_map", pipeline_catch))){

          # Try again without device
          pipeline_catch <- try(
            classifier <- sentence_transformers$SentenceTransformer(transformer), silent = TRUE
          )

        }else{
          stop(pipeline_catch, call. = FALSE)
        }

      }

    }

  }

  # Load into environment
  if(isTRUE(keep_in_env)){

    # Keep transformer module in environment
    assign(
      x = "sentence_transformers",
      value = sentence_transformers,
      envir = as.environment(envir)
    )

    # Keep classifier in environment
    assign(
      x = transformer,
      value = classifier,
      envir = as.environment(envir)
    )
  }

  # Basic preprocessing
  if(isTRUE(preprocess)){
    text <- preprocess_text( # Internal function. See `utils-transforEmotion`
      text,
      remove_stop = FALSE # Transformers will remove stop words
    )
  }

  # Message
  message("Obtaining similarities...")

  # Combine sentences
  sentences <- c(text, comparison_text)

  # Get embeddings
  embeddings <- classifier$encode(sentences)

  # Loop over text comparisons
  text_length <- length(text)
  comparison_length <- length(comparison_text)

  # Set up matrix
  similarity_matrix <- matrix(
    0, nrow = text_length, ncol = comparison_length,
    dimnames = list(text, comparison_text)
  )

  # Populate similarity matrix
  for(i in seq_len(text_length))
    for(j in seq_len(comparison_length)){

      # Compute cosine
      similarity_matrix[i,j] <- cosine(
        embed1 = embeddings[i,],
        embed2 = embeddings[j + text_length,]
      )

    }


  # Return similarities
  return(similarity_matrix)

}

#' @noRd
# Cosine function ----
# Updated 02.08.2024
cosine <- function(embed1, embed2)
{
  return(crossprod(embed1, embed2) / sqrt(crossprod(embed1) * crossprod(embed2)))
}
