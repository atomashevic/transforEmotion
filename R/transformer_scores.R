#' Sentiment Analysis Scores
#'
#' @description Uses sentiment analysis pipelines from \href{https://huggingface.co}{huggingface}
#' to compute probabilities that the text corresponds to the specified classes
#' 
#' @param text Character vector or list.
#' Text in a vector or list data format
#' 
#' @param classes Character vector.
#' Classes to score the text
#' 
#' @param multiple_classes Boolean.
#' Whether the text can belong to multiple true classes.
#' Defaults to \code{FALSE}.
#' Set to \code{TRUE} to get scores with multiple classes
#' 
#' @param transformer Character.
#' Specific zero-shot sentiment analysis transformer
#' to be used. Default options:
#' 
#' \itemize{
#' 
#' \item{\code{"cross-encoder-distilroberta"}}
#' {Uses \href{https://huggingface.co/cross-encoder/nli-distilroberta-base}{Cross-Encoder's Natural Language Interface DistilRoBERTa Base}
#' zero-shot classification model trained on the
#' \href{https://nlp.stanford.edu/projects/snli/}{Stanford Natural Language Inference}
#' (SNLI) corpus and 
#' \href{https://huggingface.co/datasets/multi_nli}{MultiNLI} datasets}
#' 
#' \item{\code{"facebook-bart"}}
#' {Uses \href{https://huggingface.co/facebook/bart-large-mnli}{Facebook's BART Large}
#' zero-shot classification model trained on the
#' \href{https://huggingface.co/datasets/multi_nli}{Multi-Genre Natural Language
#' Inference} (MultiNLI) dataset}
#' 
#' }
#' 
#' Defaults to \code{"cross-encoder-distilroberta"}
#' 
#' Also allows any zero-shot classification models with a pipeline
#' from \href{https://huggingface.co/models?pipeline_tag=zero-shot-classification}{huggingface}
#' to be used by using the specified name (e.g., \code{"typeform/distilbert-base-uncased-mnli"}; see Examples)
#' 
#' @param preprocess Boolean.
#' Should basic preprocessing be applied?
#' Includes making lowercase, keeping only alphanumeric characters,
#' removing escape characters, removing repeated characters,
#' and removing white space.
#' Defaults to \code{FALSE}.
#' Transformers generally are OK without preprocessing and handle
#' many of these functions internally. Set to \code{TRUE}
#' to be extra cautious
#' 
#' @param path_to_python Character.
#' Path to specify where "python.exe" is located on your computer.
#' Defaults to \code{NULL}, which will use \code{\link[reticulate]{py_available}}
#' to find available Python or Anaconda
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
#' @return Returns probabilities for the text classes
#' 
#' @details This function requires that you have both Python and the 
#' "transformers" module installed on your computer. For help installing Python 
#' and Python modules, see \code{browseVignettes("transforEmotion")} and click on
#' the "Python Setup" vignette.
#' 
#' Once both Python and the "transformers" module are installed, the
#' function will automatically download the necessary model to compute the
#' scores. 
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
#' # Cross-Encoder DistilRoBERTa
#' transformer_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  )
#')
#' 
#' # Facebook BART Large
#' transformer_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  transformer = "facebook-bart"
#')
#' 
#' # Directly from huggingface: typeform/distilbert-base-uncased-mnli
#' transformer_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  transformer = "typeform/distilbert-base-uncased-mnli"
#')
#' }
#' 
#' @references
#' # BART
#' Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019).
#' Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.
#' \emph{arXiv preprint arXiv:1910.13461}.
#' 
#' # RoBERTa
#' Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019).
#' Roberta: A robustly optimized bert pretraining approach.
#' \emph{arXiv preprint arXiv:1907.11692}.
#' 
#' Yin, W., Hay, J., & Roth, D. (2019).
#' Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
#' \emph{arXiv preprint arXiv:1909.00161}.
#' 
#' @export
#'
# Transformer Scores
# Updated 05.03.2022
transformer_scores <- function(
  text, classes,
  multiple_classes = FALSE,
  transformer = c(
    "cross-encoder-distilroberta",
    "facebook-bart"
  ),
  preprocess = FALSE,
  path_to_python = NULL,
  keep_in_env = TRUE,
  envir = 1
)
{
  
  # Check that input of 'text' argument is in the
  # appropriate format for the analysis
  non_text_warning(text) # see utils-transforEmotion.R for function
  
  # Check for classes
  if(missing(classes)){
    stop("Classes to classify text must be specified using the 'classes' argument (e.g., `classes = c(\"positive\", \"negative\")`)\n")
  }
  
  # Check for transformer
  if(missing(transformer)){
    transformer <- "cross-encoder-distilroberta"
  }
  
  # Check for multiple transformers
  if(length(transformer) > 1){
    stop("Only one transformer model can be used at a time.\n\nSelect one of the default models or select a model from huggingface: <https://huggingface.co/models?pipeline_tag=zero-shot-classification>\n")
  }
  
  # Check for classifiers in environment
  if(exists(transformer, envir = as.environment(envir))){
    classifier <- get(transformer, envir = as.environment(envir))
  }else{
    
    # Setup Python
    path <- python_setup(path_to_python)
    
    # If path is NULL, error
    if(is.null(path)){
      return(NULL)
    }
    
    # Check if 'transformers' module is available
    if(!reticulate::py_module_available("transformers")){
      message("'transformers' module is not available.\n\nPlease install in Python: `pip install transformers`\n\nFor help, see the \"Python Setup\" vignette in `browseVignettes(\"transforEmotion\")`\n")
    }else{
      
      # Check for 'transformers' module in environment
      if(exists("transformers", envir = as.environment(envir))){
        transformers <- get("transformers", envir = as.environment(envir))
      }else{
        
        # Import 'transformers' module
        message("Importing transformers module...")
        transformers <- reticulate::import("transformers")
        
      }
        
    }
    
    # Check for custom transformer
    if(transformer %in% c(
      "cross-encoder-distilroberta", "facebook-bart"
    )){
      
      # Load pipeline
      classifier <- switch(
        transformer,
        "cross-encoder-distilroberta" = transformers$pipeline("zero-shot-classification", model = "cross-encoder/nli-distilroberta-base"),
        "facebook-bart" = transformers$pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
      
      )
    
    }else{
      
      # Custom pipeline from huggingface
      # Try to catch non-existing pipelines
      pipeline_catch <- try(
        classifier <- transformers$pipeline("zero-shot-classification", model = transformer),
        silent = TRUE
      )
      
      # Errors
      if(any(class(pipeline_catch) == "try-error")){
        
        # Model exists but no pipeline
        if(isTRUE(grepl("Tokenizer class", pipeline_catch))){
          
          stop(
            paste(
              "Transformer model '",
              transformer,
              "' exists but does not have a working pipeline yet.\n\nTry a default model or select a model from huggingface: <https://huggingface.co/models?pipeline_tag=zero-shot-classification>\n",
              sep = ""
            )
          )
          
        }else{
          stop(pipeline_catch)
        }
        
      }
      
    }

  }

  # Load into environment
  if(isTRUE(keep_in_env)){
    
    # Keep transformer module in environment
    assign(
      x = "transformers",
      value = transformers,
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
  message("Obtaining scores...")
  
  # Apply through text
  scores <- pbapply::pblapply(text, function(x){
    
    # Classify
    scores <- classifier(
      x, classes, multi_label = multiple_classes
    )
    
    # Re-organize output
    names(scores$scores) <- scores$labels
    scores$labels <- NULL
    
    # Return
    return(scores$scores)
    
  })
  
  # Reorder classes to original input
  scores <- lapply(scores, function(x){
    x[classes]
  })
  
  # Rename lists
  if(!is.list(text)){
    names(scores) <- text
  }else{
    names(scores) <- unlist(text)
  }
  
  # Return scores
  return(scores)
  
}
