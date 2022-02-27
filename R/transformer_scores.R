#' Sentiment Analysis Scores from Facebook BART Large
#'
#' @description Checkpoint for \href{https://huggingface.co/facebook/bart-large-mnli}{Facebook's BART Large} trained on the
#' Multi-Genre Natural Language Inference \href{https://huggingface.co/datasets/multi_nli}{MultiNLI} dataset. Scores represent
#' the probabilities that the text corresponds to the specified classes
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
#' and Python modules, see \code{browseVignettes("emoxicon")} and click on
#' HTML for the "Python Setup" vignette.
#' 
#' Once both Python and the "transformers" module are installed, the
#' function will automatically download the necessary model to compute the
#' scores. 
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @examples
#' # Example text 
#' text <- neo_ipip_extraversion$friendliness
#' 
#' \dontrun{
#' transformer_scores(
#'   text = text,
#'   classes = c("sociable", "warm", "assertive", "positive"),
#'   multiple_classes = FALSE
#' )
#' }
#' 
#' @references
#' Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019).
#' Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.
#' \emph{arXiv preprint arXiv:1910.13461}.
#' 
#' Yin, W., Hay, J., & Roth, D. (2019).
#' Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
#' \emph{arXiv preprint arXiv:1909.00161}.
#' 
#' @export
#'
# Transformer Scores
# Updated 27.02.2022
transformer_scores <- function(
  text, classes,
  multiple_classes = FALSE,
  path_to_python = NULL,
  keep_in_env = TRUE,
  envir = 1
)
{
  # Check that input is text
  if(!is.character(text)){
    stop("Text expected for input into 'text' argument")
  }
  
  # Check for classes
  if(missing(classes)){
    stop("Classes to classify text must be specified using the 'classes' argument")
  }

  # Check for classifier in environment
  if(exists("classifier", envir = globalenv())){
    classifier <- get("classifier", envir = as.environment(envir))
  }else{
    
    # Setup Python
    path <- python_setup(path_to_python)
    
    # If path is NULL, error
    if(is.null(path)){
      return(NULL)
    }
    
    # Check if 'transformers' module is available
    if(!reticulate::py_module_available("transformers")){
      message("'transformers' module is not available.\n\nPlease install in Python: `pip install transformers`")
    }else{
      # Import transformers
      message("Importing transformers...")
      transformers <- reticulate::import("transformers")
    }
    
    # Load pipeline
    classifier <- transformers$pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
  
  }

  # Load into environment
  if(isTRUE(keep_in_env)){
    assign(
      x = "classifier",
      value = classifier,
      envir = as.environment(envir)
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
