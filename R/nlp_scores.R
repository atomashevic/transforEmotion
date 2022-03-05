#' Natural Language Processing Scores
#'
#' @description Natural Language Processing using word embeddings to compute
#' semantic similarities (cosine) of text and specified classes
#' 
#' @param text Character vector or list.
#' Text in a vector or list data format
#' 
#' @param classes Character vector.
#' Classes to score the text
#' 
#' @param semantic_space Character vector.
#' The semantic space used to compute the distances between words
#' (more than one allowed). Here's a list of the semantic spaces:
#' 
#' \itemize{
#' 
#' \item{\code{"baroni"}}
#' {Combination of British National Corpus, ukWaC corpus, and a 2009
#' Wikipedia dump. Space created using continuous bag of words algorithm
#' using a context window size of 11 words (5 left and right)
#' and 400 dimensions. Best word2vec model according to
#' Baroni, Dinu, & Kruszewski (2014)}
#' 
#' \item{\code{"cbow"}}
#' {Combination of British National Corpus, ukWaC corpus, and a 2009
#' Wikipedia dump. Space created using continuous bag of words algorithm with
#' a context window size of 5 (2 left and right) and 300 dimensions}
#' 
#' \item{\code{"cbow_ukwac"}}
#' {ukWaC corpus with the continuous bag of words algorithm with
#' a context window size of 5 (2 left and right) and 400 dimensions}
#' 
#' \item{\code{"en100"}}
#' {Combination of British National Corpus, ukWaC corpus, and a 2009
#' Wikipedia dump. 100,000 most frequent words. Uses moving window model
#' with a size of 5 (2 to the left and right). Positive pointwise mutual
#' information and singular value decomposition was used to reduce the
#' space to 300 dimensions}
#' 
#' \item{\code{"glove"}}
#' {\href{https://dumps.wikimedia.org/}{Wikipedia 2014 dump} and \href{https://catalog.ldc.upenn.edu/LDC2011T07}{Gigaword 5} with 400,000
#' words (300 dimensions). Uses co-occurrence of words in text
#' documents (uses cosine similarity)}
#' 
#' \item{\code{"tasa"}}
#' {Latent Semantic Analysis space from TASA corpus all (300 dimensions).
#' Uses co-occurrence of words in text documents (uses cosine similarity)}
#' 
#' }
#' 
#' @param preprocess Boolean.
#' Should basic preprocessing be applied?
#' Includes making lowercase, keeping only alphanumeric characters,
#' removing escape characters, removing repeated characters,
#' and removing white space.
#' Defaults to \code{TRUE}
#' 
#' @param remove_stop Boolean.
#' Should \code{\link[transforEmotion]{stop_words}}
#' be removed?
#' Defaults to \code{TRUE}
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
#' @return Returns semantic distances for the text classes
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
#' # GloVe
#' nlp_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  )
#' )
#' 
#' # Baroni
#' nlp_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  semantic_space = "baroni"
#' )
#'  
#' # CBOW
#' nlp_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  semantic_space = "cbow"
#' )
#' 
#' # CBOW + ukWaC
#' nlp_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  semantic_space = "cbow_ukwac"
#' )
#' 
#' # en100
#' nlp_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  semantic_space = "en100"
#' )
#' 
#' # tasa
#' nlp_scores(
#'  text = text,
#'  classes = c(
#'    "friendly", "gregarious", "assertive",
#'    "active", "excitement", "cheerful"
#'  ),
#'  semantic_space = "tasa"
#' )
#' }
#' 
#' @references
#' Baroni, M., Dinu, G., & Kruszewski, G. (2014).
#' Don't count, predict! a systematic comparison of context-counting vs. context-predicting semantic vectors.
#' In \emph{Proceedings of the 52nd annual meting of the association for computational linguistics} (pp. 238-247).
#' 
#' Landauer, T.K., & Dumais, S.T. (1997).
#' A solution to Plato's problem: The Latent Semantic Analysis theory of acquisition, induction and representation of knowledge.
#' \emph{Psychological Review}, \emph{104}, 211-240.
#' 
#' Pennington, J., Socher, R., & Manning, C. D. (2014).
#' GloVe: Global vectors for word representation.
#' In \emph{Proceedings of the 2014 conference on empirical methods in natural language processing} (pp. 1532-1543).
#' 
#' @importFrom stats na.omit
#' 
#' @export
#'
# NLP Scores
# Updated 05.03.2022
nlp_scores <- function(
  text, classes,
  semantic_space = c(
    "baroni", "cbow", "cbow_ukwac",
    "en100", "glove", "tasa"
  ),
  preprocess = TRUE,
  remove_stop = TRUE,
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
  
  # Check for semantic space
  if(missing(semantic_space)){
    semantic_space <- "glove"
  }else{
    semantic_space <- tolower(match.arg(semantic_space))
  }
  
  # Check for multiple transformers
  if(length(semantic_space) > 1){
    stop("Only one semantic space can be used at a time.\n")
  }
  
  # Check for semantic space in environment
  if(!exists(semantic_space, envir = as.environment(envir))){
    
    # Identify OSF link
    osf_link <- switch(
      semantic_space,
      "baroni" = "ztxjc",
      "cbow" = "3nfha",
      "cbow_ukwac" = "zxkjb",
      "en100" = "f2jv5",
      "glove" = "e3js4",
      "tasa" = "3kvmq"
    )
    
    # Check if semantic space exists
    if(
      !paste(semantic_space, "rdata", sep = ".") %in% # Saved semantic space
      tolower(list.files(tempdir())) # Temporary directory
    ){
      
      # Download semantic space (if not found)
      space_file <- suppressMessages(
        osfr::osf_download(
          osfr::osf_retrieve_file(
            osf_link
          ),
          path = tempdir(),
          progress = TRUE
        )
      )
      
    }else{
      
      # Create dummy space file list
      space_file <- list()
      space_file$local_path <- paste(
        tempdir(), "\\",
        semantic_space, ".RData",
        sep = ""
      )
      
    }
    
    # Let user know semantic space is loading
    message("Loading semantic space...", appendLF = FALSE)
    
    # Load semantic space
    load(space_file$local_path)
    
    # Let user know loading is finished
    message("done")
    
  }
  
  # Set semantic space to generic name
  space <- get(semantic_space)
  
  # Keep or remove semantic space from environment
  if(isTRUE(keep_in_env)){
    
    # Keep
    assign(
      x = semantic_space,
      value = space,
      envir = as.environment(envir)
    )
    
  }else{
    
    # Remove
    rm(list = ls()[which(ls() == semantic_space)])

  }
  
  # Basic preprocessing
  if(isTRUE(preprocess)){
    text <- preprocess_text( # Internal function. See `utils-transforEmotion`
      text, remove_stop = remove_stop
    )
  }
  
  # Split sentences into individual Words
  split_list <- lapply(text, strsplit, split = " ")
  
  # Shrink semantic space to only unique words
  ## Obtain unique words
  unique_words <- unique(unlist(split_list))
  ## Obtain words that exist in space
  space_index <- na.omit(match(
    unique_words, row.names(space)
  ))
  ## Obtain classes that exist in space
  class_index <- match(
    classes, row.names(space)
  )
  ## Report if any classes are not in semantic space
  if(any(is.na(class_index))){
    
    ### Bad classes
    bad_classes <- classes[is.na(class_index)]
    
    ### Message about bad classes
    bad_classes_message(bad_classes)
    
    ### Remove bad classes
    classes <- classes[!is.na(class_index)]
    class_index <- class_index[!is.na(class_index)]
    
  }
  ## Shrink space
  shrink_space <- space[c(
    space_index, # Words in text
    class_index # Words in classes
  ),]
  
  # Remove space
  rm(space)
  
  # Message
  message("Obtaining scores...")
  
  # Apply through text
  scores <- pbapply::pblapply(text, function(x){
    
    # Obtain semantic similarity
    scores <- suppressWarnings(
      LSAfun::multicostring(
        x = x, # Text
        y = classes, # Classes
        tvectors = shrink_space # Semantic space
      )
    )
    
    # Re-organize output
    scores <- as.vector(scores)
    names(scores) <- classes
    
    # Return
    return(scores)
    
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
