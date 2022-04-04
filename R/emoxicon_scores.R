#' Emoxicon Scores
#'
#' @description A bag-of-words approach for computing emotions in text data using
#' the lexicon compiled by Araque, Gatti, Staiano, and Guerini (2018).
#' 
#' @param text Matrix or data frame.
#' A data frame containing texts to be scored (one text per row)
#'
#' @param lexicon The lexicon used to score the words. The default is the \code{\link{emotions}} dataset,
#' a modification of the lexicon developed by Araque, Gatti, Staiano, and Guerini (2018).
#' To use the raw lexicon from Araque et. al (2018) containing the original probability weights, use the \code{\link{weights}} dataset.
#' If another custom lexicon is used, the first column of the lexicon should contain the terms
#' and the subsequent columns contain the scoring categories.
#'
#' @param exclude A vector listing terms that should be excluded from the lexicon.
#' Words specified in \code{exclude} will not
#' influence document scoring. Users should consider excluding 'red herring' words
#' that are more closely related to the topics of the documents,
#' rather than the documents' emotional content.
#' For example, the words "clinton" and "trump" are present in the lexicon and are both associated with the emotion 'AMUSED'.
#' Excluding these words when analyzing political opinions may produce more accurate results.
#'
#' @author Tara Valladares <tls8vx at virginia.edu> and Hudson F. Golino <hfg9s at virginia.edu>
#'
#' @examples
#' 
#' # Obtain "emotions" data
#' data("emotions")
#' 
#' # Obtain "tinytrolls" data
#' data("tinytrolls")
#'
#' \dontrun{
#' # Obtain emoxicon scores for first 10 tweets
#' emotions_tinytrolls <- emoxicon_scores(text = tinytrolls$content, lexicon = emotions)
#' }
#' 
#' @seealso \code{\link{emotions}}, where we describe how we modified the original DepecheMood++ lexicon.
#'
#' @references
#' Araque, O., Gatti, L., Staiano, J., and Guerini, M. (2018).
#' DepecheMood++: A bilingual emotion lexicon built through simple yet powerful techniques.
#' \emph{ArXiv}
#'
#' @importFrom dplyr left_join
#' @importFrom stats aggregate
#' @importFrom utils install.packages installed.packages
#'
#' @export
#'
#'
# Emoxicon Function
# Updated 04.02.2022
emoxicon_scores <- function(text, lexicon, exclude) {
  
  # Check for install packages
  installed_packages <- row.names(installed.packages())
  
  # Install 'remotes' package (if necessary)
  if(!"remotes" %in% installed_packages){
    install.packages("remotes")
  }
  
  # Install 'emoxicon' package from GitHub
  if(!"emoxicon" %in% installed_packages){
    remotes::install_github("tvall/emoxicon")
  } 
  
  # Check that input of 'text' argument is in the
  # appropriate format for the analysis
  non_text_warning(text) # see utils-transforEmotion.R for function
  
  # Check for missing lexicon
  if(missing(lexicon)){
    lexicon <- get(data("emotions", envir = environment()))
  }
  
  # Check for lexicon with "emoxicon" class
  if(any("emoxicon" %in% class(lexicon))){
    default.lexicon <- TRUE
  }else{default.lexicon <- FALSE}
  
  if(!missing(exclude)){
    lexicon <- lexicon[!(lexicon[,1] %in% tolower(exclude)),]
    message("Removing tokens in 'exclude' from the lexicon")
  }
  
  
  # Clean text and set up variables ------
  emotion_labels <- colnames(lexicon)[-1] # save emotion labels
  doc_id_full <- paste("doc", seq_along(text), sep = "")  # Create document IDs
  
  cleaning <- function(text) {
    # Clean and split text
    text <- tolower(text)  # lower case
    text <- gsub("[^[:alpha:][:space:]]*", "", text)  # remove punctuation
    text[text == ""] <- "nocontent"  # fill in empty entires with fillers to prevent dropped files
    text <- strsplit(text, split = " ")  # split text
    
    text
  }
  clean.text <- cleaning(text)  # preprocess text
  doc_id <- rep(doc_id_full, sapply(clean.text, length))  # create document ids
  words.df <- data.frame(text = unlist(clean.text))  #unlist the text
  
  
  
  # Score and format the text -----
  words.emotions <- suppressWarnings(suppressMessages(dplyr::left_join(words.df,
                                                                       lexicon, by = c(text = colnames(lexicon)[1]))))  # score the text
  
  
  
  # Quit if there is a problem with the lexicon (e.g., words with multiple
  # entries)
  if (length(doc_id) != nrow(words.emotions)) {
    stop("The number of tokens in your original text and the new scored text do not match. There are potentially duplicate entries in your lexicon.")
  }
  
  
  all.emotions <- cbind(doc_id, words.emotions)  # attach scored words back with the document ids
  
  results <- stats::aggregate(all.emotions[, emotion_labels], data.frame(doc_id),
                              sum, na.rm = T)  # sum emotions within each document
  results <- results[order(as.numeric(gsub("[a-z]+", "", as.character(results$doc_id)))), ]  # put the documents back in order
  results$text <- text  # put the text with the emotion scores
  results <- results[, c("doc_id", "text", emotion_labels)]  # reorder the columns
  
  class(results) <- append(class(results),"emoxicon")
  if(default.lexicon){
    class(results) <- append(class(results),"emotions")
  }
  
  results
  
}
