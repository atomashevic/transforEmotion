#' Punctuation Removal for Text
#'
#' @description Keeps the punctuations you want and removes the punctuations you don't
#' 
#' @param text Character vector or list.
#' Text in a vector or list data format
#' 
#' @param allowPunctuations Character vector.
#' Punctuations that should be allowed in the text.
#' Defaults to common punctuations in English text
#'
#' @return Returns text with only the allowed punctuations
#' 
#' @details Coarsely removes punctuations from text. Keeps general punctuations
#' that are used in most English language text. Apostrophes are much trickier.
#' For example, not allowing "'" will remove apostrophes from contractions
#' like "can't" becoming "cant"
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @examples
#' # Load data
#' data(neo_ipip_extraversion)
#' 
#' # Example text 
#' text <- neo_ipip_extraversion$friendliness
#' 
#' # Keep only periods
#' punctuate(text, allowPunctuations = c("."))
#' 
#' @export
#'
# Punctuations
# Updated 01.03.2022
punctuate <- function(
  text, allowPunctuations = c(
    "-", "?", "'", '"', ";", ",", ".", "!"
  )
)
{
  # Obtain text object type
  object_type <- class(text)
  
  # Punctuations to *allow*
  if(missing(allowPunctuations)){
    allowPunctuations <- c("-", "?", "'", '"', ";", ",", ".", "!")
  }else{
    allowPunctuations <- match.arg(allowPunctuations, several.ok = TRUE)
  }
  
  # Set up characters allowed in text replacement
  characters <- paste("([", paste(allowPunctuations, collapse = ""), "])|[[:punct:]]", sep = "", collapse = "")
  
  # Apply over data
  text <- lapply(text, function(x){
    gsub(characters, "\\1", x)
  })
  
  # Convert back to character vector
  if(any(class(object_type) == "character")){
    text <- unlist(text)
  }
  
  # Return text
  return(text)
  
}
