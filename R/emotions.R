#' Emotions Data
#'
#' A matrix containing words (n = 175,592) and the emotion category most frequently associated with each word.
#' This dataset is a modified version of the 'DepecheMood++' lexicon developed by
#' Araque, Gatti, Staiano, and Guerini (2018). For proper scoring, text should not be
#' stemmed prior to using this lexicon. This version of the lexicon does not
#' rely on part of speech tagging.
#'
#'
#'
#' @name emotions
#'
#' @docType data
#'
#' @usage data(emotions)
#'
#' @format A data frame with 175,592 rows and 9 columns.
#' \describe{
#'    \item{word}{An entry in the lexicon, in English}
#'    \item{AFRAID, AMUSED, ANGRY, ANNOYED, DONT_CARE, HAPPY, INSPIRED, SAD}{The emotional category. All emotions contain either a 0 or 1. If the
#'    category is most likely to be associated with the word, it recieves a 1, otherwise, 0.
#'    Words are only associated with one category.}
#' }
#'
#' @keywords datasets
#'
#' @examples
#' data("emotions")
#'
#' @references
#' Araque, O., Gatti, L., Staiano, J., and Guerini, M. (2018).
#' DepecheMood++: A bilingual emotion lexicon built through simple yet powerful techniques.
#' \emph{ArXiv}
#'

NULL
#----
