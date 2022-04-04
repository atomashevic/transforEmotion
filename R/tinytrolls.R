#' Russian Trolls Data - Small Version
#'
#' A matrix containing a smaller subset of tweets from the \code{trolls} dataset, useful for test purposes.
#' There are approximately 20,000 tweets from 50 authors.
#' This dataset includes only authored tweets by each account; retweets, reposts, and repeated tweets have been removed.
#' The original data was provided by FiveThirtyEight and Clemson University researchers Darren Linvill and Patrick Warren.
#' For more information, visit https://github.com/fivethirtyeight/russian-troll-tweets
#'
#'
#'
#' @name tinytrolls
#'
#' @docType data
#'
#' @usage data(tinytrolls)
#'
#' @format A data frame with 22,143 rows and 6 columns.
#' \describe{
#'    \item{content}{A tweet.}
#'    \item{author}{The name of the handle that authored the tweet.}
#'    \item{publish_date}{The date the tweet was published on.}
#'    \item{followers}{How many followers the handle had at the time of posting.}
#'    \item{updates}{How many interactions (including likes, tweets, retweets) the post garnered.}
#'    \item{account_type}{Left or Right}
#
#' }
#'
#' @keywords datasets
#'
#' @examples
#' data(tinytrolls)
#'
#'

NULL
#----
