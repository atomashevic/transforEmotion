#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Preprocessing functions ----
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#' Warning to user about non-text input
#'
#' @noRd
#'
# Non-text Warning
# Updated 05.03.2022
non_text_warning <- function(text)
{
  # Clear na.action attribute
  attr(text, "na.action") <- NULL

  # Check that input is vector or list
  if(!is.vector(text)){

    # Send error if not a vector or list
    stop(
      paste(
        "The 'text' argument expects a 'vector' or 'list'. Instead, class(text) = ",
        paste("'", class(text), "'", sep = "", collapse = ", ")
      )
    )

  }else{

    # Check for text
    text_index <- unlist(lapply(text, is.character))

    # Check for all text
    if(!all(text_index)){

      # Obtain bad indices
      bad_index <- which(!text_index)

      # Remove non-text
      text <- text[-bad_index]

      # Check if all elements were removed
      if(length(text) == 0){
        stop(
          "All elements of 'text' object were not identified as 'character' type.\n\nPlease check for whether your text is 'character' type: `is.character(text)`"
        )
      }

      # Adjust warning for number of non-character types
      if(length(bad_index) == 1){

        # Set up head of warning
        head <- "Index"

        # Set up body of warning
        body <- paste(bad_index, sep = "")

        # Set up end of warning
        end <- "of 'text' object was not identified 'character' type.\n\nThis index was removed from analysis."

      }else if(length(bad_index) == 2){

        # Set up head of warning
        head <- "Indices"

        # Set up body of warning
        body <- paste(bad_index[1], "and", bad_index[2])

        # Set up end of warning
        end <- "of 'text' object were not identified 'character' type.\n\nThese indices were removed from analysis."

      }else if(length(bad_index) > 2){

        # Set up head of warning
        head <- "Indices"

        # Set up body of warning
        ## All bad indices except for last
        body <- paste(bad_index[-length(bad_index)], sep = "", collapse = ", ")

        ## Add last bad index
        body <- paste(
          body,
          ", and ", bad_index[length(bad_index)],
          sep = ""
        )

        # Set up end of warning
        end <- "of 'text' object were not identified 'character' type.\n\nThese indices were removed from analysis."

      }

      # Provide warning
      warning(
        paste(
          head, body, end
        )
      )

    }

  }

}

#' Alphanumeric characters
#'
#' @noRd
#'
# Alphanumeric characters
# Updated 05.03.2022
keep_alphanumeric <- function(text)
{
  # Remove non-alphanumeric characters
  alphanum_text <- lapply(text, function(x){
    gsub("[^[:alnum:][:space:][:punct:]]", "", x)
  })

  return(alphanum_text)

}

#' Reduces characters repeated more than twice to single character
#'
#' @noRd
#'
# Remove repeated characters
# Updated 05.03.2022
remove_repeated <- function(text)
{
  # Repeated characters (more than 2) reduced to 1
  repeated_text <- lapply(text, function(x){
    gsub('([[:alnum:]])\\1{2,}', '\\1', x)
  })

  # Repeated punctuations reduced to 1
  repeated_text <- lapply(text, function(x){
    gsub('([[:punct:]])\\1+', '\\1', x)
  })

  return(repeated_text)

}

#' Remove escaped characters
#'
#' @noRd
#'
# Removes escaped characters
# Updated 05.03.2022
remove_escapes <- function(text)
{
  # Escapes removed
  escape_text <- lapply(text, function(x){

    # Remove new line
    x <- gsub("\\n", " ", x)
    # Remove tab
    x <- gsub("\\t", " ", x)
    # Remove quote
    x <- gsub('\\"', " ", x)
    # Remove single backslash
    x <- gsub("\\\\", " ", x)

    return(x)

  })

  return(escape_text)

}

#' Ensures proper spacing for punctuations
#'
#' @noRd
#'
# Proper punctuation spacing
# Updated 05.03.2022
punctuation_spaces <- function(text)
{
  # Converge spaces next to punctuation
  converge_text <- lapply(text, function(x){

    # Find punctuation
    punct_index <- grep("[[:punct:]]", x)

    # Punctuation with only one character
    punct_index <- punct_index[nchar(x[punct_index]) == 1]

    # Ensure their are punctuations
    if(length(punct_index) != 0){

      # Loop through punctuations
      replace_punct <- unlist(
        lapply(punct_index, function(i){
          paste(x[i-1], x[i], sep = "")
        })
      )

      # Replace punctuations in text
      x[punct_index] <- replace_punct

      # Remove previous word in text
      x <- x[-(punct_index-1)]

    }

    return(x)

  })

  return(converge_text)

}

#' Remove all white space between words
#' (correctly spaces punctuations as well)
#'
#' @noRd
#'
# Removes white space
# Updated 05.03.2022
remove_whitespace <- function(text)
{
  # Split and trimmed text
  split_text <- lapply(text, function(x){
    trimws(unlist(strsplit(x, split = " ")))
  })

  # Remove blank text
  blank_text <- lapply(split_text, function(x){
    na.omit(ifelse(x == "", NA, x))
  })

  # Proper spaces for punctuations
  punct_text <- punctuation_spaces(blank_text)

  # Re-paste text together
  repaste_text <- lapply(punct_text, function(x){
    paste(x, collapse = " ")
  })

  return(repaste_text)

}

#' Remove stop words from text
#'
#' @importFrom utils data
#'
#' @noRd
#'
# Removes stop words
# Updated 05.03.2022
remove_stop_words <- function(text)
{
  # Load stop words
  stop_words <- get(data("stop_words", envir = environment()))

  # Format stop words for `gsub` function
  stop_format <- paste("\\b", stop_words, "\\b", sep = "", collapse = "|")

  # Remove stop words
  stop_text <- lapply(text, function(x){
    gsub(stop_format, "", x)
  })

  return(stop_text)

}

#' Basic preprocessing
#'
#' @noRd
#'
# Preprocessing
# Updated 05.03.2022
preprocess_text <- function(text, remove_stop = FALSE)
{

  # Convert to lowercase
  text <- lapply(text, tolower)

  # Keep alphanumeric and punctuation characters
  text <- keep_alphanumeric(text)

  # Remove escaped text
  text <- remove_escapes(text)

  # Remove repeated text
  text <- remove_repeated(text)

  # Remove stop words
  if(isTRUE(remove_stop)){
    text <- remove_stop_words(text)
  }

  # Remove white space (with proper punctuation spaces)
  text <- remove_whitespace(text)

  return(text)

}

#%%%%%%%%%%%%%%%%
# nlp_scores ----
#%%%%%%%%%%%%%%%%

#' Message to user about bad classes
#'
#' @noRd
#'
# Bad Class Message
# Updated 05.03.2022
bad_classes_message <- function(bad_classes)
{
  # Adjust message for number of bad classes
  if(length(bad_classes) == 1){

    # Set up head of message
    head <- "Class"

    # Set up body of message
    body <- paste("'", bad_classes, "'", sep = "")

    # Set up end of message
    end <- "was not found in the semantic space.\n\nIt was removed from the NLP analysis."

  }else if(length(bad_classes) == 2){

    # Set up head of message
    head <- "Classes"

    # Set up body of message
    body <- paste(
      "'", bad_classes[1], "' and '", bad_classes[2], "'", sep = ""
    )

    # Set up end of message
    end <- "were not found in the semantic space.\n\nThey were removed from the NLP analysis."

  }else if(length(bad_classes) > 2){

    # Set up head of message
    head <- "Classes"

    # Set up body of message
    ## All bad classes except for last
    body <- paste(
      "'", bad_classes[-length(bad_classes)], "'", sep = "", collapse = ", "
    )

    ## Add last bad class
    body <- paste(
      body,
      ", and '", bad_classes[length(bad_classes)], "'",
      sep = ""
    )

    # Set up end of message
    end <- "were not found in the semantic space.\n\nThey were removed from the NLP analysis."

  }

  # Generate full message
  message(
    paste(
      head, body, end
    )
  )

}

#%%%%%%%%%%%%%%%%%%%%%%
# SYSTEM FUNCTIONS ----
#%%%%%%%%%%%%%%%%%%%%%%

# Silent call ----
# Updated 02.02.2024
silent_call <- function(input)
{
  # Capture any output
  sink <- capture.output(
    output <- suppressWarnings(
      suppressMessages(input)
    )
  )

  # Return output
  return(output)

}

#' Error report
#'
#' @description Gives necessary information for user reporting error
#'
#' @param result Character.
#' The error from the result
#'
#' @param SUB_FUN Character.
#' Sub-routine the error occurred in
#'
#' @param FUN Character.
#' Main function the error occurred in
#'
#' @return Error and message to send to GitHub
#'
#' @author Alexander Christensen <alexpaulchristensen@gmail.com>
#'
#' @noRd
#'
#' @importFrom utils packageVersion
#'
# Error Report
# Updated 26.02.2021
error.report <- function(result, SUB_FUN, FUN)
{
  # Let user know that an error has occurred
  message(paste("\nAn error has occurred in the '", SUB_FUN, "' function of '", FUN, "':\n", sep =""))

  # Give them the error to send to you
  cat(paste(result))

  # Tell them where to send it
  message("\nPlease open a new issue on GitHub (bug report): https://github.com/hfgolino/EGAnet/issues/new/choose")

  # Give them information to fill out the issue
  OS <- as.character(Sys.info()["sysname"])
  OSversion <- paste(as.character(Sys.info()[c("release", "version")]), collapse = " ")
  Rversion <- paste(R.version$major, R.version$minor, sep = ".")
  EGAversion <- paste(unlist(packageVersion("EGAnet")), collapse = ".")

  # Let them know to provide this information
  message(paste("\nBe sure to provide the following information:\n"))

  # To reproduce
  message(styletext("To Reproduce:", defaults = "bold"))
  message(paste(" ", textsymbol("bullet"), " Function error occurred in: ", SUB_FUN, " function of ", FUN, sep = ""))

  # R, SemNetCleaner, and SemNetDictionaries
  message(styletext("\nR and EGAnet versions:", defaults = "bold"))
  message(paste(" ", textsymbol("bullet"), " R version: ", Rversion, sep = ""))
  message(paste(" ", textsymbol("bullet"), " EGAnet version: ", EGAversion, sep = ""))

  # Desktop
  message(styletext("\nOperating System:", defaults = "bold"))
  message(paste(" ", textsymbol("bullet"), " OS: ", OS, sep = ""))
  message(paste(" ", textsymbol("bullet"), " Version: ", OSversion, sep = ""))
}

#' System check for OS and RSTUDIO
#'
#' @description Checks for whether text options are available
#'
#' @param ... Additional arguments
#'
#' @return \code{TRUE} if text options are available and \code{FALSE} if not
#'
#' @author Alexander Christensen <alexpaulchristensen@gmail.com>
#'
#' @noRd
# System Check
# Updated 08.09.2020
system.check <- function (...)
{
  OS <- unname(tolower(Sys.info()["sysname"]))

  RSTUDIO <- ifelse(Sys.getenv("RSTUDIO") == "1", TRUE, FALSE)

  TEXT <- TRUE

  if(!RSTUDIO){if(OS != "linux"){TEXT <- FALSE}}

  res <- list()

  res$OS <- OS
  res$RSTUDIO <- RSTUDIO
  res$TEXT <- TEXT

  return(res)
}

#' Colorfies Text
#'
#' Makes text a wide range of colors (8-bit color codes)
#'
#' @param text Character.
#' Text to color
#'
#' @return Colorfied text
#'
#' @author Alexander Christensen <alexpaulchristensen@gmail.com>
#'
#' @noRd
#'
# Color text
# Updated 08.09.2020
colortext <- function(text, number = NULL, defaults = NULL)
{
  # Check system
  sys.check <- system.check()

  if(sys.check$TEXT)
  {
    # Defaults for number (white text)
    if(is.null(number) || number < 0 || number > 231)
    {number <- 15}

    # Check for default color
    if(!is.null(defaults))
    {
      # Adjust highlight color based on background color
      if(defaults == "highlight")
      {
        if(sys.check$RSTUDIO)
        {

          if(rstudioapi::getThemeInfo()$dark)
          {number <- 226
          }else{number <- 208}

        }else{number <- 208}
      }else{

        number <- switch(defaults,
                         message = 204,
                         red = 9,
                         orange = 208,
                         yellow = 11,
                         "light green" = 10,
                         green = 34,
                         cyan = 14,
                         blue = 12,
                         magenta = 13,
                         pink = 211,
        )

      }

    }

    return(paste("\033[38;5;", number, "m", text, "\033[0m", sep = ""))

  }else{return(text)}
}

#' Stylizes Text
#'
#' Makes text bold, italics, underlined, and strikethrough
#'
#' @param text Character.
#' Text to stylized
#'
#' @return Sytlized text
#'
#' @author Alexander Christensen <alexpaulchristensen@gmail.com>
#'
#' @noRd
# Style text
# Updated 08.09.2020
styletext <- function(text, defaults = c("bold", "italics", "highlight",
                                         "underline", "strikethrough"))
{
  # Check system
  sys.check <- system.check()

  if(sys.check$TEXT)
  {
    if(missing(defaults))
    {number <- 0
    }else{

      # Get number code
      number <- switch(defaults,
                       bold = 1,
                       italics = 3,
                       underline = 4,
                       highlight = 7,
                       strikethrough = 9
      )

    }

    return(paste("\033[", number, ";m", text, "\033[0m", sep = ""))
  }else{return(text)}
}

#' Text Symbols
#'
#' Makes text symbols (star, checkmark, square root)
#'
#' @param symbol Character.
#'
#' @return Outputs symbol
#'
#' @author Alexander Christensen <alexpaulchristensen@gmail.com>
#'
#' @noRd
# Symbols
# Updated 24.04.2020
textsymbol <- function(symbol = c("alpha", "beta", "chi", "delta",
                                  "eta", "gamma", "lambda", "omega",
                                  "phi", "pi", "rho", "sigma", "tau",
                                  "theta", "square root", "infinity",
                                  "check mark", "x", "bullet")
)
{
  # Get number code
  sym <- switch(symbol,
                alpha = "\u03B1",
                beta = "\u03B2",
                chi = "\u03C7",
                delta = "\u03B4",
                eta = "\u03B7",
                gamma = "\u03B3",
                lambda = "\u03BB,",
                omega = "\u03C9",
                phi = "\u03C6",
                pi = "\u03C0",
                rho = "\u03C1",
                sigma = "\u03C3",
                tau = "\u03C4",
                theta = "\u03B8",
                "square root" = "\u221A",
                infinity = "\u221E",
                "check mark" = "\u2713",
                x = "\u2717",
                bullet = "\u2022"
  )

  return(sym)
}

