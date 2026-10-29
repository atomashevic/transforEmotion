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

#%%%%%%%%%%%%%%%%%%%%%%
# SYSTEM FUNCTIONS ----
#%%%%%%%%%%%%%%%%%%%%%%

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

  # Styled text uses ANSI escape codes: RStudio renders them, and so does a
  # Linux terminal, but notebooks (Jupyter, Google Colab) print them verbatim
  TEXT <- RSTUDIO || (OS == "linux" && isatty(stderr()) && Sys.getenv("TERM") != "dumb")

  res <- list()

  res$OS <- OS
  res$RSTUDIO <- RSTUDIO
  res$TEXT <- TEXT

  return(res)
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

#' @noRd
# A YouTube video URL: http(s), host youtube.com (any subdomain) or youtu.be.
# A local path that contains "youtube" is not one.
.te_is_youtube_url <- function(x) {
  grepl("^https?://([a-z0-9-]+\\.)*(youtube\\.com|youtu\\.be)(/|$)", x, ignore.case = TRUE)
}

#' @noRd
# transformers raises "... requires the protobuf library but it was not found
# in your environment" when it converts a SentencePiece tokenizer in a Python
# environment without protobuf. Its message does not say which Python is
# meant. Called with the error of a failed model load; returns FALSE for any
# other error. The Python is running when this error occurs, so py_config()
# only reads the configuration and starts nothing.
.te_stop_if_no_protobuf <- function(err) {
  if (!isTRUE(grepl("requires the protobuf library", paste(err, collapse = "\n"),
                    fixed = TRUE))) {
    return(invisible(FALSE))
  }
  python <- Sys.getenv("TRANSFOREMOTION_PYTHON", unset = "")
  fixed <- nzchar(python)
  if (fixed) python <- .te_shell_python(python)
  reason <- NULL
  if (!fixed && isTRUE(reticulate::py_available(initialize = FALSE))) {
    config <- tryCatch(reticulate::py_config(), error = function(e) NULL)
    if (is.character(config$python)) python <- config$python
    reason <- config$forced
  }
  stop(
    "The tokenizer of this model needs the Python package protobuf, which ",
    if (nzchar(python)) {
      paste0("this Python does not have: ", python, "\nInstall it there, for ",
             "example with: uv pip install --python \"", python,
             "\" \"protobuf>=3.20,<8\"\n")
    } else {
      "the Python environment that reticulate uses does not have.\n"
    },
    if (fixed) {
      paste0("This is the environment that TRANSFOREMOTION_PYTHON points to. ",
             "You can also rebuild it from python_requirements(); for a ",
             "read-only image, see ?python_requirements.")
    } else {
      paste0("reticulate selected this Python",
             if (is.character(reason) && length(reason) == 1L) paste0(" (", reason, ")"),
             ", so transforEmotion did not install its packages. To let ",
             "transforEmotion manage Python, remove that selection ",
             "(RETICULATE_PYTHON, VIRTUAL_ENV, use_python(), use_virtualenv(), ",
             "a venv, .venv, virtualenv or .virtualenv folder in the working ",
             "directory, or ~/.virtualenvs/r-reticulate) and restart R.")
    },
    call. = FALSE
  )
}
