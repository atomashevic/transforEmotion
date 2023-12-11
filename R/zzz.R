.onload <- function(libname, pkgname)
{library.dynam("transforEmotion",package=pkgname,lib.loc=libname)
}

.onAttach <- function(libname, pkgname)
{
    msg <- styletext(styletext(paste("\ntransforEmotion (version ", packageVersion("transforEmotion"), ")\n", sep = ""), defaults = "underline"), defaults = "bold")
    msg <- paste(msg,'\nFor help getting started, type browseVignettes("transforEmotion")\n')
    msg <- paste(msg,"\nFor bugs and errors, submit an issue to <https://github.com/atomashevic/transforEmotion/issues>")
    msg <- paste(msg,"\nBefore running any analysis, please run `transforEmotion::setup_miniconda()` to install the necessary Python modules.")
    packageStartupMessage(msg)
    Sys.unsetenv("RETICULATE_PYTHON")
    Sys.setenv(RETICULATE_PYTHON_ENV =  "transforEmotion")
    library(reticulate)
}
