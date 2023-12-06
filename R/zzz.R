.onload <- function(libname, pkgname)
{library.dynam("transforEmotion",package=pkgname,lib.loc=libname)
}

.onAttach <- function(libname, pkgname)
{
    msg <- styletext(styletext(paste("\ntransforEmotion (version ", packageVersion("transforEmotion"), ")\n", sep = ""), defaults = "underline"), defaults = "bold")
    msg <- paste(msg,'\nFor help getting started, type browseVignettes("transforEmotion")\n')
    msg <- paste(msg,"\nFor bugs and errors, submit an issue to <https://github.com/atomashevic/transforEmotion/issues>")
    packageStartupMessage(msg)
    Sys.unsetenv("RETICULATE_PYTHON")
    tryCatch({
    reticulate::use_condaenv("transforEmotion", required = TRUE)
    }, error = function(e) {
    message("Python transforEmotion conda environment not activate. \nPlease run `setup_miniconda()` to install and activate the the environment with all required libraries. \nIf this issue persists submit an issue to <https://github.com/atomashevic/transforEmotion/issues> \n ", e$message)
})
}
