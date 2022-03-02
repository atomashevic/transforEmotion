.onload <- function(libname, pkgname)
{library.dynam("transforEmotion",package=pkgname,lib.loc=libname)}

.onAttach <- function(libname, pkgname)
{
    msg <- styletext(styletext(paste("\ntransforEmotion (version ", packageVersion("transforEmotion"), ")\n", sep = ""), defaults = "underline"), defaults = "bold")
    msg <- paste(msg,'\nFor help getting started, type browseVignettes("transforEmotion")\n')
    msg <- paste(msg,"\nFor bugs and errors, submit an issue to <https://github.com/AlexChristensen/transforEmotion/issues>")

    packageStartupMessage(msg)
}
