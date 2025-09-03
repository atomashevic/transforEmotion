.onLoad <- function(libname, pkgname)
{
    # Initialize vision model registry with built-in models
    .init_builtin_models()
}

.onAttach <- function(libname, pkgname)
{
    msg <- styletext(styletext(paste("\ntransforEmotion (version ", packageVersion("transforEmotion"), ")\n", sep = ""), defaults = "underline"), defaults = "bold")
    msg <- paste(msg, '\nImportant: If you are using RStudio, please make sure you have the latest version installed.')
    msg <- paste(msg, '\nFor help getting started, type browseVignettes("transforEmotion")\n')
    msg <- paste(msg, "\nFor bugs and errors, submit an issue to <https://github.com/atomashevic/transforEmotion/issues>")
    msg <- paste(msg, "\nBefore running an analysis for the first time after installing the package, please run `transforEmotion::setup_miniconda()` to install the necessary Python modules.")
    msg <- paste(msg, "\nData Privacy: All processing is done locally with the downloaded model, and your data is never sent to any remote server or third-party.")
    msg <- paste(msg, "\n\nAvailable vision models: Use list_vision_models() to see all models or register_vision_model() to add custom models.")
    packageStartupMessage(msg)
    Sys.unsetenv("RETICULATE_PYTHON")
    Sys.setenv(RETICULATE_PYTHON_ENV = "transforEmotion")
    requireNamespace("reticulate")
}

# Internal: provide a mockable binding for testthat to override
# Allows tests to use with_mocked_bindings(`jsonlite::fromJSON` = ...)
`jsonlite::fromJSON` <- jsonlite::fromJSON
