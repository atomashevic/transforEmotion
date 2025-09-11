.onLoad <- function(libname, pkgname)
{
    # Initialize vision model registry with built-in models
    .init_builtin_models()
}

.onAttach <- function(libname, pkgname)
{
    # Prevent reticulate from auto-creating a default venv; we manage envs via uv
    if (identical(Sys.getenv("RETICULATE_AUTOCONFIGURE", unset = ""), "")) {
        Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
    }

    # Suggest installing uv early (interactive prompt), before reticulate initializes Python
    if (interactive()) {
        try(ensure_uv_available(prompt = TRUE), silent = TRUE)
    } else {
        # Non-interactive: gentle nudge only
        try(ensure_uv_available(prompt = FALSE), silent = TRUE)
    }

    msg <- styletext(styletext(paste("\ntransforEmotion (version ", packageVersion("transforEmotion"), ")\n", sep = ""), defaults = "underline"), defaults = "bold")
    msg <- paste(msg, '\nImportant: If you are using RStudio, please make sure you have the latest version installed.')
    msg <- paste(msg, '\nFor help getting started, type browseVignettes("transforEmotion")\n')
    msg <- paste(msg, "\nFor bugs and errors, submit an issue to <https://github.com/atomashevic/transforEmotion/issues>")
    msg <- paste(msg, "\nPython dependencies are installed automatically via uv on first use. Optionally run setup_modules() to pre-warm the environment.")
    msg <- paste(msg, "\nData Privacy: All processing is done locally with the downloaded model, and your data is never sent to any remote server or third-party.")
    msg <- paste(msg, "\n\nAvailable vision models: Use list_vision_models() to see all models or register_vision_model() to add custom models.")
    packageStartupMessage(msg)
    Sys.unsetenv("RETICULATE_PYTHON")
    requireNamespace("reticulate")

    # If the default reticulate venv exists, suggest the cleanup helper (no prompts on attach)
    default_venv <- path.expand(file.path("~", ".virtualenvs", "r-reticulate"))
    if (dir.exists(default_venv)) {
        packageStartupMessage(
          "Detected ~/.virtualenvs/r-reticulate; run transforEmotion::te_cleanup_default_venv() to remove it and prefer uv."
        )
    }
}

# Internal: provide a mockable binding for testthat to override
# Allows tests to use with_mocked_bindings(`jsonlite::fromJSON` = ...)
`jsonlite::fromJSON` <- jsonlite::fromJSON
