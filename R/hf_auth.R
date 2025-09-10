#' @noRd
# Internal helpers to ensure Hugging Face auth for gated models (Gemma 3)

# Check env for an HF token
.hf_get_token <- function() {
  tok <- Sys.getenv("HF_TOKEN", unset = NA_character_)
  if (!is.na(tok) && nzchar(tok)) return(tok)
  tok <- Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = NA_character_)
  if (!is.na(tok) && nzchar(tok)) return(tok)
  return(NA_character_)
}

# Prompt user to paste HF token and optionally save it
.hf_prompt_token <- function() {
  message(
    "A Hugging Face access token is required to use Gemma 3 models (gated repos).\n",
    "Open https://huggingface.co/settings/tokens and create a token with WRITE scope.\n"
  )
  tok <- trimws(readline(prompt = "Paste your Hugging Face token (starts with 'hf_'): "))
  if (!nzchar(tok)) {
    stop("No token provided. Aborting.", call. = FALSE)
  }

  # Set in-session env (both common var names)
  Sys.setenv(HF_TOKEN = tok)
  Sys.setenv(HUGGINGFACE_HUB_TOKEN = tok)

  # Save automatically (no prompt), warn about plaintext and how to remove
  renv_path <- path.expand("~/.Renviron")
  # Backup existing file
  if (file.exists(renv_path)) {
    bak <- paste0(renv_path, ".bak.", format(Sys.time(), "%Y%m%d%H%M%S"))
    try(suppressWarnings(file.copy(renv_path, bak, overwrite = TRUE)), silent = TRUE)
  }
  lines <- character()
  if (file.exists(renv_path)) {
    lines <- readLines(renv_path, warn = FALSE)
    # Drop existing HF_TOKEN/HUGGINGFACE_HUB_TOKEN if present
    keep <- !grepl("^(HF_TOKEN|HUGGINGFACE_HUB_TOKEN)=", lines)
    lines <- lines[keep]
  }
  # Write both env vars (do not print the token value)
  lines <- c(lines,
             paste0("HF_TOKEN=", tok),
             paste0("HUGGINGFACE_HUB_TOKEN=", tok))
  writeLines(lines, renv_path)
  message("Saved Hugging Face token to ", renv_path, " (plaintext).")
  message("You can delete the HF_TOKEN or HUGGINGFACE_HUB_TOKEN lines at any time.")
  message("Restart R for ~/.Renviron changes to apply in new sessions.")

  invisible(tok)
}

# Ensure HF auth is present (for Gemma). If interactive and missing, prompt.
ensure_hf_auth_for_gemma <- function(interactive_ok = TRUE, repo_id = NULL) {
  # Ensure reticulate uses the transforEmotion conda environment
  ensure_te_py_env()

  tok <- .hf_get_token()
  if (!is.na(tok) && nzchar(tok)) {
    # Token already present in environment; rely on env vars and avoid calling login()
    # to prevent the note about HF_TOKEN precedence.
    Sys.setenv(HUGGINGFACE_HUB_TOKEN = tok)
    Sys.setenv(HF_TOKEN = tok)
    # Optional: validate access to the gated repo if provided
    if (!is.null(repo_id)) {
      ok <- try({
        reticulate::py_run_string(
          paste0(
            "from huggingface_hub import HfApi\n",
            "import os\n",
            "api = HfApi()\n",
            "repo = '", repo_id, "'\n",
            "tok = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')\n",
            "api.model_info(repo_id=repo, token=tok)\n"
          )
        )
        TRUE
      }, silent = TRUE)
      if (inherits(ok, "try-error")) {
        message(
          "Could not validate access to ", repo_id, ". If download fails, ensure you've accepted access at https://huggingface.co/", repo_id
        )
      }
    }
    return(invisible(TRUE))
  }

  # No token in env
  if (interactive_ok && interactive()) {
    .hf_prompt_token()
    # Token is now set in env by prompt; skip login() and rely on env vars to avoid noisy notes.
    # Validate repo if provided
    if (!is.null(repo_id)) {
      ok <- try({
        reticulate::py_run_string(
          paste0(
            "from huggingface_hub import HfApi\n",
            "import os\n",
            "api = HfApi()\n",
            "repo = '", repo_id, "'\n",
            "tok = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')\n",
            "api.model_info(repo_id=repo, token=tok)\n"
          )
        )
        TRUE
      }, silent = TRUE)
      if (inherits(ok, "try-error")) {
        stop(
          paste0(
            "Your token was received but access to the gated repo '", repo_id, "' could not be validated.\n",
            "Please open https://huggingface.co/", repo_id, " and accept access with your account, then retry."
          ), call. = FALSE
        )
      }
    }
    return(invisible(TRUE))
  }

  stop(
    paste0(
      "Hugging Face token not found. Gemma 3 models are gated.\n",
      "Set env var HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) with a WRITE-scoped token,\n",
      "and ensure you have accepted the model license on the model page; or run interactively to be prompted."
    ), call. = FALSE
  )
}
