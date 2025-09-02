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
    "Open https://huggingface.co/settings/tokens and create a token with read scope.\n"
  )
  tok <- trimws(readline(prompt = "Paste your Hugging Face token (starts with 'hf_'): "))
  if (!nzchar(tok)) {
    stop("No token provided. Aborting.", call. = FALSE)
  }

  # Set in-session env (both common var names)
  Sys.setenv(HF_TOKEN = tok)
  Sys.setenv(HUGGINGFACE_HUB_TOKEN = tok)

  # Offer to save
  message(
    "\nSave this token for future sessions? [y/N]\n",
    "Warning: it will be stored in plaintext in ~/.Renviron."
  )
  ans <- tolower(trimws(readline(prompt = "> ")))
  if (ans %in% c("y", "yes")) {
    renv_path <- path.expand("~/.Renviron")
    lines <- character()
    if (file.exists(renv_path)) {
      lines <- readLines(renv_path, warn = FALSE)
      # Drop existing HF_TOKEN/HUGGINGFACE_HUB_TOKEN if present
      keep <- !grepl("^(HF_TOKEN|HUGGINGFACE_HUB_TOKEN)=", lines)
      lines <- lines[keep]
    }
    lines <- c(lines, paste0("HF_TOKEN=", tok))
    writeLines(lines, renv_path)
    message("Saved token to ", renv_path)
    message("Note: restart R for ~/.Renviron changes to take effect in new sessions.")
  } else {
    message("Token kept for current session only.")
  }

  invisible(tok)
}

# Ensure HF auth is present (for Gemma). If interactive and missing, prompt.
ensure_hf_auth_for_gemma <- function(interactive_ok = TRUE, repo_id = NULL) {
  # Ensure reticulate uses the transforEmotion conda environment
  ensure_te_py_env()

  tok <- .hf_get_token()
  if (!is.na(tok) && nzchar(tok)) {
    # Ensure huggingface_hub sees it (login caches credential). Best-effort.
    try({
      reticulate::py_run_string(
        "import os\nfrom huggingface_hub import login\n"
      )
      reticulate::py_run_string(
        "tok = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')\nif tok:\n    login(token=tok, add_to_git_credential=False)\n"
      )
    }, silent = TRUE)
    # Also mirror into both env vars in R process for Transformers
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
    # Attempt login post-prompt
    try({
      reticulate::py_run_string(
        "import os\nfrom huggingface_hub import login\n"
      )
      reticulate::py_run_string(
        "tok = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_HUB_TOKEN')\nif tok:\n    login(token=tok, add_to_git_credential=False)\n"
      )
    }, silent = TRUE)
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
      "Set env var HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) with a read-scoped token,\n",
      "or run interactively to be prompted."
    ), call. = FALSE
  )
}
