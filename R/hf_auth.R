#' @noRd
# Internal helpers to ensure Hugging Face auth for gated models (Gemma 3)

# Prompt user to paste HF token (no storage); returns token string
.hf_prompt_token <- function() {
  message(
    "A Hugging Face access token is required to use Gemma 3 models (gated repos).\n",
    "Open https://huggingface.co/settings/tokens and create a token with WRITE scope.\n"
  )
  tok <- trimws(readline(prompt = "Paste your Hugging Face token (starts with 'hf_'): "))
  if (!nzchar(tok)) stop("No token provided. Aborting.", call. = FALSE)
  return(tok)
}

# Temporarily set HF token for the duration of expr; restore afterwards
with_hf_token <- function(token, expr) {
  stopifnot(is.character(token), length(token) == 1L, nzchar(token))
  old_hf <- Sys.getenv("HF_TOKEN", unset = NA_character_)
  old_hub <- Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = NA_character_)
  on.exit({
    if (is.na(old_hf) || !nzchar(old_hf)) Sys.unsetenv("HF_TOKEN") else Sys.setenv(HF_TOKEN = old_hf)
    if (is.na(old_hub) || !nzchar(old_hub)) Sys.unsetenv("HUGGINGFACE_HUB_TOKEN") else Sys.setenv(HUGGINGFACE_HUB_TOKEN = old_hub)
  }, add = TRUE)
  Sys.setenv(HF_TOKEN = token)
  Sys.setenv(HUGGINGFACE_HUB_TOKEN = token)
  eval.parent(substitute(expr))
}

# Temporarily remove any HF tokens for the duration of expr; restore afterwards
without_hf_token <- function(expr) {
  old_hf <- Sys.getenv("HF_TOKEN", unset = NA_character_)
  old_hub <- Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = NA_character_)
  on.exit({
    if (is.na(old_hf) || !nzchar(old_hf)) Sys.unsetenv("HF_TOKEN") else Sys.setenv(HF_TOKEN = old_hf)
    if (is.na(old_hub) || !nzchar(old_hub)) Sys.unsetenv("HUGGINGFACE_HUB_TOKEN") else Sys.setenv(HUGGINGFACE_HUB_TOKEN = old_hub)
  }, add = TRUE)
  Sys.unsetenv("HF_TOKEN")
  Sys.unsetenv("HUGGINGFACE_HUB_TOKEN")
  eval.parent(substitute(expr))
}

# Heuristic to detect HF auth/permission errors
.is_hf_auth_error <- function(err) {
  msg <- paste(capture.output(print(err)), collapse = "\n")
  grepl("401|Unauthorized|Forbidden|gated|not authorized|requires authorization|You are not logged in",
        msg, ignore.case = TRUE)
}
