#' @noRd
check_nvidia_gpu <- function() {
  # This function checks if an NVIDIA GPU is available before we have access to `torch`
  if (.Platform$OS.type == "windows") {
    # Windows: Check using nvidia-smi if available
    has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))
    if (!has_nvidia_smi) return(FALSE)
    rc <- suppressWarnings(system("nvidia-smi", ignore.stdout = TRUE, ignore.stderr = TRUE))
    return(rc == 0)
  } else {
    # Linux/macOS: Prefer command existence checks to avoid noisy "command not found"
    has_gpu <- FALSE
    has_lspci <- nzchar(Sys.which("lspci"))
    has_nvidia_smi <- nzchar(Sys.which("nvidia-smi"))

    if (has_lspci) {
      rc <- suppressWarnings(system("lspci | grep -i nvidia", ignore.stdout = TRUE, ignore.stderr = TRUE))
      has_gpu <- has_gpu || (rc == 0)
    }
    if (has_nvidia_smi) {
      rc2 <- suppressWarnings(system("nvidia-smi", ignore.stdout = TRUE, ignore.stderr = TRUE))
      has_gpu <- has_gpu || (rc2 == 0)
    }
    return(has_gpu)
  }
}

# Default models, pre-downloaded by setup_modules()
.te_default_models <- c(
  "cross-encoder/nli-distilroberta-base",   # transformer_scores()
  "openai/clip-vit-base-patch32",           # image_scores(), video_scores()
  "sentence-transformers/all-MiniLM-L6-v2"  # sentence_similarity()
)

# Default models for rag(), pre-downloaded by setup_modules(extras = "rag")
.te_default_rag_models <- c(
  "BAAI/bge-small-en-v1.5",                 # document embeddings
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"      # default LLM (about 2.2 GB)
)

#' Setup Required Python Modules
#'
#' @description
#' Builds the Python environment for transforEmotion ahead of first use and
#' downloads the default models, so the first analysis call does not wait on
#' either. Calling it is optional: every function sets up what it needs on
#' first use.
#'
#' @param extras Character vector of optional feature sets to install
#' together with the core packages:
#' \describe{
#'   \item{\code{"rag"}}{LlamaIndex packages used by \code{rag()}}
#'   \item{\code{"youtube"}}{\code{pytubefix}, used by \code{video_scores()} for YouTube URLs}
#'   \item{\code{"findingemo"}}{\code{findingemo-light}, used by the FindingEmo dataset functions}
#'   \item{\code{"gpu"}}{\code{bitsandbytes}, used for 4-bit quantization of EVA-CLIP-8B}
#' }
#' @param download_models Logical. Download the default text, image and
#' sentence-similarity models into the Hugging Face cache (about 1 GB), and,
#' when \code{extras} includes \code{"rag"}, the default embedding model and
#' TinyLLAMA used by \code{rag()} (about 2.3 GB). Default \code{TRUE}.
#' Skipped when \code{HF_HUB_OFFLINE} is set.
#' @param models Character vector of additional Hugging Face model IDs to
#' download, such as \code{"facebook/bart-large-mnli"}. Use it to prepare
#' models for offline use (see \code{\link{setup_cache}}).
#'
#' @details
#' Python requirements are declared with \code{\link[reticulate]{py_require}}.
#' reticulate resolves them with uv into a cached environment, so after the
#' first run later sessions start in under a second. No conda installation or
#' manual uv installation is needed; reticulate downloads uv when it is not
#' on the PATH.
#'
#' PyTorch is installed as a CPU-only build unless an NVIDIA GPU is detected.
#' Set \code{TE_FORCE_CPU=1} to always use the CPU build, or
#' \code{TRANSFOREMOTION_USE_GPU=1} to always use the CUDA build. The choice is
#' made when the package is loaded, so set either variable in
#' \code{.Renviron} or before \code{library(transforEmotion)}, in a new R
#' session.
#'
#' @return Invisibly returns \code{NULL}.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @examples
#' \dontrun{
#' # Core packages and default models
#' setup_modules()
#'
#' # Also install the packages rag() needs
#' setup_modules(extras = "rag")
#'
#' # Prepare an additional model for offline use
#' setup_modules(models = "facebook/bart-large-mnli")
#' }
#'
#' @export
setup_modules <- function(extras = character(), download_models = TRUE,
                          models = character()) {
  extras <- .te_check_extras(extras)
  .te_require(c("core", extras))

  # Start Python so a failed resolution is reported here, not on first use
  torch <- reticulate::import("torch")
  message(
    "Python environment ready: torch ", torch$`__version__`,
    if (isTRUE(torch$cuda$is_available())) " (CUDA)" else " (CPU)"
  )

  # NLTK data llama-index would otherwise download on first import
  if ("rag" %in% extras && !.te_hf_offline()) .te_download_nltk_data()

  defaults <- if (isTRUE(download_models)) {
    c(.te_default_models, if ("rag" %in% extras) .te_default_rag_models)
  }
  if (length(c(defaults, models)) && .te_hf_offline()) {
    message("HF_HUB_OFFLINE is set; skipping model downloads.")
    return(invisible(NULL))
  }
  # The package's own defaults are public, so a stale token cannot break them
  for (repo in defaults) without_hf_token(.te_download_model(repo))
  for (repo in setdiff(models, defaults)) .te_download_model(repo)

  invisible(NULL)
}

#' @noRd
.te_hf_offline <- function() {
  tolower(Sys.getenv("HF_HUB_OFFLINE", unset = "")) %in% c("1", "true", "yes", "on")
}

#' @noRd
# Download a model repository into the Hugging Face cache. Repositories often
# carry the same weights in several formats (ONNX, OpenVINO, TensorFlow, Flax,
# and both safetensors and PyTorch .bin); transformers loads safetensors, or
# .bin when there are none, so the other copies are skipped.
.te_download_model <- function(repo) {
  hub <- reticulate::import("huggingface_hub")
  files <- unlist(hub$list_repo_files(repo))
  ignore <- c(
    "*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.h5", "*.msgpack",
    "*.ot", "*.tflite", "*.gguf", "coreml/*"
  )
  if (any(grepl("\\.safetensors$", files))) ignore <- c(ignore, "*.bin", "*.pt", "*.pth")

  message("Downloading ", repo, " ...")
  hub$snapshot_download(repo_id = repo, ignore_patterns = as.list(ignore))
  invisible(TRUE)
}
