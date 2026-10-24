#' transforEmotion--package
#' 
#' @description Zero-shot emotion classification for text, images, video and speech, and
#' tools to find faces and code them by hand. Text is scored with
#' \href{https://huggingface.co}{Hugging Face} transformer pipelines (default
#' \href{https://huggingface.co/cross-encoder/nli-distilroberta-base}{cross-encoder/nli-distilroberta-base}).
#' Images and video are scored with \href{https://huggingface.co/openai/clip-vit-base-patch32}{OpenAI CLIP};
#' faces can be limited to those found by OpenCV YuNet. Speech segments are scored
#' with LAION CLAP. Frame scores can be calibrated and combined into posteriors, and
#' detected faces or other items can be turned into a local coding task served from R.
#' Any other zero-shot model can be selected by its name on
#' \href{https://huggingface.co/models}{Hugging Face}.
#' 
#' @references
#' Tomašević, A., Golino, H., & Christensen, A. P. (2026).
#' transforEmotion: An open-source R package for emotion analysis using transformer-based generative AI models.
#' \emph{Computational Communication Research}, \emph{8}(2).
#' \doi{10.5117/CCR2026.2.2.TOMA}
#'
#' Yin, W., Hay, J., & Roth, D. (2019).
#' Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
#' arXiv preprint arXiv:1909.00161.
#' 
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>, Hudson Golino <hfg9s@virginia.edu> and Aleksandar Tomasevic <atomashevic@gmail.com>
#' 
#' @seealso \code{citation("transforEmotion")}
#'
#' @importFrom googledrive as_id drive_auth drive_download
#' @importFrom httpuv excludeStaticPath randomPort service startServer staticPath stopServer
#' @importFrom pbapply pblapply
#' @importFrom utils packageDescription
#'
"_PACKAGE"
#> [1] "_PACKAGE"
