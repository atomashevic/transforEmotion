#' Install GPU Python Modules
#'
#' @description
#' Adds \code{bitsandbytes}, used for 4-bit quantization of the EVA-CLIP-8B
#' vision model, to the Python environment. Equivalent to
#' \code{setup_modules(extras = "gpu", download_models = FALSE)}.
#'
#' @details
#' The CUDA build of PyTorch does not need this function: it is selected
#' automatically when an NVIDIA GPU is detected (see \code{\link{setup_modules}}).
#' \code{image_scores()} and \code{video_scores()} add \code{bitsandbytes}
#' themselves when EVA-CLIP-8B is used on a GPU.
#'
#' @note
#' This function requires NVIDIA GPU and drivers to be properly installed.
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @export
setup_gpu_modules <- function() {
  setup_modules(extras = "gpu", download_models = FALSE)
}
