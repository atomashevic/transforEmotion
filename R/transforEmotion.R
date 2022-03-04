#' transforEmotion--package
#' 
#' @description Implements sentiment analysis using \href{https://huggingface.co}{huggingface} transformer
#' zero-shot classification model pipelines. The default pipeline is
#' \href{https://huggingface.co/cross-encoder/nli-distilroberta-base}{Cross-Encoder's DistilRoBERTa}
#' trained on the \href{https://nlp.stanford.edu/projects/snli/}{Stanford Natural Language Inference} (SNLI) and
#'  \href{https://huggingface.co/datasets/multi_nli}{Multi-Genre Natural Language Inference}
#' (MultiNLI) datasets. Using similar models, zero-shot classification transformers have 
#' demonstrated superior performance relative to other natural language processing models (Yin, Hay, & Roth, 2019).
#' All other zero-shot classification model pipelines can be implemented using
#' their model name from \href{https://huggingface.co/models?pipeline_tag=zero-shot-classification}{https://huggingface.co/models?pipeline_tag=zero-shot-classification}.
#' 
#' @references
#' Yin, W., Hay, J., & Roth, D. (2019).
#' Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
#' arXiv preprint arXiv:1909.00161.
#' 
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com> and Hudson Golino <hfg9s@virginia.edu>
#' 
#' @importFrom utils packageDescription
#'
"_PACKAGE"
#> [1] "_PACKAGE"
