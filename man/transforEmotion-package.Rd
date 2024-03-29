% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/transforEmotion.R
\docType{package}
\name{transforEmotion-package}
\alias{transforEmotion}
\alias{transforEmotion-package}
\title{transforEmotion--package}
\description{
Implements sentiment and emotion analysis using \href{https://huggingface.co}{huggingface} transformer
zero-shot classification model pipelines on text and image data. The default text pipeline is
\href{https://huggingface.co/cross-encoder/nli-distilroberta-base}{Cross-Encoder's DistilRoBERTa} and default image/video pipeline
is \href{https://huggingface.co/openai/clip-vit-base-patch32}{Open AI's CLIP}. All other zero-shot classification model pipelines can be implemented using
their model name from \href{https://huggingface.co/models?pipeline_tag=zero-shot-classification}{https://huggingface.co/models?pipeline_tag=zero-shot-classification}.
}
\references{
Yin, W., Hay, J., & Roth, D. (2019).
Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
arXiv preprint arXiv:1909.00161.
}
\author{
Alexander P. Christensen <alexpaulchristensen@gmail.com>, Hudson Golino <hfg9s@virginia.edu> and Aleksandar Tomasevic <atomashevic@ff.uns.ac.rs>
}
