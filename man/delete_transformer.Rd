% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/delete_transformer.R
\name{delete_transformer}
\alias{delete_transformer}
\title{Delete a Transformer Model}
\usage{
delete_transformer(model_name, delete = FALSE)
}
\arguments{
\item{model_name}{Character vector.
If no model is provided, then a list of models that are locally stored on the
computer are printed}

\item{delete}{Boolean (length = 1).
Should model skip delete question?
Defaults to \code{FALSE}.
Set to \code{TRUE} for less interactive deletion}
}
\value{
Returns list of models or confirmed deletion
}
\description{
Large language models can be quite large and, when stored locally,
can take up a lot of space on your computer. The direct paths to where the
models are on your computer is not necessarily intuitive.

This function quickly identifies the models on your computer and
informs you which ones can be deleted from it to open up storage space
}
\examples{
if(interactive()){
  delete_transformer()
}

}
\author{
Alexander P. Christensen <alexpaulchristensen@gmail.com>
}
