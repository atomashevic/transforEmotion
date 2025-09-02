#' RAG JSON utilities
#'
#' Helpers for validating, parsing, and flattening structured output
#' from `rag(output = "json"|"table")`.
#'
#' @name rag_json_utils
NULL

#' @noRd
rag_json_schema <- function()
{
  list(
    labels = "character[]",
    confidences = "numeric[] (0..1)",
    intensity = "numeric (0..1)",
    evidence_chunks = list(
      type = "array",
      items = list(
        doc_id = "character",
        span = "character",
        score = "numeric"
      )
    )
  )
}

#' Validate a RAG JSON structure
#'
#' Ensures the object has the expected fields and types:
#' `labels`, `confidences`, `intensity`, and `evidence_chunks`.
#'
#' @param x A list (parsed JSON) to validate.
#' @param error Logical; if TRUE, throws an error on invalid input.
#' @return Invisibly returns TRUE when valid; otherwise FALSE or error.
#' @export
validate_rag_json <- function(x, error = TRUE)
{
  ok <- TRUE
  msg <- NULL

  # Must be a list
  if (!is.list(x)) { ok <- FALSE; msg <- "Input must be a list (parsed JSON)." }

  # Required names
  required <- c("labels", "confidences", "intensity", "evidence_chunks")
  if (ok && !all(required %in% names(x))) {
    ok <- FALSE
    missing <- paste(setdiff(required, names(x)), collapse = ", ")
    msg <- paste0("Missing required fields: ", missing)
  }

  # labels
  if (ok && !is.null(x$labels)) {
    if (!is.atomic(x$labels) || !is.character(x$labels)) {
      ok <- FALSE; msg <- "'labels' must be a character vector."
    }
  }

  # confidences
  if (ok && !is.null(x$confidences)) {
    if (!is.atomic(x$confidences) || !is.numeric(x$confidences)) {
      ok <- FALSE; msg <- "'confidences' must be a numeric vector."
    } else if (length(x$confidences) != length(x$labels)) {
      ok <- FALSE; msg <- "'confidences' must match length/order of 'labels'."
    } else if (any(is.na(x$confidences)) || any(x$confidences < 0 | x$confidences > 1)) {
      ok <- FALSE; msg <- "'confidences' values must be in [0,1]."
    }
  }

  # intensity
  if (ok && !is.null(x$intensity)) {
    if (!is.atomic(x$intensity) || !is.numeric(x$intensity) || length(x$intensity) != 1) {
      ok <- FALSE; msg <- "'intensity' must be a single numeric value."
    } else if (is.na(x$intensity) || x$intensity < 0 || x$intensity > 1) {
      ok <- FALSE; msg <- "'intensity' must be in [0,1]."
    }
  }

  # evidence_chunks
  if (ok) {
    ec <- x$evidence_chunks
    if (is.null(ec)) {
      # allow NULL by normalizing to empty list
      x$evidence_chunks <- list()
    } else if (!(is.list(ec))) {
      ok <- FALSE; msg <- "'evidence_chunks' must be a list (array)."
    } else if (length(ec) > 0) {
      for (i in seq_along(ec)) {
        item <- ec[[i]]
        if (!is.list(item) || !all(c("doc_id", "span", "score") %in% names(item))) {
          ok <- FALSE; msg <- "Each evidence chunk must have doc_id, span, score."; break
        }
        if (!is.character(item$doc_id) || length(item$doc_id) != 1) {
          ok <- FALSE; msg <- "evidence_chunks[[i]]$doc_id must be character scalar."; break
        }
        if (!is.character(item$span) || length(item$span) != 1) {
          ok <- FALSE; msg <- "evidence_chunks[[i]]$span must be character scalar."; break
        }
        if (!is.numeric(item$score) || length(item$score) != 1) {
          ok <- FALSE; msg <- "evidence_chunks[[i]]$score must be numeric scalar."; break
        }
      }
    }
  }

  if (!ok) {
    if (isTRUE(error)) stop(msg, call. = FALSE)
    return(FALSE)
  }

  invisible(TRUE)
}

#' Parse RAG JSON
#'
#' Parses a JSON string (or list) matching the enforced RAG schema and returns
#' a normalized list: `list(labels=chr, confidences=num, intensity=num,
#' evidence=data.frame(doc_id, span, score))`.
#'
#' @param x JSON string or list.
#' @param validate Logical; validate structure after parse.
#' @return A normalized list with atomic vectors and an `evidence` data.frame.
#' @examples
#' j <- '{"labels":["joy"],"confidences":[0.9],"intensity":0.8,"evidence_chunks":[]}'
#' parse_rag_json(j)
#' @export
#' @importFrom jsonlite fromJSON toJSON
parse_rag_json <- function(x, validate = TRUE)
{
  obj <- x

  # Accept JSON string or already-parsed list
  if (is.character(x) && length(x) == 1L) {
    txt <- strip_code_fence(x)
    # first attempt
    obj <- try(jsonlite::fromJSON(txt, simplifyVector = TRUE), silent = TRUE)
    if (inherits(obj, "try-error") || !is.list(obj)) {
      # attempt to extract JSON object substring
      txt2 <- extract_first_json_object(txt)
      obj <- jsonlite::fromJSON(txt2, simplifyVector = TRUE)
    }
  }

  if (!is.list(obj)) stop("Unable to parse JSON into a list.", call. = FALSE)

  # Lenient normalization before validation
  # - If confidences is a single scalar but multiple labels exist, replicate to match length
  if (!is.null(obj$labels) && !is.null(obj$confidences)) {
    if (is.atomic(obj$confidences) && is.numeric(obj$confidences) && length(obj$confidences) == 1L) {
      nlab <- tryCatch(length(obj$labels), error = function(e) 1L)
      if (is.finite(nlab) && nlab >= 1L) {
        obj$confidences <- rep(as.numeric(obj$confidences), nlab)
      }
    }
  }

  # - Ensure evidence_chunks exists; then normalize allowing missing fields and character spans
  if (is.null(obj$evidence_chunks)) {
    obj$evidence_chunks <- list()
  }
  if (!is.null(obj$evidence_chunks)) {
    ec <- obj$evidence_chunks
    if (is.list(ec)) {
      norm <- vector("list", length(ec))
      for (i in seq_along(ec)) {
        item <- ec[[i]]
        if (is.character(item) && length(item) == 1L) {
          norm[[i]] <- list(doc_id = as.character(i), span = item, score = NA_real_)
        } else if (is.list(item)) {
          did <- if (!is.null(item$doc_id)) as.character(item$doc_id) else as.character(i)
          spn <- if (!is.null(item$span)) as.character(item$span) else ""
          scr <- if (!is.null(item$score)) as.numeric(item$score) else NA_real_
          norm[[i]] <- list(doc_id = did, span = spn, score = scr)
        } else {
          # Fallback: coerce to string and store as span
          norm[[i]] <- list(doc_id = as.character(i), span = as.character(item), score = NA_real_)
        }
      }
      obj$evidence_chunks <- norm
    }
  }

  if (isTRUE(validate)) validate_rag_json(obj, error = TRUE)

  # Normalize evidence into data.frame for convenience
  ev <- obj$evidence_chunks
  if (is.null(ev) || length(ev) == 0L) {
    ev_df <- data.frame(doc_id = character(0), span = character(0), score = numeric(0))
  } else {
    ev_df <- data.frame(
      doc_id = vapply(ev, function(e) as.character(e$doc_id), character(1)),
      span = vapply(ev, function(e) as.character(e$span), character(1)),
      score = vapply(ev, function(e) as.numeric(e$score), numeric(1)),
      stringsAsFactors = FALSE
    )
  }

  list(
    labels = as.character(obj$labels),
    confidences = as.numeric(obj$confidences),
    intensity = as.numeric(obj$intensity),
    evidence = ev_df
  )
}

#' Convert RAG JSON to a table
#'
#' Produces a long-form data.frame with columns:
#' `label`, `confidence`, `intensity`, `doc_id`, `span`, `score`.
#'
#' @param x JSON string or parsed list.
#' @param validate Logical; validate structure first.
#' @return A data.frame suitable for statistical analysis.
#' @examples
#' j <- '{"labels":["joy","surprise"],"confidences":[0.8,0.5],"intensity":0.7,"evidence_chunks":[]}'
#' as_rag_table(j)
#' @export
as_rag_table <- function(x, validate = TRUE)
{
  parsed <- if (is.list(x) && !is.null(x$labels) && !is.null(x$confidences) && !is.null(x$intensity)) {
    # Might be the raw JSON shape or already normalized by parse_rag_json
    if (!is.null(x$evidence) && is.data.frame(x$evidence)) {
      x
    } else {
      parse_rag_json(x, validate = validate)
    }
  } else {
    parse_rag_json(x, validate = validate)
  }

  labels <- parsed$labels
  confidences <- parsed$confidences
  intensity <- parsed$intensity
  ev <- parsed$evidence

  if (nrow(ev) == 0L) {
    # No evidence: one row per label
    out <- data.frame(
      label = labels,
      confidence = confidences,
      intensity = rep(intensity, length(labels)),
      doc_id = NA_character_,
      span = NA_character_,
      score = NA_real_,
      stringsAsFactors = FALSE
    )
    return(out)
  }

  # Cartesian product: each label x evidence chunk
  out <- do.call(rbind, lapply(seq_along(labels), function(i) {
    data.frame(
      label = labels[[i]],
      confidence = confidences[[i]],
      intensity = intensity,
      doc_id = ev$doc_id,
      span = ev$span,
      score = ev$score,
      stringsAsFactors = FALSE
    )
  }))

  rownames(out) <- NULL
  out
}

# Internal helpers ---------------------------------------------------------

#' @noRd
strip_code_fence <- function(x)
{
  x <- trimws(x)
  # remove ```json ... ``` fences if present
  if (grepl("^```", x)) {
    x <- gsub("^```[a-zA-Z]*\\n?", "", x)
    x <- gsub("```$", "", x)
  }
  x
}

#' @noRd
extract_first_json_object <- function(x)
{
  # Find the first balanced JSON object by tracking brace depth
  start_match <- regexpr("\\{", x)
  if (start_match[1] == -1L) stop("No JSON object found.", call. = FALSE)
  start <- as.integer(start_match[1])
  n <- nchar(x)
  depth <- 0L
  end <- NA_integer_
  for (i in start:n) {
    ch <- substr(x, i, i)
    if (identical(ch, "{")) depth <- depth + 1L else if (identical(ch, "}")) {
      depth <- depth - 1L
      if (depth == 0L) { end <- i; break }
    }
  }
  if (is.na(end)) stop("No JSON object found.", call. = FALSE)
  substr(x, start, end)
}
