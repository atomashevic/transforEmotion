# Most vignette code is shown without being run (eval = FALSE), so a call
# with an argument that the function does not have is not found when the
# vignettes are built. This test parses every R chunk of every vignette and
# matches each call to an exported function against that function's
# arguments. It needs no Python and no network.

vignette_sources <- function() {
  dirs <- c(testthat::test_path("..", "..", "vignettes"),
            system.file("doc", package = "transforEmotion"))
  dirs <- dirs[nzchar(dirs) & dir.exists(dirs)]
  # character(0), not NULL, when R CMD check runs without built vignettes
  files <- as.character(unlist(lapply(dirs, list.files, pattern = "\\.Rmd$",
                                      full.names = TRUE)))
  files[!duplicated(basename(files))]
}

vignette_chunks <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  starts <- grep("^\\s*```+\\s*\\{[rR]", lines)
  ends <- grep("^\\s*```+\\s*$", lines)
  lapply(starts, function(s) {
    e <- ends[ends > s][1]
    list(line = s, code = lines[seq_len(e - s - 1) + s])
  })
}

calls_in <- function(exprs) {
  out <- list()
  walk <- function(x) {
    if (is.call(x)) {
      out[[length(out) + 1L]] <<- x
      for (a in as.list(x)) if (!missing(a)) walk(a)
    }
  }
  for (e in exprs) walk(e)
  out
}

test_that("vignette code calls package functions with arguments they have", {
  files <- vignette_sources()
  skip_if(length(files) == 0, "No vignette sources found")
  ns <- asNamespace("transforEmotion")
  exported <- getNamespaceExports(ns)
  bad <- character()
  for (f in files) {
    for (ch in vignette_chunks(f)) {
      # Printed output goes in plain ``` blocks, so every R chunk must parse
      exprs <- tryCatch(parse(text = ch$code, keep.source = FALSE),
                        error = function(e) {
                          bad <<- c(bad, sprintf("%s, chunk at line %d: does not parse",
                                                 basename(f), ch$line))
                          NULL
                        })
      for (cl in calls_in(exprs)) {
        fn <- cl[[1]]
        if (is.call(fn) && identical(fn[[1]], as.name("::")) &&
            identical(as.character(fn[[2]]), "transforEmotion")) {
          fn <- fn[[3]]
        }
        if (!is.name(fn) || !as.character(fn) %in% exported) next
        msg <- tryCatch({
          match.call(get(as.character(fn), envir = ns), cl)
          NULL
        }, error = function(e) conditionMessage(e))
        if (!is.null(msg)) {
          bad <- c(bad, sprintf("%s, chunk at line %d: %s(): %s",
                                basename(f), ch$line, as.character(fn), msg))
        }
      }
    }
  }
  expect_identical(bad, character())
})
