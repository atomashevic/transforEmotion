# view_faces() inside a notebook (Jupyter, Google Colab, VS Code).
#
# A page served by R on 127.0.0.1 cannot be reached when the notebook's
# kernel runs on another computer (Colab), and an R kernel cannot receive
# clicks from its output. So in a notebook view_faces() shows the page in the
# cell's output instead, as one self-contained document in an iframe, and the
# page gives the selection back as a line of R code to run in a new cell.

#' @noRd
# TRUE in a Jupyter kernel for R (IRkernel sets this option), as in Colab
.te_in_notebook <- function() isTRUE(getOption("jupyter.in_kernel"))

#' @noRd
# data: URIs of the display copies (JPEG files)
.te_data_uri <- function(paths) {
  vapply(paths, function(p) {
    paste0("data:image/jpeg;base64,", jsonlite::base64_enc(readBin(p, "raw", file.size(p))))
  }, character(1), USE.NAMES = FALSE)
}

#' @noRd
# The page is UTF-8 text and is handled as bytes throughout, so nothing is
# converted to the session's encoding (a C locale would turn "·" into
# "<U+00B7>")
.te_read_utf8 <- function(path) .te_utf8(rawToChar(readBin(path, "raw", file.size(path))))
.te_utf8 <- function(x) {
  Encoding(x) <- "UTF-8"
  x
}

#' @noRd
# Replaces the first exact occurrence of pattern, byte for byte
.te_replace_once <- function(x, pattern, value) {
  at <- regexpr(pattern, x, fixed = TRUE, useBytes = TRUE)
  if (at < 0) stop("Cannot build the page: '", pattern, "' not found.", call. = FALSE)
  bytes <- charToRaw(x)
  end <- at + nchar(pattern, type = "bytes")
  .te_utf8(rawToChar(c(bytes[seq_len(at - 1)], charToRaw(value),
                       if (end <= length(bytes)) bytes[end:length(bytes)])))
}

#' @noRd
# The page written by .te_write_face_page(notebook = ...) as one HTML document:
# the stylesheet and scripts inline, the images already in data.js
.te_single_file_page <- function(dir) {
  read <- function(f) .te_read_utf8(file.path(dir, f))
  # "</script" inside a script would end it early (in a file label, say)
  script <- function(f) {
    js <- gsub("</(script)", "<\\\\/\\1", read(f), ignore.case = TRUE, perl = TRUE, useBytes = TRUE)
    paste0("<script>", .te_utf8(js), "</script>")
  }
  page <- read("index.html")
  csp <- regmatches(page, regexpr("<meta http-equiv=\"Content-Security-Policy\"[^>]*>", page, useBytes = TRUE))
  page <- .te_replace_once(page, csp, paste0(
    "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; ",
    "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; ",
    "connect-src 'none'; base-uri 'none'; form-action 'none'\">"))
  page <- .te_replace_once(page, "<link rel=\"stylesheet\" href=\"style.css\">",
                           paste0("<style>\n", read("style.css"), "\n</style>"))
  for (f in c("data.js", "logic.js", "app.js")) {
    page <- .te_replace_once(page, sprintf("<script src=\"%s\"></script>", f), script(f))
  }
  .te_utf8(page)
}

#' @noRd
# Shows the page below the cell: an iframe whose document is the page, and a
# small script that sets the iframe's height to the page's
.te_show_in_notebook <- function(page, task_id) {
  html <- .te_read_utf8(page)
  attr_escape <- function(x) {
    x <- gsub("&", "&amp;", x, fixed = TRUE, useBytes = TRUE)
    .te_utf8(gsub("\"", "&quot;", x, fixed = TRUE, useBytes = TRUE))
  }
  id <- paste0("te-", gsub("[^A-Za-z0-9-]", "-", task_id), "-", .te_random_token())
  out <- paste0(
    "<div class=\"te-faces\"><iframe id=\"", id, "\" title=\"Faces\" allow=\"clipboard-write\" ",
    "style=\"width:100%;height:640px;border:1px solid #d8d5ce;border-radius:8px\" srcdoc=\"",
    attr_escape(html), "\"></iframe>",
    "<script>(function () { var f = document.getElementById('", id, "');",
    " window.addEventListener('message', function (e) {",
    " if (f && e.source === f.contentWindow && e.data && e.data.teFacesHeight) {",
    " f.style.height = Math.min(Math.max(e.data.teFacesHeight + 2, 240), 860) + 'px'; } }); })();",
    "</script></div>")
  .te_display_html(.te_utf8(out), page)
}

#' @noRd
# Sends HTML to the notebook; outside a notebook (IRdisplay missing), says
# where the page is
.te_display_html <- function(html, page) {
  if (requireNamespace("IRdisplay", quietly = TRUE)) {
    IRdisplay::display_html(html)
  } else {
    message("Open ", page, " in a web browser to see the faces.")
  }
  invisible(TRUE)
}
