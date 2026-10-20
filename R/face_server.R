# A local web server for the face pages that view_faces() writes. httpuv
# serves the page's folder on 127.0.0.1 only, under a random token, so only
# someone with the URL can open it, and hands POSTs to <token>/api/<name> to
# R functions. Static files are served by httpuv's own thread; the R
# functions run when R is idle at the console or waiting in .te_wait_for().

#' @noRd
# Serves dir. handlers: named list of function(body) returning a list, sent
# back as JSON; body is the request's JSON, parsed. files: page-relative
# names served from other files instead (a data.js that knows the server,
# say). Returns list(url, port, stop).
.te_serve_page <- function(dir, handlers = list(), port = NULL, files = list()) {
  token <- .te_random_token()
  if (is.null(port)) {
    port <- .te_without_rng(httpuv::randomPort(host = "127.0.0.1"))
  }
  port <- as.integer(port)
  prefix <- paste0("/", token)
  # Any address outside the page is a file missing from an empty folder, so
  # httpuv answers it with 404 without waking R
  empty <- tempfile("te-empty-")
  dir.create(empty)
  static <- list(
    httpuv::staticPath(dir, indexhtml = TRUE, fallthrough = FALSE,
                       headers = list("Cache-Control" = "no-store",
                                      "X-Content-Type-Options" = "nosniff")),
    httpuv::excludeStaticPath(),
    httpuv::staticPath(empty, indexhtml = FALSE, fallthrough = FALSE)
  )
  names(static) <- c(prefix, paste0(prefix, "/api"), "/")
  for (name in names(files)) {
    static[[paste0(prefix, "/", name)]] <- httpuv::staticPath(
      files[[name]], headers = list("Cache-Control" = "no-store", "X-Content-Type-Options" = "nosniff")
    )
  }
  app <- list(
    call = function(req) .te_page_api(req, prefix, port, handlers),
    staticPaths = static
  )
  server <- tryCatch(
    httpuv::startServer("127.0.0.1", port, app, quiet = TRUE),
    error = function(e) {
      unlink(empty, recursive = TRUE)
      stop("Cannot start the page's local server on port ", port,
           " (is it in use?). Try another port, or live = FALSE.", call. = FALSE)
    }
  )
  stopped <- FALSE
  list(
    url = sprintf("http://127.0.0.1:%d%s/", port, prefix),
    port = port,
    stop = function() {
      if (!stopped) {
        stopped <<- TRUE
        httpuv::stopServer(server)
        unlink(empty, recursive = TRUE)
      }
      invisible(NULL)
    }
  )
}

#' @noRd
# The server's answer to anything that is not a file of the page
.te_page_api <- function(req, prefix, port, handlers) {
  reply <- function(status, body) {
    list(status = status,
         headers = list("Content-Type" = "application/json", "Cache-Control" = "no-store"),
         body = as.character(jsonlite::toJSON(body, auto_unbox = TRUE, null = "null",
                                              na = "null", json_verbatim = TRUE)))
  }
  # Another site in the browser cannot reach the server through a name that
  # points at 127.0.0.1 (DNS rebinding): the Host must be the server itself
  host <- req$HTTP_HOST
  if (is.null(host) || !host %in% paste0(c("127.0.0.1:", "localhost:"), port)) {
    return(reply(403L, list(error = "Forbidden.")))
  }
  path <- if (is.null(req$PATH_INFO)) "" else req$PATH_INFO
  api <- paste0(prefix, "/api/")
  name <- if (startsWith(path, api)) substring(path, nchar(api) + 1) else ""
  if (!nzchar(name) || !name %in% names(handlers)) return(reply(404L, list(error = "Not found.")))
  if (!identical(req$REQUEST_METHOD, "POST")) return(reply(405L, list(error = "Use POST.")))
  if (!grepl("^application/json", if (is.null(req$CONTENT_TYPE)) "" else req$CONTENT_TYPE)) {
    return(reply(415L, list(error = "Send JSON.")))
  }
  text <- rawToChar(req$rook.input$read())
  Encoding(text) <- "UTF-8"
  # A handler with attribute simplify = FALSE gets the JSON as nested lists
  simplify <- !isFALSE(attr(handlers[[name]], "simplify"))
  body <- tryCatch(jsonlite::parse_json(text, simplifyVector = simplify), error = function(e) NULL)
  if (!is.list(body)) return(reply(400L, list(error = "The request is not a JSON object.")))
  out <- tryCatch(handlers[[name]](body), error = function(e) list(error = conditionMessage(e)))
  if (!is.null(out$error)) return(reply(400L, out))
  reply(200L, out)
}

#' @noRd
# Lets the server answer the page until done() is TRUE. Esc (Ctrl+C in a
# terminal) interrupts it.
.te_wait_for <- function(done) {
  while (!isTRUE(done())) .te_service()
  # once more, so the answer to the page's last request is sent
  .te_service()
  invisible(TRUE)
}

#' @noRd
.te_service <- function() httpuv::service(100)

#' @noRd
# Says where the page is, then opens it in the browser
.te_open_page <- function(server, open, text) {
  hint <- if (nzchar(Sys.getenv("SSH_CONNECTION"))) {
    paste0("\nR runs on a remote computer. To see the page on yours, run ",
           "ssh -N -L ", server$port, ":127.0.0.1:", server$port,
           " <that computer> there, then open the address.")
  } else ""
  message(sprintf(text, server$url), hint)
  if (isTRUE(open)) utils::browseURL(server$url)
  invisible(server$url)
}

#' @noRd
# A token for the page's URL, without touching the user's random numbers
.te_random_token <- function() {
  .te_without_rng(paste(sprintf("%02x", sample.int(256L, 16L, replace = TRUE) - 1L), collapse = ""))
}

#' @noRd
# Evaluates expr with a fresh random seed and restores the user's afterwards,
# so a set.seed() before view_faces() still gives the same numbers after it
.te_without_rng <- function(expr) {
  env <- globalenv()
  had <- exists(".Random.seed", envir = env, inherits = FALSE)
  if (had) seed <- get(".Random.seed", envir = env, inherits = FALSE)
  on.exit({
    if (had) {
      assign(".Random.seed", seed, envir = env)
    } else if (exists(".Random.seed", envir = env, inherits = FALSE)) {
      rm(".Random.seed", envir = env)
    }
  })
  set.seed(NULL)
  expr
}
