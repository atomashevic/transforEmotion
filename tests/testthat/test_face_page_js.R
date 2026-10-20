# The pages' JavaScript (inst/www/face-annotator, inst/www/emotion-annotator).
# Runs the Node tests in tests/js against the installed pages, so R CMD check
# runs them too. Needs
# Node 18 or later on the PATH; skipped without it.

skip_without_node <- function() {
  skip_on_cran()
  node <- Sys.which("node")
  skip_if(!nzchar(node), "Node is not installed")
  version <- tryCatch(system2(node, "--version", stdout = TRUE), error = function(e) "")
  major <- suppressWarnings(as.integer(sub("^v(\\d+).*", "\\1", version[1])))
  skip_if(is.na(major) || major < 18, "Node 18 or later is needed")
  node
}

run_node <- function(node, args, env = character()) {
  out <- withr::with_envvar(env, suppressWarnings(
    system2(node, args, stdout = TRUE, stderr = TRUE)
  ))
  list(ok = is.null(attr(out, "status")) || attr(out, "status") == 0, out = out)
}

test_that("the pages' scripts parse", {
  node <- skip_without_node()
  pages <- system.file("www", c("face-annotator", "emotion-annotator"), package = "transforEmotion",
                       mustWork = TRUE)
  for (script in list.files(pages, pattern = "\\.js$", full.names = TRUE)) {
    run <- run_node(node, c("--check", shQuote(script)))
    expect(run$ok, paste(c(basename(script), run$out), collapse = "\n"))
  }
})

test_that("the pages' logic passes its Node tests", {
  node <- skip_without_node()
  tests <- list.files(test_path("..", "js"), pattern = "\\.test\\.js$", full.names = TRUE)
  skip_if(!length(tests), "tests/js is not available")
  logic <- system.file("www", c("face-annotator", "emotion-annotator"), "logic.js",
                       package = "transforEmotion", mustWork = TRUE)
  run <- run_node(node, c("--test", shQuote(normalizePath(tests))),
                  env = c(TE_FACE_LOGIC = normalizePath(logic[1]), TE_ANNOTATE_LOGIC = normalizePath(logic[2])))
  expect(run$ok, paste(c("node --test failed:", run$out), collapse = "\n"))
})
