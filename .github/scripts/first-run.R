# First-run smoke test: what a new user runs after installing the package.
# Called by .github/workflows/first-run.yml in a fresh R process with empty
# uv and Hugging Face caches. Usage: Rscript first-run.R [first|second]
#   first   Python setup, model downloads and one call per feature
#   second  a new R session reusing the caches; must not install anything

args <- commandArgs(trailingOnly = TRUE)
phase <- if (length(args)) args[[1]] else "first"
run_rag <- !identical(Sys.getenv("SMOKE_RAG"), "false")

results <- data.frame(phase = character(), step = character(), status = character(),
                      seconds = numeric(), note = character())

step <- function(name, expr) {
  message("\n==> ", name)
  t0 <- Sys.time()
  out <- tryCatch({ force(expr); list(status = "ok", note = "") },
                  error = function(e) list(status = "FAILED", note = conditionMessage(e)))
  secs <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  message("<== ", name, ": ", out$status, " (", secs, " s)",
          if (nzchar(out$note)) paste0("\n", out$note))
  results[nrow(results) + 1, ] <<- list(phase, name, out$status, secs,
                                        gsub("[\r\n|]+", " ", substr(out$note, 1, 200)))
}

check <- function(ok, msg) if (!isTRUE(ok)) stop(msg, call. = FALSE)

message("R ", getRversion(), " on ", R.version$platform,
        "; interactive() = ", interactive())
message("uv on PATH: ", if (nzchar(Sys.which("uv"))) Sys.which("uv") else "no")

step("library(transforEmotion)", library(transforEmotion))

emotions <- c("happiness", "sadness", "anger", "fear", "surprise", "neutral")
image <- system.file("extdata", "boris-1.png", package = "transforEmotion")

if (phase == "first") {
  step("transformer_scores (sets up Python)", {
    s <- transformer_scores(text = "I am so happy to see you today!",
                            classes = c("joy", "anger", "fear"))
    print(s)
    check(names(which.max(s[[1]])) == "joy", "top class is not joy")
  })

  step("torch device", {
    torch <- reticulate::import("torch")
    message("torch ", torch$`__version__`,
            "; cuda: ", torch$cuda$is_available(),
            "; mps: ", torch$backends$mps$is_available())
  })

  step("sentence_similarity", {
    s <- sentence_similarity(
      text = "The cat sat on the mat",
      comparison_text = c("A cat is sitting on a rug", "Stock markets fell sharply today")
    )
    print(s)
    check(s[1, 1] > s[1, 2], "similar sentence did not score higher")
  })

  step("image_scores", {
    s <- image_scores(image, classes = emotions)
    print(round(s, 3))
    check(nrow(s) >= 1, "no faces scored")
  })

  step("video_scores", {
    cv2 <- reticulate::import("cv2", convert = FALSE)
    frame1 <- cv2$imread(image)
    dims <- reticulate::py_to_r(frame1$shape)
    size <- reticulate::tuple(dims[[2]], dims[[1]])
    frame2 <- cv2$resize(cv2$imread(system.file("extdata", "boris-2.png",
                                                package = "transforEmotion")), size)
    video <- file.path(tempdir(), "demo.mp4")
    writer <- cv2$VideoWriter(video, cv2$VideoWriter_fourcc("m", "p", "4", "v"), 25L, size)
    for (i in 1:200) writer$write(if (i <= 100) frame1 else frame2)
    invisible(writer$release())
    s <- video_scores(video, classes = emotions, nframes = 10,
                      save_dir = file.path(tempdir(), "frames"))
    print(round(s, 3))
    check(nrow(s) >= 1, "no frames scored")
  })

  step("vad_scores", {
    s <- vad_scores("We won the final! This is the best day of my life.",
                    input_type = "text")
    print(s)
  })

  if (run_rag) {
    step("rag (TinyLLAMA)", {
      s <- rag(text = c(
        "The team won the championship and fans celebrated in the streets.",
        "After the loss, supporters left the stadium in silence and tears."
      ), query = "What emotions are described?", transformer = "TinyLLAMA")
      print(s)
    })
  }
} else {
  step("transformer_scores (cached environment)", {
    s <- transformer_scores(text = "I am so happy to see you today!",
                            classes = c("joy", "anger", "fear"))
    check(names(which.max(s[[1]])) == "joy", "top class is not joy")
  })
  step("image_scores (cached model)", image_scores(image, classes = emotions))
}

print(results[, c("step", "status", "seconds")], row.names = FALSE)

summary_file <- Sys.getenv("GITHUB_STEP_SUMMARY")
if (nzchar(summary_file)) {
  lines <- c(
    sprintf("### %s run: %s", phase, R.version$platform),
    "", "| step | status | seconds | note |", "|---|---|---|---|",
    sprintf("| %s | %s | %s | %s |", results$step, results$status,
            results$seconds, results$note), ""
  )
  cat(lines, file = summary_file, sep = "\n", append = TRUE)
}

if (any(results$status != "ok")) quit(status = 1)
