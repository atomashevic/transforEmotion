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

  step("detect_faces and face_scores (images)", {
    images <- system.file("extdata", c("boris-1.png", "trump1.jpg", "boris-2.png"),
                          package = "transforEmotion")
    faces <- detect_faces(images)
    print(faces[, c("face", "person_id", "x", "y", "width", "height", "confidence")])
    check(nrow(faces) == 3, "expected one face per image")
    check(identical(faces$person_id, c("p1", "p2", "p1")), "same person not recognised")
    s <- face_scores(select_faces(faces, persons = "p1"), classes = emotions)
    print(s[, c("person_id", emotions)])
    check(nrow(s) == 2 && !anyNA(s[emotions]), "faces of p1 not scored")
  })

  step("image_scores (all faces; Haar)", {
    s <- image_scores(image, classes = emotions, face_selection = "all")
    check(nrow(s) == 1 && !anyNA(s[emotions]), "face_selection = 'all' scored no face")
    s <- image_scores(image, classes = emotions, face_detector = "haar")
    check(!anyNA(s), "the Haar detector found no face")
  })

  step("detect_faces and video_scores (one person)", {
    faces <- detect_faces(video, nframes = 10)
    print(faces[, c("frame", "time", "face", "track_id", "person_id", "width")])
    check(nrow(faces) == 10, "expected one face per frame")
    check(all(faces$track_id == 1) && all(faces$person_id == "p1"), "expected one track and person")
    s <- video_scores(video, classes = emotions, face_selection = select_faces(faces, tracks = 1))
    check(nrow(s) == 10 && !anyNA(s[emotions]), "faces of track 1 not scored")
  })

  step("draw_faces (two-person example picture and clip)", {
    two <- detect_faces(system.file("extdata", "two-people.jpg", package = "transforEmotion"))
    check(identical(two$person_id, c("p1", "p2")), "expected two people")
    check(all(file.exists(draw_faces(two, plot = FALSE))), "picture not drawn")
    clip <- detect_faces(system.file("extdata", "two-people.mp4", package = "transforEmotion"),
                         nframes = 20)
    print(table(clip$track_id, clip$person_id))
    tracks <- unique(as.data.frame(clip)[, c("track_id", "person_id")])
    check(nrow(tracks) == 3 && length(unique(tracks$person_id)) == 2,
          "the person who comes back does not keep their person_id")
    check(length(draw_faces(clip, max_images = 4, plot = FALSE)) == 4, "frames not drawn")
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
  step("detect_faces (cached face model)", {
    check(identical(detect_faces(image)$person_id, "p1"), "person_id missing")
  })
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
