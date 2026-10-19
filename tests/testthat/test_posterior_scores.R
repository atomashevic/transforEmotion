# calibrate_scores() and posterior_scores() run in R only: no Python needed

classes <- c("neutral", "happy", "sad")

fake_clip <- function(n, emotion, lean = 0.6, seed = 1) {
  set.seed(seed)
  s <- matrix(runif(n * 3), n, 3, dimnames = list(NULL, classes))
  s[, "neutral"] <- s[, "neutral"] + lean
  s[, emotion] <- s[, emotion] + 0.5
  as.data.frame(s / rowSums(s))
}

clips <- list(a = fake_clip(40, "happy", seed = 1), b = fake_clip(40, "sad", seed = 2),
              c = fake_clip(40, "neutral", seed = 3), d = fake_clip(40, "sad", seed = 4))

test_that("calibrate_scores() without labels learns the mean score of each class", {
  calibration <- calibrate_scores(clips)
  expect_s3_class(calibration, "te_calibration")
  expect_equal(calibration$likelihood, "prior")
  expect_null(calibration$confusion)
  expect_equal(unname(calibration$prior), unname(colMeans(do.call(rbind, clips))))
  expect_equal(calibration$n_videos, 4)
  expect_equal(calibration$n_frames, 160)
  expect_true(calibration$alpha > 0 && calibration$alpha <= 1)
  expect_output(print(calibration), "prior likelihood")
})

test_that("face-table columns are not taken for classes", {
  v <- cbind(data.frame(frame = 1:40, time = (0:39) / 30, x = 10, width = 50, confidence = 0.9),
             clips$a)
  expect_equal(calibrate_scores(v)$classes, classes)
  expect_error(calibrate_scores(clips$a[, 1, drop = FALSE]), "at least two class columns")
})

test_that("labels give a confusion matrix whose rows sum to one", {
  labels <- c("happy", "sad", "neutral", "sad")
  calibration <- calibrate_scores(clips, labels = labels)
  expect_equal(calibration$likelihood, "confusion")
  expect_equal(dim(calibration$confusion), c(3, 3))
  expect_equal(unname(rowSums(calibration$confusion)), rep(1, 3))
  expect_equal(calibration$n_labelled, 4)
  expect_error(calibrate_scores(clips, labels = labels[1:3]), "one entry per video")
  expect_error(calibrate_scores(clips, labels = c("happy", "sad", "angry", NA)), "not among the classes")
  expect_warning(calibrate_scores(clips, labels = c("happy", NA, NA, NA)), "No labelled video")
})

test_that("frames that are all the same get the smallest alpha", {
  still <- lapply(clips, function(v) v[rep(1, 30), ])
  expect_equal(calibrate_scores(still)$alpha, 0.01)
  # Also with labels: copies of a frame never count as independent looks
  labelled <- calibrate_scores(still, labels = c("happy", "sad", "neutral", "sad"))
  expect_lte(labelled$alpha, 0.01)
  # Too few frames to measure a correlation: each frame counts in full
  short <- lapply(clips, function(v) v[1:2, ])
  expect_equal(calibrate_scores(short)$alpha, 1)
})

test_that("with labels alpha is fitted only on emotions with two or more labelled videos", {
  # Only "sad" has two labelled videos, so alpha is fitted on them alone
  fitted <- calibrate_scores(clips, labels = c("happy", "sad", "neutral", "sad"))
  expect_true(fitted$alpha > 0.01 && fitted$alpha <= 1)
  # No emotion with two labelled videos: alpha from the autocorrelation, which
  # is near 1 for these independent frames
  single <- calibrate_scores(clips, labels = c("happy", "sad", "neutral", NA))
  expect_gt(single$alpha, 0.5)
})

test_that("alpha can be given and is checked", {
  expect_equal(calibrate_scores(clips, alpha = 0.2)$alpha, 0.2)
  expect_error(calibrate_scores(clips, alpha = 0), "alpha")
  expect_error(calibrate_scores(clips, alpha = 2), "alpha")
})

test_that("alpha is small when neighbouring frames repeat each other", {
  repeated <- lapply(clips, function(v) v[rep(seq_len(8), each = 5), ])
  independent <- calibrate_scores(clips)$alpha
  copies <- calibrate_scores(repeated)$alpha
  expect_lt(copies, independent)
  expect_lt(copies, 0.4)
})

test_that("posterior_scores() returns posteriors per frame and per clip", {
  calibration <- calibrate_scores(clips)
  result <- posterior_scores(clips, calibration)
  expect_named(result, c("frames", "clips"))
  expect_named(result$frames, names(clips))
  expect_equal(nrow(result$clips), 4)
  expect_equal(result$clips$video, names(clips))
  expect_equal(result$clips$n_frames, rep(40L, 4))
  for (f in result$frames) {
    expect_equal(rowSums(f[classes]), rep(1, 40))
    expect_true(all(f$label %in% classes))
  }
  expect_equal(unname(rowSums(result$clips[classes])), rep(1, 4))
  # The clip posterior is the posterior at the last frame
  expect_equal(unlist(result$clips[1, classes]),
               unlist(result$frames$a[40, classes]), ignore_attr = TRUE)

  one <- posterior_scores(clips$b, calibration)
  expect_s3_class(one$frames, "data.frame")
  expect_equal(nrow(one$clips), 1)
})

test_that("videos without a name are numbered by their position", {
  calibration <- calibrate_scores(clips)
  partly <- list(first = clips$a, clips$b, third = clips$c, clips$d)
  result <- posterior_scores(partly, calibration)
  expect_equal(result$clips$video, c("first", "2", "third", "4"))
  expect_named(result$frames, c("first", "2", "third", "4"))
  unnamed <- posterior_scores(unname(clips), calibration)
  expect_equal(unnamed$clips$video, 1:4)
  expect_null(names(unnamed$frames))
})

test_that("correcting for the model's lean finds the shown emotion", {
  # Raw scores favour neutral in every clip; the calibrated posterior does not
  calibration <- calibrate_scores(clips)
  raw <- vapply(clips, function(v) classes[which.max(colMeans(v))], character(1))
  expect_true(all(raw == "neutral"))
  result <- posterior_scores(clips, calibration)
  expect_equal(result$clips$label, c("happy", "sad", "neutral", "sad"))
})

test_that("with stay = 1 the clip posterior is the product of the frames' evidence", {
  calibration <- calibrate_scores(clips, alpha = 0.3)
  result <- posterior_scores(clips$a, calibration, stay = 1)
  p <- as.matrix(clips$a)
  s <- 0.3 * colSums(sweep(log(p), 2, log(calibration$prior)))
  expected <- exp(s - max(s)) / sum(exp(s - max(s)))
  expect_equal(unlist(result$clips[classes]), expected, ignore_attr = TRUE)
})

test_that("smoothing agrees with filtering at the last frame and uses later frames", {
  calibration <- calibrate_scores(clips, alpha = 0.5)
  v <- rbind(clips$c[1:20, ], clips$a[1:20, ])
  filtered <- posterior_scores(v, calibration, stay = 0.95)
  smoothed <- posterior_scores(v, calibration, stay = 0.95, method = "smooth")
  expect_equal(smoothed$frames[40, classes], filtered$frames[40, classes], ignore_attr = TRUE)
  expect_equal(smoothed$clips, filtered$clips)
  # Just before the change, the smoother already sees the happy frames that follow
  expect_gt(smoothed$frames$happy[19], filtered$frames$happy[19])
})

test_that("frames without scores carry the posterior over", {
  calibration <- calibrate_scores(clips, alpha = 0.5)
  v <- clips$a
  v[10:12, classes] <- NA
  result <- posterior_scores(v, calibration, stay = 1)
  expect_equal(result$frames[12, classes], result$frames[9, classes], ignore_attr = TRUE)
  expect_false(anyNA(result$frames[classes]))
})

test_that("rows are put in time order and one face per frame is required", {
  calibration <- calibrate_scores(clips)
  v <- cbind(data.frame(frame = 1:40), clips$a)
  shuffled <- v[c(40:21, 1:20), ]
  expect_equal(posterior_scores(shuffled, calibration)$clips,
               posterior_scores(v, calibration)$clips)
  two_faces <- rbind(v, v[1, ])
  expect_error(posterior_scores(two_faces, calibration), "several faces in one frame")
})

test_that("classes cannot take the names of output columns", {
  v <- clips$a
  names(v)[2] <- "label"
  expect_error(calibrate_scores(v), "cannot be called \"label\"")
  names(v)[2] <- "n_frames"
  expect_error(calibrate_scores(v), "cannot be called \"n_frames\"")
  # A non-numeric label column already in scores is not a class and is replaced
  w <- cbind(clips$a, label = "old")
  result <- posterior_scores(w, calibrate_scores(clips))
  expect_true(all(result$frames$label %in% classes))
})

test_that("inputs are checked", {
  calibration <- calibrate_scores(clips)
  expect_error(posterior_scores(clips, list()), "calibrate_scores")
  expect_error(posterior_scores(clips, calibration, stay = 0), "stay")
  expect_error(posterior_scores(clips$a[, 1:2], calibration), "no column for: sad")
  expect_error(calibrate_scores("not scores"), "data frame")
  negative <- clips$a
  negative$happy[1] <- -1
  expect_error(calibrate_scores(negative), "negative")
})
