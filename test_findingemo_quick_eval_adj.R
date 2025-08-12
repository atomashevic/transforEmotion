  #!/usr/bin/env Rscript

  # Quick Evaluation: FindingEmo 50 images, adjective labels, no-face scoring

  suppressPackageStartupMessages({
    if (requireNamespace("pkgload", quietly = TRUE)) pkgload::load_all(quiet = TRUE)
  })

  cat("\nQuick Evaluation (FindingEmo 50, adjective labels, no faces)\n")
  cat("=============================================================\n\n")

  total_t0 <- Sys.time()

  # Ensure Python environment and modules
  cat("Setting up Python dependencies (if needed)...\n")
  try(setup_modules(), silent = TRUE)

  set.seed(42)

  # Config
  TARGET_DIR <- "./findingemo_eval_quick"
  MAX_IMAGES <- 50
  MODEL <- "oai-base"

  # Emo8 nouns and adjectives (aligned index-wise)
  EMO8 <- c("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
  EMO8_ADJ <- c("joyful", "trusting", "fearful", "surprised", 
                "sad", "disgusted", "angry", "anticipating")

  cat(sprintf("Target dir: %s\n", TARGET_DIR))
  cat(sprintf("Images requested: %d\n", MAX_IMAGES))
  cat(sprintf("Model: %s\n\n", MODEL))

  # 1) Download dataset (+ annotations merged to downloaded images)
  cat("Downloading FindingEmo-Light subset and preparing evaluation data...\n")
  dl <- download_findingemo_data(
    target_dir = TARGET_DIR,
    max_images = MAX_IMAGES,
    randomize = TRUE,
    skip_existing = TRUE
  )

  if (isTRUE(dl$success) && !is.null(dl$evaluation_data) && nrow(dl$evaluation_data) > 0) {
    eval_base <- dl$evaluation_data
    cat(sprintf("✓ Ready evaluation rows: %d (matched images)\n", nrow(eval_base)))
  } else {
    stop("Download or evaluation data preparation failed; cannot proceed.")
  }

  images_dir <- file.path(dl$target_dir, "images")
  if (!dir.exists(images_dir)) stop("Images directory not found: ", images_dir)

  # 2) Batch image scoring with adjective labels and no faces
  cat("\nRunning batch image scoring (adjectives, no faces)...\n")
  infer_t0 <- Sys.time()
  preds <- image_scores_dir(
    dir = images_dir,
    classes = EMO8_ADJ,
    face_selection = "none",
    model = MODEL
  )
  infer_t1 <- Sys.time()
  infer_time <- as.numeric(difftime(infer_t1, infer_t0, units = "secs"))
  cat(sprintf("✓ Inference done in %.2f sec\n", infer_time))

  if (!"image_id" %in% names(preds)) stop("Batch scoring output missing 'image_id' column")

  # 3) Prepare evaluation data: align predictions to ground-truth
  cat("\nPreparing evaluation frame...\n")
  merged <- merge(
    eval_base,
    preds,
    by.x = "image_file",
    by.y = "image_id",
    all.x = FALSE,
    all.y = FALSE
  )

  if (nrow(merged) == 0) stop("No overlap between evaluation set and predictions.")

  # Build minimal evaluation frame; pred will be auto-inferred from probs
  eval_df <- data.frame(
    id = merged$id,
    truth = merged$truth,
    stringsAsFactors = FALSE
  )
  eval_df <- cbind(eval_df, merged[, EMO8_ADJ, drop = FALSE])

  # 4) Evaluate
  cat("\nEvaluating...\n")
  res <- evaluate_emotions(
    data = eval_df,
    truth_col = "truth",
    pred_col = "pred",
    probs_cols = EMO8_ADJ,
    classes = EMO8,
    return_plot = FALSE
  )

  # 5) Print results and timings
  total_t1 <- Sys.time()
  total_time <- as.numeric(difftime(total_t1, total_t0, units = "secs"))

  cat("\nResults\n")
  cat("-------\n")
  cat(sprintf("Samples evaluated: %d\n", nrow(eval_df)))
  if (!is.null(res$accuracy)) cat(sprintf("Accuracy: %.3f\n", res$accuracy))
  if (!is.null(res$f1_macro)) cat(sprintf("F1 (macro): %.3f\n", res$f1_macro))
  if (!is.null(res$f1_micro)) cat(sprintf("F1 (micro): %.3f\n", res$f1_micro))
  if (!is.null(res$precision_macro)) cat(sprintf("Precision (macro): %.3f\n", res$precision_macro))
  if (!is.null(res$recall_macro)) cat(sprintf("Recall (macro): %.3f\n", res$recall_macro))
  if (!is.null(res$auroc) && is.list(res$auroc) && !is.null(res$auroc$macro) && !is.na(res$auroc$macro)) {
    cat(sprintf("AUROC (macro): %.3f\n", res$auroc$macro))
  }
  if (!is.null(res$ece)) cat(sprintf("ECE: %.3f\n", res$ece))

  cat(sprintf("\nTotal inference time: %.2f sec\n", infer_time))
  cat(sprintf("Total script time: %.2f sec\n", total_time))

  invisible(TRUE)
