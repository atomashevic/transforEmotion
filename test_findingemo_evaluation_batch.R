#!/usr/bin/env Rscript
# FindingEmo Dataset Comparative Evaluation Script (batch, face=none, oai-base)

pkgload::load_all()

cat("🚀 FindingEmo Dataset Evaluation Script (batch, face=none, oai-base)\n")
cat("==================================================================\n\n")

# Install required Python modules
cat("📦 Setting up Python dependencies...\n")
setup_modules()

set.seed(42)

TARGET_DIR <- "./findingemo_eval_300"
MAX_IMAGES <- 300
EMO8_CLASSES <- c("joy", "trust", "fear", "surprise",
                  "sadness", "disgust", "anger", "anticipation")

EMO8_CLASSES_ADJ <- c("joyful", "trusting", "fearful", "surprised",
                      "sad", "disgusted", "angry", "anticipating")

EMO8_CLASSES_ADJ_FE <- c("joyful facial expression", "trusting facial expression", "fearful facial expression", "surprised facial expression",
                         "sad facial expression", "disgusted facial expression", "angry facial expression", "anticipating facial expression")

EMO8_CLASSES_ADJ_P <- c("a person with joyful facial expression", "a person with trusting facial expression", "a person with fearful facial expression", "a person with surprised facial expression",
                        "a person with sad facial expression", "a person with disgusted facial expression", "a person with angry facial expression", "a person with anticipating facial expression")

cat("✓ Setup completed\n")
cat("  Target directory:", TARGET_DIR, "\n")
cat("  Max images:", MAX_IMAGES, "\n")
cat("  Emotion classes:", length(EMO8_CLASSES), "Emo8 labels\n\n")

# 1. DOWNLOAD FINDINGEMO DATASET
cat("📥 STEP 1: Downloading FindingEmo Dataset\n")
cat("------------------------------------------\n")

download_result <- download_findingemo_data(
  target_dir = TARGET_DIR,
  max_images = MAX_IMAGES,
  randomize = TRUE,
  skip_existing = TRUE
)

if (download_result$success) {
  images_dir <- file.path(TARGET_DIR, "images")
  image_files <- if (dir.exists(images_dir)) list.files(images_dir, pattern = "\\.(jpg|jpeg|png|bmp|gif)$", ignore.case = TRUE) else character(0)
  derived_image_count <- length(image_files)
  
  urls_path <- file.path(TARGET_DIR, "urls.json")
  derived_total_selected <- tryCatch({
    if (file.exists(urls_path)) length(jsonlite::fromJSON(urls_path, simplifyVector = TRUE)) else NA_integer_
  }, error = function(e) NA_integer_)
  
  metadata_path <- file.path(TARGET_DIR, "metadata.json")
  derived_failed <- tryCatch({
    if (file.exists(metadata_path)) {
      md <- jsonlite::fromJSON(metadata_path, simplifyVector = TRUE)
      if (!is.null(md$failed_downloads)) md$failed_downloads else NA_integer_
    } else NA_integer_
  }, error = function(e) NA_integer_)
  
  image_count <- if (!is.null(download_result$image_count)) download_result$image_count else derived_image_count
  failed_count <- if (!is.null(download_result$failed_count)) download_result$failed_count else derived_failed
  total_selected <- if (!is.null(download_result$total_selected)) download_result$total_selected else derived_total_selected
  
  cat("✓ Dataset download completed\n")
  cat("  Images downloaded:", image_count, "\n")
  cat("  Failed downloads:", ifelse(is.na(failed_count), "NA", failed_count), "\n")
  cat("  Total selected:", ifelse(is.na(total_selected), "NA", total_selected), "\n")
  success_rate <- if (!is.na(total_selected) && total_selected > 0) round(100 * image_count / total_selected, 1) else NA_real_
  cat("  Success rate:", ifelse(is.na(success_rate), "NA", paste0(success_rate, " %")), "\n\n")
  
  if (isTRUE(image_count == 0)) {
    stop("No images were downloaded successfully. Cannot proceed with evaluation.")
  }
} else {
  cat("✗ Dataset download failed:", download_result$message, "\n")
  stop("Cannot proceed without dataset")
}

# 2. LOAD ANNOTATIONS AND MAP TO EMO8
cat("📋 STEP 2: Loading Annotations and Mapping to Emo8\n")
cat("---------------------------------------------------\n")

annotations <- load_findingemo_annotations(TARGET_DIR)
cat("✓ Loaded", nrow(annotations), "annotations\n")
cat("  Available columns:", paste(names(annotations), collapse = ", "), "\n")

annotations$emo8_label <- map_to_emo8(annotations$emotion)

mapping_summary <- table(annotations$emotion, annotations$emo8_label, useNA = "always")
cat("\n📊 Emotion Mapping Summary:\n")
print(mapping_summary)

valid_annotations <- annotations[!is.na(annotations$emo8_label), ]
cat("\n✓ Retained", nrow(valid_annotations), "annotations with valid Emo8 mappings\n")

emo8_dist <- table(valid_annotations$emo8_label)
cat("\n📈 Emo8 Label Distribution:\n")
print(emo8_dist)
cat("\n")

# 3. MATCH ANNOTATIONS WITH DOWNLOADED IMAGES
cat("🔗 STEP 3: Matching Annotations with Downloaded Images\n")
cat("-------------------------------------------------------\n")

images_dir <- file.path(TARGET_DIR, "images")
downloaded_files <- list.files(images_dir, pattern = "\\.(jpg|jpeg|png|bmp|gif)$", ignore.case = TRUE, full.names = FALSE)
cat("📁 Found", length(downloaded_files), "downloaded image files\n")

full_data <- load_findingemo_annotations(TARGET_DIR, output_format = "list")

valid_annotations$image_file <- basename(valid_annotations$image_path)
matched_annotations <- valid_annotations[valid_annotations$image_file %in% downloaded_files, ]
matched_annotations <- matched_annotations[match(downloaded_files, matched_annotations$image_file, nomatch = 0), ]
matched_annotations <- matched_annotations[matched_annotations$image_file %in% downloaded_files, ]

matched_annotations <- matched_annotations[!is.na(matched_annotations$image_file), ]
eval_annotations <- matched_annotations
downloaded_files_matched <- eval_annotations$image_file

cat("✓ Matched", nrow(eval_annotations), "annotations with downloaded images\n")
cat("📊 Final evaluation dataset size:", nrow(eval_annotations), "samples\n\n")

###############################################################################
# 4. GENERATE PREDICTIONS (BATCH) FOR FOUR LABEL SETS AND EVALUATE
###############################################################################

cat("🎯 STEP 4: Running batch predictions with four label sets (face=none) ...\n")

label_sets <- list(
  emo8_noun   = EMO8_CLASSES,
  emo8_adj    = EMO8_CLASSES_ADJ,
  emo8_adj_fe = EMO8_CLASSES_ADJ_FE,
  emo8_adj_p  = EMO8_CLASSES_ADJ_P
)

run_predictions_for_set_batch <- function(set_name, set_classes, eval_annotations, images_dir, base_classes, model_name) {
  # Run batch scoring
  scores_df <- image_scores_dir(
    dir = images_dir,
    classes = set_classes,
    face_selection = "none",
    model = model_name
  )
  # scores_df has columns: image_id (filename), and one col per class in set_classes
  colnames(scores_df)[1] <- "image_file"

  # Ensure probability columns are numeric
  for (cl in set_classes) {
    if (cl %in% names(scores_df)) {
      scores_df[[cl]] <- suppressWarnings(as.numeric(scores_df[[cl]]))
    }
  }

  # Keep only files we evaluate
  scores_df <- scores_df[scores_df$image_file %in% eval_annotations$image_file, , drop = FALSE]
  # Merge to get numeric id (index)
  merged <- merge(
    eval_annotations[, c("index", "image_file")],
    scores_df,
    by = "image_file",
    all.x = TRUE,
    sort = FALSE
  )
  names(merged)[names(merged) == "index"] <- "image_id"

  # Build output df
  out <- data.frame(
    image_id = merged$image_id,
    image_file = merged$image_file,
    stringsAsFactors = FALSE
  )
  pred_col <- paste0("pred_", set_name)
  out[[pred_col]] <- NA_character_
  # Prefixed prob columns in canonical order
  for (emotion in base_classes) {
    out[[paste0("prob_", set_name, "_", emotion)]] <- NA_real_
  }

  # Determine predictions and map probs positionally
  probs_mat <- as.matrix(merged[, set_classes, drop = FALSE])
  suppressWarnings(storage.mode(probs_mat) <- "numeric")
  for (i in seq_len(nrow(merged))) {
    row_probs <- as.numeric(probs_mat[i, ])
    if (all(is.na(row_probs))) next
    max_idx <- which.max(row_probs)
    out[[pred_col]][i] <- base_classes[max_idx]
    for (j in seq_along(base_classes)) {
      out[[paste0("prob_", set_name, "_", base_classes[j])]][i] <- row_probs[j]
    }
  }

  success <- sum(!is.na(out[[pred_col]]))
  fail <- nrow(out) - success
  cat(sprintf("  • %s: %d success, %d failed (%.1f%%)\n", set_name, success, fail, if (nrow(out) > 0) 100 * success / nrow(out) else 0))
  list(data = out, success = success, fail = fail)
}

cache_path <- file.path(TARGET_DIR, "predictions_cache-batch.rds")
use_cache <- FALSE
overwrite <- Sys.getenv("OVERWRITE_PREDICTIONS", unset = "")
if (file.exists(cache_path) && tolower(overwrite) %in% c("", "false", "0")) {
  cat(sprintf("⚡ Using cached predictions: %s (set OVERWRITE_PREDICTIONS=true to recompute)\n", cache_path))
  use_cache <- TRUE
}

if (use_cache) {
  cache <- readRDS(cache_path)
  predictions_by_set <- cache$predictions_by_set
  merged_preds <- cache$merged_preds
} else {
  predictions_by_set <- list()
  for (nm in names(label_sets)) {
    predictions_by_set[[nm]] <- run_predictions_for_set_batch(
      set_name = nm,
      set_classes = label_sets[[nm]],
      eval_annotations = eval_annotations,
      images_dir = images_dir,
      base_classes = EMO8_CLASSES,
      model_name = "oai-base"
    )
  }
  merged_preds <- Reduce(function(x, y) merge(x, y, by = c("image_id", "image_file"), all = TRUE),
                         lapply(predictions_by_set, `[[`, "data"))
  if (!dir.exists(TARGET_DIR)) dir.create(TARGET_DIR, recursive = TRUE)
  saveRDS(list(predictions_by_set = predictions_by_set, merged_preds = merged_preds), cache_path)
  cat(sprintf("💾 Saved predictions cache to %s\n", cache_path))
}

# Join ground truth
truth_cols <- c("index", "image_file", "emo8_label")
if (all(c("valence", "arousal") %in% names(eval_annotations))) {
  truth_cols <- c(truth_cols, "valence", "arousal")
}
truth_info <- unique(eval_annotations[, truth_cols])
merged_preds <- merge(
  merged_preds,
  truth_info,
  by.x = c("image_id", "image_file"),
  by.y = c("index", "image_file"),
  all.x = TRUE,
  sort = FALSE
)

valid_flags <- lapply(names(label_sets), function(nm) !is.na(merged_preds[[paste0("pred_", nm)]]))
names(valid_flags) <- names(label_sets)
valid_any <- Reduce(`|`, valid_flags)
valid_all <- Reduce(`&`, valid_flags)
cat(sprintf("✓ Valid predictions (any set): %d/%d\n", sum(valid_any), nrow(merged_preds)))
cat(sprintf("✓ Valid predictions (all sets): %d/%d\n\n", sum(valid_all), nrow(merged_preds)))

# 5. EVALUATE EACH LABEL SET
cat("📊 STEP 5: Evaluating each label set...\n")
metrics_rows <- list()
results_by_set <- list()

for (nm in names(label_sets)) {
  pred_col <- paste0("pred_", nm)
  prob_cols <- paste0("prob_", nm, "_", EMO8_CLASSES)
  valid_idx <- !is.na(merged_preds[[pred_col]])
  eval_data_set <- data.frame(
    id = merged_preds$image_id[valid_idx],
    truth = merged_preds$emo8_label[valid_idx],
    pred = merged_preds[[pred_col]][valid_idx],
    stringsAsFactors = FALSE
  )
  eval_data_set[, prob_cols] <- merged_preds[valid_idx, prob_cols]
  if (all(c("valence", "arousal") %in% names(merged_preds))) {
    eval_data_set$valence <- merged_preds$valence[valid_idx]
    eval_data_set$arousal <- merged_preds$arousal[valid_idx]
  }
  if (nrow(eval_data_set) < 1) next
  res <- evaluate_emotions(
    data = eval_data_set,
    truth_col = "truth",
    pred_col = "pred",
    probs_cols = prob_cols,
    classes = EMO8_CLASSES,
    return_plot = FALSE
  )
  balanced_acc <- mean(sapply(unique(eval_data_set$truth), function(class) {
    class_data <- eval_data_set[eval_data_set$truth == class, ]
    if (nrow(class_data) > 0) mean(class_data$truth == class_data$pred) else 0
  }))
  metrics_rows[[nm]] <- data.frame(
    set = nm,
    samples = nrow(eval_data_set),
    accuracy = res$accuracy,
    f1_macro = res$f1_macro,
    f1_micro = res$f1_micro,
    f1_weighted = if (!is.null(res$f1_weighted)) res$f1_weighted else NA_real_,
    precision_macro = res$precision_macro,
    recall_macro = res$recall_macro,
    precision_micro = res$precision_micro,
    recall_micro = res$recall_micro,
    ece = if (!is.null(res$ece)) res$ece else NA_real_,
    brier_score = if (!is.null(res$brier_score)) res$brier_score else NA_real_,
    balanced_accuracy = balanced_acc,
    stringsAsFactors = FALSE
  )
  results_by_set[[nm]] <- list(eval_data = eval_data_set, results = res)
}

metrics_comparison <- do.call(rbind, metrics_rows)

# 6. SAVE OUTPUTS
cat("📝 STEP 6: Writing CSV and TXT summaries...\n")
if (!dir.exists(TARGET_DIR)) dir.create(TARGET_DIR, recursive = TRUE)
combined_predictions <- merged_preds

preds_csv <- file.path(TARGET_DIR, "predictions_all_variants-batch.csv")
write.csv(combined_predictions, preds_csv, row.names = FALSE)

metrics_csv <- file.path(TARGET_DIR, "metrics_comparison-batch.csv")
write.csv(metrics_comparison, metrics_csv, row.names = FALSE)

summary_file <- file.path(TARGET_DIR, "evaluation_comparative_summary-batch.txt")
sink(summary_file)
cat("FindingEmo Comparative Evaluation Summary (batch, face=none, oai-base)\n")
cat("===================================================================\n")
cat("Generated:", Sys.time(), "\n\n")
cat(sprintf("Dataset: FindingEmo-Light (%d requested, %s downloaded)\n", MAX_IMAGES, if (exists("image_count")) as.character(image_count) else "NA"))
cat("Model: transforEmotion (oai-base, batch)\n")
cat("Label sets: emo8_noun, emo8_adj, emo8_adj_fe, emo8_adj_p\n\n")
cat("Metrics by label set:\n"); print(metrics_comparison); cat("\n")
for (nm in names(results_by_set)) {
  cat(sprintf("-- %s details --\n", nm))
  res <- results_by_set[[nm]]$results
  cat(sprintf("Samples: %d\n", nrow(results_by_set[[nm]]$eval_data)))
  cat(sprintf("Accuracy: %.3f\n", res$accuracy))
}
sink()

cat("\n📄 Files written:\n")
cat("  - ", preds_csv, "\n")
cat("  - ", metrics_csv, "\n")
cat("  - ", summary_file, "\n\n")
cat("🎯 Batch script completed successfully!\n")
