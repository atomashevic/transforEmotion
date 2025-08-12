#!/usr/bin/env Rscript
#' FindingEmo Dataset Comparative Evaluation Script
#' 
#' This script performs a comprehensive evaluation and comparison of emotion
#' recognition using the FindingEmo-Light dataset with 50 randomly selected
#' images. It benchmarks four prompt label variants for the image scoring step
#' to assess their impact on downstream metrics.
#' 
#' Steps:
#' 1. Download 50 images from FindingEmo dataset
#' 2. Load annotations and map to Emo8 labels
#' 3. Generate predictions using image_scores with four label sets
#' 4. Evaluate each set and compare metrics
#' 5. Save combined CSV and TXT summaries

# =============================================================================
# SETUP AND DEPENDENCIES
# =============================================================================

pkgload::load_all()

cat("🚀 FindingEmo Dataset Evaluation Script\n")
cat("=====================================\n\n")

# Install required Python modules
cat("📦 Setting up Python dependencies...\n")
setup_modules()

# Set random seed for reproducibility
set.seed(42)

# Define constants
TARGET_DIR <- "./findingemo_eval_50"
MAX_IMAGES <- 50
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

# =============================================================================
# 1. DOWNLOAD FINDINGEMO DATASET
# =============================================================================

cat("📥 STEP 1: Downloading FindingEmo Dataset\n")
cat("------------------------------------------\n")

download_result <- download_findingemo_data(
  target_dir = TARGET_DIR,
  max_images = MAX_IMAGES,
  randomize = TRUE,
  skip_existing = TRUE
)

if (download_result$success) {
  # Derive counts robustly (handle 'skipped' responses that omit counts)
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

# =============================================================================
# 2. LOAD ANNOTATIONS AND MAP TO EMO8
# =============================================================================

cat("📋 STEP 2: Loading Annotations and Mapping to Emo8\n")
cat("---------------------------------------------------\n")

# Load annotations
annotations <- load_findingemo_annotations(TARGET_DIR)
cat("✓ Loaded", nrow(annotations), "annotations\n")

# Show available columns
cat("  Available columns:", paste(names(annotations), collapse = ", "), "\n")

# Map to Emo8 labels
annotations$emo8_label <- map_to_emo8(annotations$emotion)

# Check mapping results
mapping_summary <- table(annotations$emotion, annotations$emo8_label, useNA = "always")
cat("\n📊 Emotion Mapping Summary:\n")
print(mapping_summary)

# Remove unmapped emotions
valid_annotations <- annotations[!is.na(annotations$emo8_label), ]
cat("\n✓ Retained", nrow(valid_annotations), "annotations with valid Emo8 mappings\n")

# Show Emo8 distribution
emo8_dist <- table(valid_annotations$emo8_label)
cat("\n📈 Emo8 Label Distribution:\n")
print(emo8_dist)
cat("\n")

# =============================================================================
# 3. MATCH ANNOTATIONS WITH DOWNLOADED IMAGES
# =============================================================================

cat("🔗 STEP 3: Matching Annotations with Downloaded Images\n")
cat("-------------------------------------------------------\n")

# Get list of downloaded image files
images_dir <- file.path(TARGET_DIR, "images")
downloaded_files <- list.files(
  images_dir,
  pattern = "\\.(jpg|jpeg|png|bmp|gif)$",
  ignore.case = TRUE,
  full.names = FALSE
)

cat("📁 Found", length(downloaded_files), "downloaded image files\n")

# Load full dataset to get URLs mapping
full_data <- load_findingemo_annotations(TARGET_DIR, output_format = "list")

# Match valid annotations to downloaded image files by filename
valid_annotations$image_file <- basename(valid_annotations$image_path)
matched_annotations <- valid_annotations[valid_annotations$image_file %in% downloaded_files, ]
matched_annotations <- matched_annotations[match(downloaded_files, matched_annotations$image_file, nomatch = 0), ]
matched_annotations <- matched_annotations[matched_annotations$image_file %in% downloaded_files, ]

# Remove any NA matches and keep only those with both annotation and file
matched_annotations <- matched_annotations[!is.na(matched_annotations$image_file), ]
eval_annotations <- matched_annotations
downloaded_files_matched <- eval_annotations$image_file

cat("✓ Matched", nrow(eval_annotations), "annotations with downloaded images\n")
cat("📊 Final evaluation dataset size:", nrow(eval_annotations), "samples\n\n")

###############################################################################
# 4. GENERATE PREDICTIONS FOR FOUR LABEL SETS AND EVALUATE
###############################################################################

cat("🎯 STEP 4: Running comparative predictions with four label sets...\n")

# Define the four label sets to benchmark (names used as prefixes)
label_sets <- list(
  emo8_noun   = EMO8_CLASSES,
  emo8_adj    = EMO8_CLASSES_ADJ,
  emo8_adj_fe = EMO8_CLASSES_ADJ_FE,
  emo8_adj_p  = EMO8_CLASSES_ADJ_P
)

# Helper to run predictions for a given label set
run_predictions_for_set <- function(set_name, set_classes, eval_annotations, images_dir, base_classes) {
  n <- nrow(eval_annotations)
  df <- data.frame(
    image_id = eval_annotations$index,
    image_file = eval_annotations$image_file,
    stringsAsFactors = FALSE
  )
  pred_col <- paste0("pred_", set_name)
  df[[pred_col]] <- NA_character_
  # Add prefixed probability columns in canonical Emo8 order
  for (emotion in base_classes) {
    df[[paste0("prob_", set_name, "_", emotion)]] <- NA_real_
  }

  success <- 0
  fail <- 0
  cat(sprintf("  • %s: processing %d images...\n", set_name, n))

  for (i in seq_len(n)) {
    image_file <- df$image_file[i]
    image_path <- file.path(images_dir, image_file)
    if (file.exists(image_path)) {
      tryCatch({
        scores <- image_scores(
          image = image_path,
          classes = set_classes,
          model = "oai-base"
        )
        # Map top index back to canonical Emo8 label
        max_idx <- which.max(scores[1, ])
        df[[pred_col]][i] <- base_classes[max_idx]
        # Store probabilities in canonical order
        for (j in seq_along(base_classes)) {
          prob_name <- paste0("prob_", set_name, "_", base_classes[j])
          df[[prob_name]][i] <- scores[1, j]
        }
        success <- success + 1
        if (success %% 5 == 0 || success == n) {
          cat(sprintf("    ✓ %s: %d/%d done\r", set_name, success, n))
        }
      }, error = function(e) {
        fail <<- fail + 1
        cat(sprintf("\n    ⚠ %s failed on %s: %s\n", set_name, image_file, e$message))
      })
    } else {
      fail <- fail + 1
      cat(sprintf("\n    ⚠ %s missing file: %s\n", set_name, image_path))
    }
  }

  cat(sprintf("\n  %s summary: %d success, %d failed (%.1f%%)\n\n", set_name, success, fail, 100 * success / n))
  list(data = df, success = success, fail = fail)
}

cache_path <- file.path(TARGET_DIR, "predictions_cache.rds")
use_cache <- FALSE
overwrite <- Sys.getenv("OVERWRITE_PREDICTIONS", unset = "")
if (file.exists(cache_path) && tolower(overwrite) %in% c("", "false", "0")) {
  # Ask user only in interactive sessions
  if (interactive()) {
    ans <- readline(prompt = sprintf("Cached predictions found at %s. Recompute? [y/N]: ", cache_path))
    use_cache <- !(tolower(ans) %in% c("y", "yes"))
  } else {
    cat(sprintf("⚡ Using cached predictions: %s (set OVERWRITE_PREDICTIONS=true to recompute)\n", cache_path))
    use_cache <- TRUE
  }
}

if (use_cache) {
  cache <- readRDS(cache_path)
  predictions_by_set <- cache$predictions_by_set
  merged_preds <- cache$merged_preds
} else {
  # Run predictions for each label set
  predictions_by_set <- list()
  for (nm in names(label_sets)) {
    predictions_by_set[[nm]] <- run_predictions_for_set(
      set_name = nm,
      set_classes = label_sets[[nm]],
      eval_annotations = eval_annotations,
      images_dir = images_dir,
      base_classes = EMO8_CLASSES
    )
  }
  # Merge predictions
  merged_preds <- Reduce(function(x, y) merge(x, y, by = c("image_id", "image_file"), all = TRUE),
                         lapply(predictions_by_set, `[[`, "data"))
  # Save cache
  if (!dir.exists(TARGET_DIR)) dir.create(TARGET_DIR, recursive = TRUE)
  saveRDS(list(predictions_by_set = predictions_by_set, merged_preds = merged_preds), cache_path)
  cat(sprintf("💾 Saved predictions cache to %s\n", cache_path))
}

# Join ground-truth (and valence/arousal if available) to merged predictions by keys
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

# Determine valid rows per set and overall
valid_flags <- lapply(names(label_sets), function(nm) !is.na(merged_preds[[paste0("pred_", nm)]]))
names(valid_flags) <- names(label_sets)

valid_any <- Reduce(`|`, valid_flags)
valid_all <- Reduce(`&`, valid_flags)

cat(sprintf("✓ Valid predictions (any set): %d/%d\n", sum(valid_any), nrow(merged_preds)))
cat(sprintf("✓ Valid predictions (all sets): %d/%d\n\n", sum(valid_all), nrow(merged_preds)))

###############################################################################
# 5. EVALUATE EACH LABEL SET
###############################################################################

cat("📊 STEP 5: Evaluating each label set...\n")
metrics_rows <- list()
results_by_set <- list()

for (nm in names(label_sets)) {
  pred_col <- paste0("pred_", nm)
  prob_cols <- paste0("prob_", nm, "_", EMO8_CLASSES)

  # Build eval data for this set on its valid subset
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

  if (nrow(eval_data_set) < 1) {
    warning(sprintf("No valid samples for set %s; skipping evaluation", nm))
    next
  }

  # Evaluate
  res <- evaluate_emotions(
    data = eval_data_set,
    truth_col = "truth",
    pred_col = "pred",
  probs_cols = prob_cols,
  classes = EMO8_CLASSES,
    return_plot = FALSE
  )

  # Compute balanced accuracy on this set
  balanced_acc <- mean(sapply(unique(eval_data_set$truth), function(class) {
    class_data <- eval_data_set[eval_data_set$truth == class, ]
    if (nrow(class_data) > 0) mean(class_data$truth == class_data$pred) else 0
  }))

  # Collect metrics
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

###############################################################################
# 6. SAVE COMPARATIVE OUTPUTS (CSV + TXT) AND PRINT SUMMARY
###############################################################################

cat("📝 STEP 6: Writing comparative CSV and TXT summaries...\n")

# Ensure target directory exists
if (!dir.exists(TARGET_DIR)) dir.create(TARGET_DIR, recursive = TRUE)

# 6a. Combined predictions CSV (includes truth and all per-set preds + probs)
combined_predictions <- merged_preds

preds_csv <- file.path(TARGET_DIR, "predictions_all_variants.csv")
write.csv(combined_predictions, preds_csv, row.names = FALSE)

# 6b. Metrics comparison CSV
metrics_csv <- file.path(TARGET_DIR, "metrics_comparison.csv")
write.csv(metrics_comparison, metrics_csv, row.names = FALSE)

# 6c. Comparative TXT summary
summary_file <- file.path(TARGET_DIR, "evaluation_comparative_summary.txt")
sink(summary_file)
cat("FindingEmo Comparative Evaluation Summary\n")
cat("========================================\n")
cat("Generated:", Sys.time(), "\n\n")
cat(sprintf("Dataset: FindingEmo-Light (%d requested, %s downloaded)\n", MAX_IMAGES, if (exists("image_count")) as.character(image_count) else "NA"))
cat("Model: transforEmotion (oai-base)\n")
cat("Label sets benchmarked: emo8_noun, emo8_adj, emo8_adj_fe, emo8_adj_p\n\n")

cat("Metrics by label set:\n")
print(metrics_comparison)
cat("\n")

for (nm in names(results_by_set)) {
  cat(sprintf("-- %s details --\n", nm))
  res <- results_by_set[[nm]]$results
  cat(sprintf("Samples: %d\n", nrow(results_by_set[[nm]]$eval_data)))
  cat(sprintf("Accuracy: %.3f\n", res$accuracy))
  cat(sprintf("F1 (macro/micro): %.3f / %.3f\n", res$f1_macro, res$f1_micro))
  if (!is.null(res$f1_weighted)) cat(sprintf("F1 (weighted): %.3f\n", res$f1_weighted))
  cat(sprintf("Precision (macro/micro): %.3f / %.3f\n", res$precision_macro, res$precision_micro))
  cat(sprintf("Recall (macro/micro): %.3f / %.3f\n", res$recall_macro, res$recall_micro))
  if (!is.null(res$ece)) cat(sprintf("ECE: %.3f\n", res$ece))
  if (!is.null(res$brier_score)) cat(sprintf("Brier: %.3f\n", res$brier_score))
  cat("Confusion matrix:\n"); print(res$confusion_matrix)
  if (!is.null(res$per_class_metrics)) {
    cat("Per-class metrics (rounded):\n")
    pcm <- res$per_class_metrics
    num_cols <- vapply(pcm, is.numeric, logical(1))
    if (any(num_cols)) {
      pcm[, num_cols] <- lapply(pcm[, num_cols, drop = FALSE], function(x) round(x, 3))
    }
    print(pcm)
  }
  cat("\n")
}
sink()

cat("\n📄 Files written:\n")
cat("  - ", preds_csv, "\n")
cat("  - ", metrics_csv, "\n")
cat("  - ", summary_file, "\n\n")

# Console summary
cat("📈 COMPARATIVE RESULTS (key metrics)\n")
cat("====================================\n")
print(metrics_comparison)
cat("\n🎯 Script completed successfully!\n")
