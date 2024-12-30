library(transforEmotion)

# Define input files and labels
cat("\n=== INITIALIZATION ===\n")
video_h <- "MAFW/data/00024.mp4"
video_a <- "MAFW/data/00123.mp4"
cat("Processing videos:", video_h, "and", video_a, "\n")

labels <- c("angry", "disgusted", "fearful", "happy", "neutral",
            "sad", "surprised", "contemptuous", "anxious", "helpless", "disappointed")
cat("Using emotion labels:", paste(labels, collapse=", "), "\n\n")

# Process happy video (00024)
cat("=== PROCESSING HAPPY VIDEO (00024) ===\n")
cat("Analyzing video frames...\n")
results_h <- transforEmotion::video_scores(video_h, labels, nframes = 20, ffreq=2)

cat("\nMean scores for each emotion:\n")
mean_scores_h <- sapply(results_h, function(x) mean(x, na.rm = TRUE))
print(round(mean_scores_h, 4))

highest_score_h <- names(which.max(mean_scores_h))
cat("\nPredicted emotion:", highest_score_h, "\n\n")

# Process angry video (00123)
cat("=== PROCESSING ANGRY VIDEO (00123) ===\n")
cat("Analyzing video frames...\n")
results_a <- transforEmotion::video_scores(video_a, labels, nframes = 30, ffreq=2)

cat("\nMean scores for each emotion:\n")
mean_scores_a <- sapply(results_a, function(x) mean(x, na.rm = TRUE))
print(round(mean_scores_a, 4))

highest_score_a <- names(which.max(mean_scores_a))
cat("\nPredicted emotion:", highest_score_a, "\n\n")

# Load and process video descriptions
cat("=== PROCESSING VIDEO DESCRIPTIONS ===\n")
cat("Loading descriptions from CSV...\n")
video_descriptions <- data.frame(read.csv("MAFW/labels/descriptive.csv", header=FALSE))
colnames(video_descriptions) <- c("ID", "Chinese", "English")

# Process happy video description
cat("\n--- Happy Video Description Analysis ---\n")
desc_h <- video_descriptions[2, "English"]
cat("Description:", desc_h, "\n")

cat("Computing transformer scores...\n")
res_desc_h <- transformer_scores(desc_h, labels)
cat("Transformer scores:\n")
res_desc_h

# Process angry video description
cat("\n--- Angry Video Description Analysis ---\n")
desc_a <- video_descriptions[video_descriptions$ID == "00123.mp4", "English"]
cat("Description:", desc_a, "\n")

cat("Computing transformer scores...\n")
res_desc_a <- transformer_scores(desc_a, labels)
cat("Transformer scores:\n")
res_desc_a

# RAG Analysis
cat("\n=== RAG ANALYSIS ===\n")
cat("Processing query for angry video description...\n")
query <- paste("Which of the following emotional facial expressions best fits this video description?",
              "Pick one of these label:", paste(labels, collapse=", "))
cat("Query:", query, "\n\n")

cat("Computing RAG response...\n")
rag_desc_a <- rag(desc_a, query = query)
cat("RAG Response:", "\n")

rag_desc_a

cat("\n=== ANALYSIS COMPLETE ===\n")
