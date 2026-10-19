#' Calibrate frame scores for posterior_scores()
#'
#' @description
#' Learns how far to trust a model's frame scores, so that
#' \code{\link{posterior_scores}} can treat them as noisy evidence instead of
#' taking them at face value. Without labels it learns the model's average
#' score for each class, its lean towards some labels before it has seen a
#' face. With the true emotion of some videos it also learns the model's
#' confusions: how often it gives each top label when the true emotion is a
#' given one.
#'
#' @param scores A data frame of scores for one video, one row per frame, as
#'   returned by \code{\link{video_scores}} or \code{\link{face_scores}} (one
#'   face per frame), or a list of such data frames, one per video. Use
#'   scores from the setting you will analyse: the same model, classes and
#'   wording, and similar videos.
#' @param labels Optional. A character vector with the true emotion of each
#'   video, one entry per data frame in \code{scores}; \code{NA} for videos
#'   without one. Every label must be one of \code{classes}.
#' @param classes Character vector. The score columns. Default: every numeric
#'   column except the columns of a faces table (\code{frame}, \code{time},
#'   \code{x}, \code{confidence} and so on).
#' @param alpha Number between 0 and 1: how much one frame's evidence counts.
#'   \code{NULL} (default) estimates it: with labels, as the value that gives
#'   the labelled videos the best clip posteriors (log score, with each video
#'   left out of the confusion matrix it is scored with, and only videos whose
#'   emotion has at least one other labelled video), but never above the
#'   value from the correlation of the evidence between neighbouring frames;
#'   without labels, or with no such video, that value itself.
#'
#' @return An object of class \code{te_calibration}, a list with
#' \describe{
#'   \item{classes}{The score columns, in order.}
#'   \item{prior}{The model's mean score of each class over all frames.}
#'   \item{confusion}{With labels: a matrix whose row \code{c} gives the share
#'     of frames of videos with true emotion \code{c} that have each top label
#'     (add-one smoothed). \code{NULL} without labels.}
#'   \item{likelihood}{\code{"prior"} or \code{"confusion"}: how
#'     \code{\link{posterior_scores}} turns scores into evidence.}
#'   \item{alpha}{The weight of one frame's evidence.}
#'   \item{n_videos, n_frames, n_labelled}{The videos, frames with scores and
#'     labelled videos it was fitted on.}
#' }
#'
#' @details
#' A model's scores are not probabilities to be read as they are. CLIP's
#' scores have a lean of their own: averaged over the frames of CREMA-D clips
#' with six bare labels, standard CLIP gives "neutral" 37\% of its
#' probability, whichever emotion the actor shows. They make the same
#' mistakes again and again, such as reading disgust as sadness. And frames
#' a fraction of a second apart are nearly copies of each other, so 80
#' frames are not 80 independent looks at a face.
#'
#' The calibration turns the score vector \eqn{p_t} of frame \eqn{t} into
#' evidence \eqn{L_t(c)} for each emotion \eqn{c}:
#' \itemize{
#'   \item without labels, \eqn{L_t(c) = (p_t(c) / \pi(c))^\alpha}, where
#'     \eqn{\pi(c)} is the model's mean score for \eqn{c} (\code{prior});
#'   \item with labels, \eqn{L_t(c) = (\sum_k p_t(k) M_{ck})^\alpha}, where
#'     \eqn{M_{ck}} is the share of frames with true emotion \eqn{c} whose top
#'     label is \eqn{k} (\code{confusion}).
#' }
#' \eqn{\alpha} discounts frames that repeat each other: with \eqn{\alpha =
#' 0.1}, ten frames count as one independent look. Without labels it is
#' \eqn{(1 - \rho) / (1 + \rho)}, where \eqn{\rho} is the correlation of the
#' evidence between neighbouring frames, the share of independent
#' observations in a series with that correlation. Videos whose frames all
#' have the same scores are left out of the correlation; if every video is
#' like that, \eqn{\alpha} is 0.01.
#'
#' Scores are renormalised over \code{classes} in every row, which is the
#' same as scoring with only those labels.
#'
#' @examples
#' # Frame scores of nine short clips (in practice from video_scores())
#' set.seed(1)
#' clip <- function(n, emotion) {
#'   s <- matrix(runif(n * 3), n, 3,
#'               dimnames = list(NULL, c("neutral", "happy", "sad")))
#'   s[, "neutral"] <- s[, "neutral"] + 0.6  # the model leans to neutral
#'   s[, emotion] <- s[, emotion] + 0.5
#'   as.data.frame(s / rowSums(s))
#' }
#' emotions <- rep(c("neutral", "happy", "sad"), each = 3)
#' scores <- lapply(emotions, function(e) clip(40, e))
#'
#' # Without labels: correct for the model's lean
#' calibrate_scores(scores)
#'
#' # With labels: also learn its confusions
#' calibration <- calibrate_scores(scores, labels = emotions)
#' calibration
#'
#' @seealso \code{\link{posterior_scores}}
#' @export
calibrate_scores <- function(scores, labels = NULL, classes = NULL, alpha = NULL) {
  videos <- .te_score_videos(scores)
  classes <- .te_score_classes(videos, classes)
  mats <- lapply(videos, function(v) .te_score_matrix(.te_frames_in_order(v), classes))
  all <- do.call(rbind, mats)
  ok <- stats::complete.cases(all)
  if (!any(ok)) stop("No frame in scores has scores for every class.", call. = FALSE)

  prior <- colMeans(all[ok, , drop = FALSE])
  names(prior) <- classes
  counts <- NULL
  n_labelled <- 0L
  if (!is.null(labels)) {
    labels <- as.character(labels)
    if (length(labels) != length(videos)) {
      stop("labels must have one entry per video (", length(videos), ").", call. = FALSE)
    }
    unknown <- setdiff(unique(labels[!is.na(labels)]), classes)
    if (length(unknown)) {
      stop("labels not among the classes: ", paste(unknown, collapse = ", "), call. = FALSE)
    }
    n_labelled <- sum(!is.na(labels))
    if (n_labelled == 0) stop("labels holds no label.", call. = FALSE)
    absent <- setdiff(classes, labels)
    if (length(absent)) {
      warning("No labelled video for ", paste(absent, collapse = ", "),
              "; its row of the confusion matrix is uniform.", call. = FALSE)
    }
    counts <- lapply(mats, .te_top_label_counts, C = length(classes))
  }

  calibration <- structure(list(
    classes = classes,
    prior = prior,
    confusion = if (!is.null(counts)) .te_confusion(counts, labels, classes),
    likelihood = if (is.null(counts)) "prior" else "confusion",
    alpha = 1,
    n_videos = length(videos),
    n_frames = sum(ok),
    n_labelled = n_labelled
  ), class = "te_calibration")

  if (is.null(alpha)) {
    alpha <- if (is.null(counts)) {
      .te_alpha_autocorrelation(mats, calibration)
    } else {
      .te_alpha_fit(mats, labels, counts, calibration)
    }
  } else if (!is.numeric(alpha) || length(alpha) != 1 || is.na(alpha) ||
             alpha <= 0 || alpha > 1) {
    stop("alpha must be a number greater than 0 and at most 1.", call. = FALSE)
  }
  calibration$alpha <- alpha
  calibration
}

#' Posterior emotions from frame scores
#'
#' @description
#' Combines each frame's scores with what the earlier (and, with
#' \code{method = "smooth"}, the later) frames showed, and returns the
#' probability of each emotion in every frame and for the whole video. The
#' scores are treated as noisy evidence, as learned by
#' \code{\link{calibrate_scores}}, and the emotion is assumed to stay the
#' same from one frame to the next with probability \code{stay}.
#'
#' @param scores A data frame of scores for one video, one row per frame, or
#'   a list of them, one per video (see \code{\link{calibrate_scores}}). Rows
#'   are put in time order by \code{time} or \code{frame} when the data frame
#'   has them. A row with a missing score (a frame without a face) adds no
#'   evidence: its posterior comes from the other frames.
#' @param calibration The result of \code{\link{calibrate_scores}}.
#' @param stay Probability that the emotion is the same in the next row
#'   (default 0.99); otherwise it changes to one of the other emotions, each
#'   equally likely. It is per row, so lower it when rows are far apart in
#'   time: 0.99 between frames 1/30 s apart means an emotion lasts about 3 s on
#'   average. Use 1 for one emotion per video.
#' @param method \code{"filter"} (default): each frame's posterior uses that
#'   frame and the ones before it, as when following a video as it plays.
#'   \code{"smooth"}: each frame's posterior uses all frames, before and after
#'   (the forward-backward algorithm).
#'
#' @return A list with
#' \describe{
#'   \item{frames}{\code{scores} (a data frame, or a list of them) in time
#'     order, with the class columns holding the posterior probabilities and a
#'     \code{label} column with the most probable emotion (replacing any
#'     \code{label} column already there). No class may be called
#'     \code{label}, \code{video} or \code{n_frames}.}
#'   \item{clips}{A data frame with one row per video: \code{video} (the name
#'     in the list, or its number for a video without a name; \code{frames}
#'     is named the same way), \code{n_frames}, the posterior of each
#'     class and \code{label}. The clip posterior is the posterior at the last
#'     frame, which has seen every frame; it is the same for both methods. With
#'     \code{stay} near 1 it is the posterior of one emotion for the whole
#'     clip. For long videos in which the emotion changes, summarise
#'     \code{frames} instead, for example with \code{colMeans()}.}
#' }
#'
#' @details
#' The posterior of emotion \eqn{c} in frame \eqn{t} is
#' \deqn{P_t(c) \propto L_t(c) \sum_{c'} T(c' \to c) P_{t-1}(c')}
#' where \eqn{L_t} is the frame's evidence (see \code{\link{calibrate_scores}})
#' and \eqn{T(c \to c) = } \code{stay}. The first frame starts from equal
#' probabilities. This is a hidden Markov model over the emotions, with the
#' model's scores as noisy observations.
#'
#' @examples
#' set.seed(1)
#' clip <- function(n, emotion) {
#'   s <- matrix(runif(n * 3), n, 3,
#'               dimnames = list(NULL, c("neutral", "happy", "sad")))
#'   s[, "neutral"] <- s[, "neutral"] + 0.6
#'   s[, emotion] <- s[, emotion] + 0.5
#'   as.data.frame(s / rowSums(s))
#' }
#' scores <- list(a = clip(40, "happy"), b = clip(40, "sad"),
#'                c = clip(40, "neutral"), d = clip(40, "sad"))
#' calibration <- calibrate_scores(scores)
#'
#' result <- posterior_scores(scores, calibration)
#' result$clips
#' head(result$frames$a)
#'
#' # Each frame using the whole clip
#' smoothed <- posterior_scores(scores$a, calibration, method = "smooth")
#' table(smoothed$frames$label)
#'
#' @seealso \code{\link{calibrate_scores}}, \code{\link{video_scores}}
#' @export
posterior_scores <- function(scores, calibration, stay = 0.99,
                             method = c("filter", "smooth")) {
  if (!inherits(calibration, "te_calibration")) {
    stop("calibration must be the result of calibrate_scores().", call. = FALSE)
  }
  method <- match.arg(method)
  if (!is.numeric(stay) || length(stay) != 1 || is.na(stay) || stay <= 0 || stay > 1) {
    stop("stay must be a number greater than 0 and at most 1.", call. = FALSE)
  }
  single <- is.data.frame(scores)
  videos <- .te_score_videos(scores)
  classes <- .te_score_classes(videos, calibration$classes)

  results <- lapply(videos, function(v) {
    v <- .te_frames_in_order(v)
    if (nrow(v) == 0) stop("A video in scores has no rows.", call. = FALSE)
    evidence <- .te_log_evidence(.te_score_matrix(v, classes), calibration)
    filtered <- .te_forward(evidence, stay)
    posterior <- if (method == "smooth") .te_smooth(evidence, filtered, stay) else filtered
    for (j in seq_along(classes)) v[[classes[j]]] <- posterior[, j]
    v$label <- classes[max.col(posterior, ties.method = "first")]
    list(frames = v, clip = filtered[nrow(filtered), ])
  })

  # The name of each video, or its number where it has none
  ids <- names(videos)
  if (is.null(ids)) {
    ids <- seq_along(videos)
  } else {
    unnamed <- is.na(ids) | ids == ""
    ids[unnamed] <- as.character(which(unnamed))
  }
  clip <- do.call(rbind, lapply(results, `[[`, "clip"))
  colnames(clip) <- classes
  clips <- data.frame(video = ids,
                      n_frames = vapply(results, function(r) nrow(r$frames), integer(1)),
                      stringsAsFactors = FALSE)
  clips <- cbind(clips, as.data.frame(clip, optional = TRUE))
  clips$label <- classes[max.col(clip, ties.method = "first")]
  rownames(clips) <- NULL

  frames <- lapply(results, `[[`, "frames")
  if (!is.null(names(videos))) names(frames) <- ids
  list(frames = if (single) frames[[1]] else frames, clips = clips)
}

#' @export
print.te_calibration <- function(x, ...) {
  cat("Calibration of frame scores (", x$likelihood, " likelihood)\n", sep = "")
  cat("  ", x$n_videos, " video(s), ", x$n_frames, " frames",
      if (x$n_labelled > 0) paste0(", ", x$n_labelled, " labelled"), "\n", sep = "")
  cat("  alpha = ", format(signif(x$alpha, 3)), "\n\n", sep = "")
  cat("Mean score of each class:\n")
  print(round(x$prior, 3))
  if (!is.null(x$confusion)) {
    cat("\nShare of frames with each top label (rows: true emotion):\n")
    print(round(x$confusion, 3))
  }
  invisible(x)
}

# Columns posterior_scores() adds next to the class columns, so no class may have
# these names
.te_posterior_columns <- c("label", "video", "n_frames")

# Columns of a faces table (detect_faces(), face_scores()) that are not scores
.te_face_columns <- c("source", "frame", "frame_index", "time", "face", "track_id",
                      "person_id", "x", "y", "width", "height", "confidence",
                      "detector", "image_width", "image_height")

#' @noRd
.te_score_videos <- function(scores) {
  if (is.data.frame(scores)) return(list(scores))
  if (is.list(scores) && length(scores) > 0 &&
      all(vapply(scores, is.data.frame, logical(1)))) {
    return(scores)
  }
  stop("scores must be a data frame of frame scores or a list of them, one per video.",
       call. = FALSE)
}

#' @noRd
.te_score_classes <- function(videos, classes = NULL) {
  if (is.null(classes)) {
    first <- videos[[1]]
    numeric_cols <- vapply(first, is.numeric, logical(1))
    classes <- names(first)[numeric_cols & !names(first) %in% .te_face_columns]
  }
  classes <- as.character(classes)
  if (length(classes) < 2) {
    stop("scores needs at least two class columns; name them with classes.", call. = FALSE)
  }
  reserved <- intersect(classes, .te_posterior_columns)
  if (length(reserved)) {
    stop("A class cannot be called ", paste0("\"", reserved, "\"", collapse = ", "),
         ": posterior_scores() writes a column of that name. Rename the score column.",
         call. = FALSE)
  }
  for (v in videos) {
    missing <- setdiff(classes, names(v))
    if (length(missing)) {
      stop("scores has no column for: ", paste(missing, collapse = ", "), call. = FALSE)
    }
    if (!all(vapply(v[classes], is.numeric, logical(1)))) {
      stop("The class columns of scores must be numeric.", call. = FALSE)
    }
  }
  classes
}

#' @noRd
# Puts rows in time order and checks there is one face per frame
.te_frames_in_order <- function(v) {
  if ("frame" %in% names(v)) {
    frames <- v$frame[!is.na(v$frame)]
    if (anyDuplicated(frames)) {
      stop("scores has several faces in one frame. Keep one face per frame first, ",
           "for example with select_faces(faces, rule = \"largest\") or ",
           "select_faces(faces, persons = \"p1\").", call. = FALSE)
    }
  }
  key <- if ("time" %in% names(v) && !all(is.na(v$time))) v$time
         else if ("frame" %in% names(v) && !all(is.na(v$frame))) v$frame
  if (!is.null(key) && is.unsorted(key, na.rm = TRUE)) v <- v[order(key), , drop = FALSE]
  v
}

#' @noRd
# Scores as a matrix whose rows sum to 1; rows with a missing score are NA
.te_score_matrix <- function(v, classes) {
  m <- as.matrix(v[classes])
  storage.mode(m) <- "double"
  if (any(m < 0, na.rm = TRUE)) stop("Scores must not be negative.", call. = FALSE)
  m[!stats::complete.cases(m), ] <- NA
  m <- pmax(m, 1e-6)
  m / rowSums(m)
}

#' @noRd
.te_top_label_counts <- function(m, C) {
  m <- m[stats::complete.cases(m), , drop = FALSE]
  tabulate(max.col(m, ties.method = "first"), nbins = C)
}

#' @noRd
# Add-one smoothed share of frames with each top label, per true emotion
.te_confusion <- function(counts, labels, classes, leave_out = NULL) {
  C <- length(classes)
  totals <- matrix(1, C, C, dimnames = list(truth = classes, top = classes))
  for (i in which(!is.na(labels))) {
    if (!identical(i, leave_out)) totals[labels[i], ] <- totals[labels[i], ] + counts[[i]]
  }
  totals / rowSums(totals)
}

#' @noRd
# Log evidence per frame and class, before alpha; rows without scores are 0
.te_raw_evidence <- function(p, calibration, confusion = calibration$confusion) {
  e <- if (calibration$likelihood == "prior") {
    sweep(log(p), 2, log(calibration$prior))
  } else {
    log(p %*% t(confusion))
  }
  e[is.na(e)] <- 0
  e
}

#' @noRd
.te_log_evidence <- function(p, calibration) {
  calibration$alpha * .te_raw_evidence(p, calibration)
}

#' @noRd
# Forward filter: rows are P(emotion in frame t | frames 1..t)
.te_forward <- function(evidence, stay) {
  n <- nrow(evidence)
  C <- ncol(evidence)
  switch_to <- (1 - stay) / (C - 1)
  posterior <- matrix(NA_real_, n, C)
  previous <- rep(1 / C, C)
  for (t in seq_len(n)) {
    prior <- if (t == 1) previous else stay * previous + switch_to * (1 - previous)
    w <- log(prior) + evidence[t, ]
    w <- exp(w - max(w))
    previous <- w / sum(w)
    posterior[t, ] <- previous
  }
  posterior
}

#' @noRd
# Forward-backward smoother: rows are P(emotion in frame t | all frames)
.te_smooth <- function(evidence, filtered, stay) {
  n <- nrow(evidence)
  C <- ncol(evidence)
  switch_to <- (1 - stay) / (C - 1)
  smoothed <- filtered
  beta <- rep(1, C)
  if (n > 1) {
    for (t in (n - 1):1) {
      e <- exp(evidence[t + 1, ] - max(evidence[t + 1, ])) * beta
      beta <- stay * e + switch_to * (sum(e) - e)
      beta <- beta / sum(beta)
      s <- filtered[t, ] * beta
      smoothed[t, ] <- s / sum(s)
    }
  }
  smoothed
}

#' @noRd
# alpha from the lag-1 correlation of the evidence within videos: the share of
# independent observations in a series with that correlation
.te_alpha_autocorrelation <- function(mats, calibration) {
  num <- 0
  den <- 0
  constant <- 0
  for (m in mats) {
    e <- .te_raw_evidence(m, calibration)
    e <- e[stats::complete.cases(m), , drop = FALSE]
    if (nrow(e) < 3) next
    e <- sweep(e, 2, colMeans(e))
    # Every frame the same (a still video, duplicated frames): no variation to
    # correlate, but the frames are copies of each other
    if (max(abs(e)) < 1e-10) {
      constant <- constant + 1
      next
    }
    num <- num + sum(e[-1, ] * e[-nrow(e), ])
    den <- den + sum(e^2)
  }
  # Only videos of copies: weight them as one look per video. No video with
  # three frames (images, very short clips): nothing says frames repeat
  if (den == 0) return(if (constant > 0) 0.01 else 1)
  rho <- num / den
  if (rho <= 0) return(1)
  max(0.01, (1 - rho) / (1 + rho))
}

#' @noRd
# alpha that maximises the mean log posterior of the true label of each
# labelled video, the video left out of the confusion matrix it is scored with.
# Videos whose emotion has no other labelled video are not used: without them
# their row of the matrix is uniform. If no video is left, alpha comes from
# the autocorrelation, as without labels.
.te_alpha_fit <- function(mats, labels, counts, calibration) {
  repeated <- names(which(table(labels) >= 2))
  labelled <- which(!is.na(labels) & labels %in% repeated)
  # The correlation between neighbouring frames bounds alpha from above: the log
  # score can make frames count less, never more, than their independent share
  upper <- .te_alpha_autocorrelation(mats, calibration)
  if (length(labelled) == 0) return(upper)
  totals <- lapply(labelled, function(i) {
    confusion <- .te_confusion(counts, labels, calibration$classes, leave_out = i)
    colSums(.te_raw_evidence(mats[[i]], calibration, confusion))
  })
  truth <- match(labels[labelled], calibration$classes)
  score <- function(alpha) {
    mean(vapply(seq_along(labelled), function(j) {
      s <- alpha * totals[[j]]
      s[truth[j]] - (max(s) + log(sum(exp(s - max(s)))))
    }, numeric(1)))
  }
  stats::optimize(score, c(1e-4, upper), maximum = TRUE)$maximum
}
