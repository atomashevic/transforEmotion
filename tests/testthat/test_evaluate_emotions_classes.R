# AUROC pairs probs_cols[i] with classes[i]. Without classes, evaluate_emotions()
# takes the classes from probs_cols. These tests need no Python and no network.

# Twelve rows, four per class. The true class has the highest probability, so
# every per-class AUROC is 1 when each column pairs with its own class. The
# small offsets prevent ties.
probs_data <- function(classes) {
  truth <- rep(classes, each = 4)
  probs <- outer(truth, classes, function(t, k) ifelse(t == k, 0.8, 0.1))
  probs <- probs + seq(0, 0.011, length.out = length(probs))
  d <- data.frame(id = seq_along(truth), truth = truth, pred = truth,
                  stringsAsFactors = FALSE)
  d[paste0("prob_", classes)] <- probs
  d
}

test_that("evaluate_emotions pairs probs_cols with the classes they name", {
  emotions <- c("joy", "anger", "sadness")
  d <- probs_data(emotions)
  probs_cols <- paste0("prob_", emotions)

  # Sorted classes (anger, joy, sadness) would pair prob_joy with anger and
  # prob_anger with joy, which gives AUROC 0 for both
  res <- evaluate_emotions(d, probs_cols = probs_cols)
  expect_equal(res$auroc$per_class$class, emotions)
  expect_equal(res$auroc$per_class$auroc, c(1, 1, 1))
  expect_equal(res$auroc$macro, 1)
  expect_identical(res, evaluate_emotions(d, probs_cols = probs_cols,
                                          classes = emotions))

  # Columns named by class, as do.call(rbind, transformer_scores(...)) gives
  names(d)[match(probs_cols, names(d))] <- emotions
  res <- evaluate_emotions(d, probs_cols = emotions)
  expect_equal(res$auroc$per_class$auroc, c(1, 1, 1))
})

test_that("evaluate_emotions results do not change when probs_cols are sorted", {
  emotions <- c("anger", "joy", "sadness")
  d <- probs_data(emotions)
  probs_cols <- paste0("prob_", emotions)
  expect_identical(
    evaluate_emotions(d, probs_cols = probs_cols),
    evaluate_emotions(d, probs_cols = probs_cols, classes = emotions)
  )
})

test_that("evaluate_emotions names argmax predictions by class", {
  emotions <- c("joy", "anger", "sadness")
  d <- probs_data(emotions)
  d$pred <- NULL
  # The predictions were "prob_joy" and so on, which never match the truth
  expect_no_warning(res <- evaluate_emotions(d, probs_cols = paste0("prob_", emotions)))
  expect_equal(res$accuracy, 1)
  expect_equal(res$auroc$macro, 1)
})

test_that("evaluate_emotions scores AUROC for the classes in the data", {
  emotions <- c("joy", "anger", "sadness")
  d <- probs_data(emotions)
  d$prob_fear <- 0.001
  probs_cols <- c(paste0("prob_", emotions), "prob_fear")
  expect_no_warning(res <- evaluate_emotions(d, probs_cols = probs_cols))
  expect_equal(res$auroc$per_class$class, emotions)
  expect_equal(res$auroc$macro, 1)
  # A class that is in neither truth nor pred does not enter the macro F1
  expect_equal(res$f1_macro, evaluate_emotions(d)$f1_macro)
})

test_that("evaluate_emotions warns when probs_cols do not name the classes", {
  emotions <- c("anger", "joy", "sadness")
  d <- probs_data(emotions)
  names(d)[4:6] <- c("p1", "p2", "p3")
  expect_warning(
    res <- evaluate_emotions(d, probs_cols = c("p1", "p2", "p3")),
    "p1 = anger, p2 = joy, p3 = sadness"
  )
  # The columns pair with the sorted classes by position, as before
  expect_equal(res$auroc$per_class$auroc, c(1, 1, 1))
  expect_no_warning(
    evaluate_emotions(d, probs_cols = c("p1", "p2", "p3"), classes = emotions)
  )
})

test_that("evaluate_emotions keeps the type of numeric and factor labels", {
  # Numeric labels with probs_cols in sorted order give the same result as
  # the call with the classes given
  d <- probs_data(c(1, 2, 3))
  probs_cols <- paste0("prob_", 1:3)
  res <- evaluate_emotions(d, probs_cols = probs_cols)
  expect_type(res$auroc$per_class$class, "double")
  expect_identical(res, evaluate_emotions(d, probs_cols = probs_cols,
                                          classes = c(1, 2, 3)))

  # Unsorted probs_cols reorder the numeric labels
  res <- evaluate_emotions(d, probs_cols = paste0("prob_", c(3, 1, 2)))
  expect_identical(res$auroc$per_class$class, c(3, 1, 2))
  expect_equal(res$auroc$macro, 1)

  # Factor labels
  d <- probs_data(c("c1", "c2", "c3"))
  d$truth <- factor(d$truth)
  d$pred <- factor(d$pred, levels = levels(d$truth))
  probs_cols <- paste0("prob_c", 1:3)
  res <- evaluate_emotions(d, probs_cols = probs_cols)
  expect_s3_class(res$auroc$per_class$class, "factor")
  expect_identical(res, evaluate_emotions(d, probs_cols = probs_cols,
                                          classes = factor(c("c1", "c2", "c3"))))
})
