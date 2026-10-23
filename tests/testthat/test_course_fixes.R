skip_on_cran()

# A fake cgroup tree under a temp folder: files written as the kernel shows them
write_cgroup <- function(dir, files) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  for (f in names(files)) writeLines(files[[f]], file.path(dir, f))
}

test_that("linux_available_bytes reads MemAvailable in KiB", {
  meminfo <- tempfile()
  writeLines(c("MemTotal:       16000000 kB", "MemFree:          900000 kB",
               "MemAvailable:    2000000 kB"), meminfo)
  expect_equal(transforEmotion:::linux_available_bytes(meminfo), 2000000 * 1024)
  expect_true(is.na(transforEmotion:::linux_available_bytes(tempfile())))
})

test_that("cgroup_available_bytes takes the tightest v2 limit up the tree", {
  root <- tempfile("cgroup")
  self <- tempfile()
  writeLines("0::/user.slice/session.scope", self)
  dir.create(root); writeLines("cpuset cpu io memory pids", file.path(root, "cgroup.controllers"))
  # Root: no files of its own. Parent: 14 GiB limit, 10 GiB used, 1 GiB of it
  # reclaimable page cache. Leaf: no limit
  write_cgroup(file.path(root, "user.slice"), list(
    memory.max = as.character(14 * 2^30), memory.current = as.character(10 * 2^30),
    memory.stat = c("anon 9663676416", paste("inactive_file", 2^30))
  ))
  write_cgroup(file.path(root, "user.slice", "session.scope"), list(
    memory.max = "max", memory.current = as.character(2^30), memory.stat = "inactive_file 0"
  ))
  expect_equal(transforEmotion:::cgroup_available_bytes(root, self), 5 * 2^30)

  # A tighter limit on the leaf wins
  writeLines(as.character(4 * 2^30), file.path(root, "user.slice", "session.scope", "memory.max"))
  expect_equal(transforEmotion:::cgroup_available_bytes(root, self), 3 * 2^30)
})

test_that("cgroup_available_bytes is NA without a limit, and reads cgroup v1", {
  root <- tempfile("cgroup")
  self <- tempfile()
  writeLines("0::/", self)
  write_cgroup(root, list(memory.max = "max", memory.current = "1000", cgroup.controllers = "memory"))
  expect_true(is.na(transforEmotion:::cgroup_available_bytes(root, self)))

  # cgroup v1: no "0::" line; the memory controller has its own folder. In a
  # container the listed path is the host's, so the top folder is read
  unlink(root, recursive = TRUE)
  writeLines(c("12:memory:/docker/abc", "1:name=systemd:/docker/abc"), self)
  write_cgroup(file.path(root, "memory"), list(
    memory.limit_in_bytes = as.character(8 * 2^30),
    memory.usage_in_bytes = as.character(6 * 2^30),
    memory.stat = paste("total_inactive_file", 2^30)
  ))
  expect_equal(transforEmotion:::cgroup_available_bytes(root, self), 3 * 2^30)
  writeLines("9223372036854771712", file.path(root, "memory", "memory.limit_in_bytes"))
  expect_true(is.na(transforEmotion:::cgroup_available_bytes(root, self)))
})

test_that("evaluate_emotions summary uses the named truth and pred columns", {
  # truth and pred are not the 2nd and 3rd columns
  d <- data.frame(id = 1:4, text = c("a", "b", "c", "d"), confidence = c(.9, .8, .7, .6),
                  gold = c("joy", "joy", "anger", "fear"),
                  guess = c("joy", "anger", "anger", "joy"), stringsAsFactors = FALSE)
  res <- evaluate_emotions(d, truth_col = "gold", pred_col = "guess",
                           metrics = c("accuracy"))
  expect_equal(res$summary$classes, c("anger", "fear", "joy"))
  expect_equal(res$summary$n_classes, 3)
  expect_output(print(res), "Classes: 3 \\(anger, fear, joy\\)")
})

test_that("validate_rag_json accepts evidence_chunks as jsonlite makes them", {
  j <- '{"labels":["joy","surprise"],"confidences":[0.8,0.5],"intensity":0.7,
         "evidence_chunks":[{"doc_id":"d1","span":"a","score":0.9},{"doc_id":"d2","span":"b","score":0.4}]}'
  simplified <- jsonlite::fromJSON(j)
  expect_s3_class(simplified$evidence_chunks, "data.frame")
  expect_true(validate_rag_json(simplified, error = FALSE))
  expect_true(validate_rag_json(jsonlite::fromJSON(j, simplifyDataFrame = FALSE), error = FALSE))

  # A data frame still has to carry the required fields
  simplified$evidence_chunks$score <- NULL
  expect_false(validate_rag_json(simplified, error = FALSE))
})

test_that("emoxicon_scores returns a data frame dplyr (vctrs) accepts", {
  data("emotions", package = "transforEmotion", envir = environment())
  expect_equal(class(emotions), c("emoxicon", "data.frame"))
  scores <- emoxicon_scores(c("I am so happy today", "This makes me sad and angry"), lexicon = emotions)
  expect_equal(class(scores), c("emoxicon", "emotions", "data.frame"))
  # With "data.frame" before the subclasses, vctrs rejected all three (and
  # tidyr's pivot_longer())
  expect_equal(nrow(dplyr::filter(scores, doc_id == "doc1")), 1)
  expect_equal(nrow(dplyr::bind_rows(scores, scores)), 4)
  expect_equal(nrow(dplyr::bind_rows(emotions[1:2, ], emotions[3, ])), 3)
})

test_that("emoxicon_scores does not install packages", {
  body_text <- paste(deparse(body(emoxicon_scores)), collapse = "\n")
  expect_false(grepl("install", body_text))
})

test_that("cgroup_available_bytes walks a host-style cgroup v1 path", {
  root <- tempfile("cgroup")
  self <- tempfile()
  writeLines(c("11:memory:/user.slice/user-1007.slice/session-1.scope",
               "1:name=systemd:/user.slice/user-1007.slice/session-1.scope"), self)
  top <- file.path(root, "memory")
  unlimited <- "9223372036854771712"
  write_cgroup(top, list(memory.limit_in_bytes = unlimited, memory.usage_in_bytes = "0"))
  write_cgroup(file.path(top, "user.slice", "user-1007.slice"), list(
    memory.limit_in_bytes = as.character(8 * 2^30),
    memory.usage_in_bytes = as.character(5 * 2^30),
    memory.stat = paste("total_inactive_file", 2^30)
  ))
  write_cgroup(file.path(top, "user.slice", "user-1007.slice", "session-1.scope"), list(
    memory.limit_in_bytes = unlimited, memory.usage_in_bytes = as.character(2^30)
  ))
  expect_equal(transforEmotion:::cgroup_available_bytes(root, self), 4 * 2^30)
})

test_that("auto_device refuses a CPU load that the 32-bit weights do not fit", {
  local_mocked_bindings(ensure_te_py_env = function(...) invisible(TRUE),
                        get_cpu_memory = function() 13000)
  # Qwen2.5-3B: 6200 MB in 16-bit, about 14260 MB to load in 32-bit
  expect_error(auto_device("cpu", "qwen2.5-3b", verbose = FALSE), "about 14260MB.*13000MB available")
  expect_equal(auto_device("cpu", "tinyllama", verbose = FALSE), "cpu")
})
