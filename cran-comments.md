## Test environments

* Local Arch Linux, R 4.6.1, `R CMD check --as-cran`
* GitHub Actions R-CMD-check: macOS release, Windows release, Ubuntu devel, Ubuntu release, Ubuntu oldrel-1, all with `--as-cran` (NOT_CRAN=true so the test suite runs)

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Days since last update: 1

## Why an update so soon after 0.1.8

0.1.8 was published on 2026-10-25. This update fixes bugs that affect every user of `rag()` and of three other functions. We found them while preparing course materials with the package, too late for 0.1.8:

* `rag()` failed with "got multiple values for keyword argument 'max_new_tokens'" for every Hugging Face model whenever `max_new_tokens` was set.
* When a Gemma 3 model failed to load for a reason other than authentication, `rag()` silently answered with the previously loaded model instead of raising the error.
* `rag()` ignored its `prompt` argument for the Hugging Face models.
* `sentence_similarity()` failed for every model given by name with the default `device = "auto"`.
* `emoxicon_scores()` returned a data frame that dplyr and tidyr reject, because its subclass came after "data.frame". The `emotions` dataset had the same class order and was resaved with the fix; only its class attribute changed.
* The printed summary of `evaluate_emotions()` listed the wrong classes unless the data were ordered id, truth, prediction.

The update also adds Qwen 2.5 Instruct (1.5B and 3B) to `rag()` and `rag_sentemo()`. They load with the Python packages transforEmotion installs and need no Hugging Face account, and a course that starts on 2026-11-04 uses them.

No function or argument was removed; the `transformer` argument of `rag()` and `rag_sentemo()` gains the two Qwen choices. `emoxicon_scores()` no longer installs a package from GitHub, so no function uses `remotes` any more; it stays in Suggests.
