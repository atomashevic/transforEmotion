# Dependencies of vision models that the Python adapters need at load time.
# The Python side is mocked; no models are downloaded.

test_that("EVA-CLIP is recognised by alias, direct id and custom alias", {
  uses_eva <- transforEmotion:::.te_uses_eva_adapter
  expect_true(uses_eva("eva-8B"))
  expect_true(uses_eva("BAAI/EVA-CLIP-8B-448"))
  expect_false(uses_eva("oai-base"))
  expect_false(uses_eva("jina-v2"))
  expect_false(uses_eva("openai/clip-vit-base-patch32"))
  expect_false(uses_eva("Salesforce/blip-eval-model"))

  register_vision_model("my-eva", "BAAI/EVA-CLIP-8B-448", architecture = "clip-custom")
  withr::defer(remove_vision_model("my-eva", confirm = FALSE))
  expect_true(uses_eva("my-eva"))
  # A registered plain CLIP architecture loads with the CLIP adapter
  register_vision_model("my-clip", "someone/eva-named-clip", architecture = "clip")
  withr::defer(remove_vision_model("my-clip", confirm = FALSE))
  expect_false(uses_eva("my-clip"))
})

test_that("image_scores() adds bitsandbytes on a GPU for a direct EVA-CLIP id", {
  required <- character()
  local_mocked_bindings(
    ensure_te_py_env = function() invisible(TRUE),
    .te_uses_gpu = function() TRUE,
    .te_require = function(features) required <<- c(required, features),
    setup_modules = function(...) stop("stop here")
  )
  # Stop right after the requirements are declared
  local_mocked_bindings(source_python = function(...) stop("stop here"), .package = "reticulate")
  try(suppressMessages(image_scores("x.png", c("a", "b"), model = "BAAI/EVA-CLIP-8B-448")), silent = TRUE)
  expect_true("gpu" %in% required)

  required <- character()
  try(suppressMessages(image_scores("x.png", c("a", "b"), model = "oai-base")), silent = TRUE)
  expect_false("gpu" %in% required)
})

fake_hub <- function(configs, downloaded) {
  root <- withr::local_tempdir(.local_envir = parent.frame(2))
  list(
    list_repo_files = function(repo) list("config.json", "model.safetensors"),
    snapshot_download = function(repo_id, ignore_patterns) {
      downloaded(repo_id)
      path <- file.path(root, gsub("/", "--", repo_id))
      dir.create(path, showWarnings = FALSE)
      if (!is.null(configs[[repo_id]])) {
        jsonlite::write_json(configs[[repo_id]], file.path(path, "config.json"), auto_unbox = TRUE)
      }
      path
    }
  )
}

test_that("preparing a vision-only CLIP checkpoint also caches its base model", {
  got <- character()
  hub <- fake_hub(list(
    "tanganke/clip-vit-large-patch14_fer2013" = list(
      model_type = "clip_vision_model", `_name_or_path` = "openai/clip-vit-large-patch14"),
    "openai/clip-vit-large-patch14" = list(model_type = "clip")
  ), function(repo) got <<- c(got, repo))
  local_mocked_bindings(import = function(...) hub, .package = "reticulate")
  suppressMessages(transforEmotion:::.te_download_model("tanganke/clip-vit-large-patch14_fer2013"))
  expect_equal(got, c("tanganke/clip-vit-large-patch14_fer2013", "openai/clip-vit-large-patch14"))
})

test_that("full checkpoints download only themselves", {
  got <- character()
  hub <- fake_hub(list(
    "openai/clip-vit-base-patch32" = list(model_type = "clip", `_name_or_path` = "openai/other")
  ), function(repo) got <<- c(got, repo))
  local_mocked_bindings(import = function(...) hub, .package = "reticulate")
  suppressMessages(transforEmotion:::.te_download_model("openai/clip-vit-base-patch32"))
  expect_equal(got, "openai/clip-vit-base-patch32")
})

test_that("a cached vision adapter is not reused for another architecture", {
  skip_on_cran()
  skip_if_not(reticulate::py_module_available("transformers"))
  image_py <- system.file("python", "image.py", package = "transforEmotion")
  # As in image_scores(): source into __main__, again on every call, so the
  # adapter cache survives. Adapters load their model lazily, so creating
  # them downloads nothing.
  reticulate::source_python(image_py, envir = NULL)
  first <- reticulate::py$get_vision_adapter("someone/vision-model", NULL, "blip")
  reticulate::source_python(image_py, envir = NULL)
  second <- reticulate::py$get_vision_adapter("someone/vision-model", NULL, "clip")
  expect_equal(first$`__class__`$`__name__`, "BLIPAdapter")
  expect_equal(second$`__class__`$`__name__`, "CLIPAdapter")
})
