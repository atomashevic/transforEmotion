test_that("without_hf_token temporarily unsets and restores tokens", {
  old_hf <- Sys.getenv("HF_TOKEN", unset = NA_character_)
  old_hub <- Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = NA_character_)

  # Set dummy tokens
  Sys.setenv(HF_TOKEN = "dummy_hf")
  Sys.setenv(HUGGINGFACE_HUB_TOKEN = "dummy_hub")

  inside <- without_hf_token({
    c(hf = Sys.getenv("HF_TOKEN", unset = ""), hub = Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = ""))
  })

  expect_identical(inside[["hf"]], "")
  expect_identical(inside[["hub"]], "")

  # Restored
  expect_identical(Sys.getenv("HF_TOKEN"), "dummy_hf")
  expect_identical(Sys.getenv("HUGGINGFACE_HUB_TOKEN"), "dummy_hub")

  # Cleanup
  if (is.na(old_hf) || !nzchar(old_hf)) Sys.unsetenv("HF_TOKEN") else Sys.setenv(HF_TOKEN = old_hf)
  if (is.na(old_hub) || !nzchar(old_hub)) Sys.unsetenv("HUGGINGFACE_HUB_TOKEN") else Sys.setenv(HUGGINGFACE_HUB_TOKEN = old_hub)
})

test_that("with_hf_token temporarily sets and restores tokens", {
  old_hf <- Sys.getenv("HF_TOKEN", unset = NA_character_)
  old_hub <- Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = NA_character_)

  # Ensure cleared
  Sys.unsetenv("HF_TOKEN")
  Sys.unsetenv("HUGGINGFACE_HUB_TOKEN")

  inside <- with_hf_token("ephemeral123", {
    c(hf = Sys.getenv("HF_TOKEN", unset = ""), hub = Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = ""))
  })

  expect_identical(inside[["hf"]], "ephemeral123")
  expect_identical(inside[["hub"]], "ephemeral123")

  # Restored (both should be empty strings)
  expect_identical(Sys.getenv("HF_TOKEN", unset = ""), "")
  expect_identical(Sys.getenv("HUGGINGFACE_HUB_TOKEN", unset = ""), "")

  # Cleanup restore if pre-set
  if (!is.na(old_hf) && nzchar(old_hf)) Sys.setenv(HF_TOKEN = old_hf)
  if (!is.na(old_hub) && nzchar(old_hub)) Sys.setenv(HUGGINGFACE_HUB_TOKEN = old_hub)
})

test_that(".is_hf_auth_error flags common auth messages", {
  e1 <- simpleError("401 Client Error: Unauthorized for url")
  e2 <- simpleError("Access to this model is gated and requires authorization")
  e3 <- simpleError("Some other unrelated error")
  expect_true(.is_hf_auth_error(e1))
  expect_true(.is_hf_auth_error(e2))
  expect_false(.is_hf_auth_error(e3))
})

