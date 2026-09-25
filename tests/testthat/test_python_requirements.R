test_that("Linux x86_64 uses CPU PyTorch wheels unless a GPU is used", {
  cpu <- transforEmotion:::.te_torch_requirements(FALSE, "Linux", "x86_64")
  expect_length(cpu, 2)
  expect_true(all(grepl("download.pytorch.org/whl/cpu/", cpu, fixed = TRUE)))
  expect_match(cpu[1], "^torch @ .*manylinux_2_28_x86_64\\.whl$")
  expect_match(cpu[2], "^torchvision @ ")

  gpu <- transforEmotion:::.te_torch_requirements(TRUE, "Linux", "x86_64")
  expect_true(all(grepl("/whl/cu126/", gpu, fixed = TRUE)))
})

test_that("Windows uses PyPI torch on CPU and CUDA wheels on GPU", {
  expect_equal(
    transforEmotion:::.te_torch_requirements(FALSE, "Windows", "x86-64"),
    c("torch==2.14.0", "torchvision==0.29.0")
  )
  gpu <- transforEmotion:::.te_torch_requirements(TRUE, "Windows", "x86-64")
  expect_true(all(grepl("cu126-cp312-cp312-win_amd64.whl", gpu, fixed = TRUE)))
})

test_that("macOS uses PyPI torch, with the last Intel build on x86_64", {
  expect_equal(
    transforEmotion:::.te_torch_requirements(FALSE, "Darwin", "arm64"),
    c("torch==2.14.0", "torchvision==0.29.0")
  )
  expect_equal(
    transforEmotion:::.te_torch_requirements(FALSE, "Darwin", "x86_64"),
    c("torch==2.2.2", "torchvision==0.17.2")
  )
})

test_that("feature sets are defined and core excludes heavy unused packages", {
  core <- transforEmotion:::.te_py_requirements("core")
  expect_true(any(grepl("^transformers", core)))
  expect_true(any(grepl("^opencv-python-headless", core)))
  expect_false(any(grepl("tensorflow|bitsandbytes|llama-index|pytubefix", core)))

  for (feature in c("rag", "youtube", "findingemo", "gpu")) {
    expect_type(transforEmotion:::.te_py_requirements(feature), "character")
  }
  expect_error(transforEmotion:::.te_py_requirements("nope"), "Unknown Python feature set")
})

test_that("setup code does not modify the Python environment with pip", {
  for (f in c("setup_modules.R", "reticulate_env.R", "setup_gpu_modules.R")) {
    path <- normalizePath(testthat::test_path("..", "..", "R", f), mustWork = FALSE)
    skip_if_not(file.exists(path))
    src <- paste(readLines(path, warn = FALSE), collapse = "\n")
    expect_false(grepl("py_install|pip\"|readline\\(", src), info = f)
  }
})

test_that("python_requirements() combines core and extras for a target platform", {
  reqs <- python_requirements(extras = "rag", gpu = TRUE, sysname = "Windows", machine = "x86-64")
  expect_true(all(grepl("cu126-cp312-cp312-win_amd64.whl", reqs[1:2], fixed = TRUE)))
  expect_true(any(grepl("^llama-index-core", reqs)))
  expect_false(any(duplicated(reqs)))

  cpu <- python_requirements(sysname = "Darwin", machine = "arm64")
  expect_equal(cpu[1:2], c("torch==2.14.0", "torchvision==0.29.0"))
  expect_false(any(grepl("llama-index", cpu)))

  expect_error(python_requirements(extras = "nope"), "Unknown extras")
})

test_that("setup_cache() points every cache at one folder", {
  skip_if(reticulate::py_available(initialize = FALSE), "Python already started")
  vars <- c("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON_PREFERENCE",
            "HF_HOME", "R_USER_CACHE_DIR", "UV_OFFLINE", "HF_HUB_OFFLINE")
  withr::local_envvar(setNames(rep(NA_character_, length(vars)), vars))
  dir <- withr::local_tempdir()

  set <- setup_cache(dir, offline = TRUE)
  root <- normalizePath(dir, winslash = "/")
  expect_equal(Sys.getenv("UV_CACHE_DIR"), file.path(root, "uv"))
  expect_equal(Sys.getenv("HF_HOME"), file.path(root, "huggingface"))
  expect_equal(Sys.getenv("R_USER_CACHE_DIR"), file.path(root, "R"))
  expect_equal(Sys.getenv("HF_HUB_OFFLINE"), "1")
  expect_equal(unname(set["UV_OFFLINE"]), "1")

  expect_error(setup_cache(c("a", "b")), "single folder")
})

test_that("a TRANSFOREMOTION_PYTHON that does not exist is reported clearly", {
  withr::local_envvar(TRANSFOREMOTION_PYTHON = file.path(tempdir(), "no-such-python"))
  expect_error(transforEmotion:::.te_require("core"), "does not exist")
})

test_that("C compilers can find Python.h once Python starts (Triton on CUDA)", {
  skip_on_cran()
  skip_if(!nzchar(Sys.which("gcc")), "gcc not available")
  skip_if_not(reticulate::py_module_available("torch"))
  reticulate::py_run_string(local = FALSE, paste(
    "import os, subprocess, tempfile",
    "_te_src = os.path.join(tempfile.mkdtemp(), 't.c')",
    "open(_te_src, 'w').write('#include <Python.h>\\nint f(void){return 0;}\\n')",
    "_te_rc = subprocess.run(['gcc', '-c', '-fPIC', _te_src, '-o', _te_src + '.o'], capture_output=True).returncode",
    sep = "\n"
  ))
  expect_equal(reticulate::py$`_te_rc`, 0L)
})
