# setup_modules() with a fixed Python environment (TRANSFOREMOTION_PYTHON)
# compares its packages with the requirements, and a function that needs an
# extra feature set checks that its modules are there. No test runs Python
# except the last one: reticulate::import() returns fake importlib.metadata,
# importlib.util and packaging.requirements modules.

requirement_name <- function(req) sub("^([A-Za-z0-9._-]+).*$", "\\1", req)

# Installed versions that meet every requirement of the given feature sets
complete_environment <- function(extras = character(), torch = "2.14.0+cpu") {
  reqs <- python_requirements(extras, gpu = grepl("+cu", torch, fixed = TRUE))
  installed <- setNames(as.list(rep("1.0", length(reqs))), requirement_name(reqs))
  installed$torch <- torch
  installed
}

# reticulate::import() for a Python with the installed versions and the
# requirements of installed packages (`requires`). The specifier of a
# requirement in `rejects` does not contain the installed version; that of a
# requirement in `invalid` raises InvalidVersion, as packaging before 25 does
# for a version outside PEP 440. find_spec() finds the import names in
# `modules` (all when NULL), and adds each name it is called with to `specs`
# in the caller.
fake_import <- function(installed, rejects = character(), packaging = TRUE,
                        requires = list(), invalid = character(),
                        modules = NULL, env = parent.frame()) {
  force(env)
  metadata <- list(
    version = function(name) {
      if (is.null(installed[[name]])) stop("PackageNotFoundError: ", name)
      installed[[name]]
    },
    requires = function(name) requires[[name]]
  )
  requirement <- function(req) list(
    name = requirement_name(req),
    url = if (grepl(" @ ", req, fixed = TRUE)) sub("^.* @ ", "", req),
    specifier = list(contains = function(version, prereleases = FALSE) {
      if (req %in% invalid) stop("InvalidVersion: Invalid version: '", version, "'")
      !req %in% rejects
    })
  )
  # As in Python, find_spec() raises when a parent package is missing
  find_spec <- function(name) {
    env$specs <- c(env$specs, name)
    if (is.null(modules) || name %in% modules) return("spec")
    parent <- sub("\\.[^.]*$", "", name)
    if (parent != name && !any(modules == parent | startsWith(modules, paste0(parent, ".")))) {
      stop("ModuleNotFoundError: No module named '", parent, "'")
    }
    NULL
  }
  function(module, ...) {
    switch(module,
      "importlib.metadata" = metadata,
      "importlib.util" = list(find_spec = find_spec),
      "packaging.requirements" = if (packaging) {
        list(Requirement = requirement)
      } else {
        stop("ModuleNotFoundError: No module named 'packaging'")
      },
      "torch" = list(`__version__` = installed$torch,
                     cuda = list(is_available = function() FALSE)),
      stop("unexpected import: ", module)
    )
  }
}

# A fixed environment that .te_require() has already selected
local_fixed_python <- function(python = "~/te-venv/bin/python", env = parent.frame()) {
  withr::local_envvar(TRANSFOREMOTION_PYTHON = python, HF_HUB_OFFLINE = "1",
                      .local_envir = env)
  state <- transforEmotion:::.te_py_state
  old <- list(fixed_python = state$fixed_python, fixed_features = state$fixed_features)
  withr::defer({
    state$fixed_python <- old$fixed_python
    state$fixed_features <- old$fixed_features
  }, envir = env)
  state$fixed_python <- TRUE
  state$fixed_features <- character()
}

test_that("setup_modules() warns about packages a fixed environment lacks or has too old", {
  local_fixed_python()
  installed <- complete_environment("audio")
  installed$protobuf <- NULL
  installed$av <- NULL
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(
    import = fake_import(installed, rejects = "transformers>=4.40,<4.47"),
    .package = "reticulate"
  )
  expect_message(
    w <- expect_warning(setup_modules(extras = "audio", download_models = FALSE),
                        "does not meet these requirements"),
    "Python environment ready"
  )
  msg <- conditionMessage(w)
  expect_match(msg, "  protobuf>=3.20,<8: not installed", fixed = TRUE)
  expect_match(msg, "  av>=14,<20: not installed", fixed = TRUE)
  expect_match(msg, "  transformers>=4.40,<4.47: transformers 1.0 is installed", fixed = TRUE)
  expect_no_match(msg, "numpy", fixed = TRUE)
  # The command works in a shell: "~" is expanded
  python <- normalizePath("~/te-venv/bin/python", winslash = "/", mustWork = FALSE)
  expect_match(msg, paste0('uv pip install --python "', python, '" "transformers>=4.40,<4.47" ',
                           '"protobuf>=3.20,<8" "av>=14,<20"'), fixed = TRUE)
  expect_match(msg, 'python_requirements(extras = "audio")', fixed = TRUE)
})

test_that("printed commands keep the link to the Python of a virtual environment", {
  skip_on_os("windows")
  dir <- withr::local_tempdir()
  base <- file.path(dir, "python3.12")
  file.create(base)
  bin <- file.path(dir, "venv", "bin")
  dir.create(bin, recursive = TRUE)
  link <- file.path(bin, "python")
  skip_if_not(file.symlink(base, link))
  # uv pip install --python with the resolved path would install into the
  # base interpreter
  python <- file.path(normalizePath(bin, winslash = "/"), "python")
  expect_equal(transforEmotion:::.te_shell_python(link), python)
  expect_false(identical(python, normalizePath(link, winslash = "/")))
  expect_false(startsWith(transforEmotion:::.te_shell_python("~/venv/bin/python"), "~"))

  withr::local_envvar(TRANSFOREMOTION_PYTHON = link)
  err <- "ImportError: \n requires the protobuf library but it was not found in your environment."
  expect_error(transforEmotion:::.te_stop_if_no_protobuf(err),
               paste0('--python "', python, '"'), fixed = TRUE)
  installed <- complete_environment()
  installed$protobuf <- NULL
  local_mocked_bindings(import = fake_import(installed), .package = "reticulate")
  expect_warning(transforEmotion:::.te_warn_unmet_requirements(),
                 paste0('--python "', python, '"'), fixed = TRUE)
})

test_that("a complete fixed environment gives no warning", {
  local_fixed_python()
  extras <- c("rag", "youtube", "audio", "findingemo", "gpu")
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(import = fake_import(complete_environment(extras)), .package = "reticulate")
  expect_no_warning(expect_message(setup_modules(extras = extras, download_models = FALSE),
                                   "Python environment ready"))
})

test_that("any OpenCV build provides cv2, and a URL requirement needs only the package", {
  installed <- complete_environment()
  installed[["opencv-python-headless"]] <- NULL
  installed[["opencv-python"]] <- "4.10.0"
  torch <- grep("^torch\\b", python_requirements(), value = TRUE)[1]
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(import = fake_import(installed, rejects = torch), .package = "reticulate")
  # On Linux x86_64 and Windows the PyTorch requirements are wheel URLs
  if (grepl(" @ ", torch, fixed = TRUE)) {
    expect_length(transforEmotion:::.te_unmet_requirements()$unmet, 0)
  }

  # The version of the build found is checked
  installed[["opencv-python"]] <- "5.0.0"
  local_mocked_bindings(
    import = fake_import(installed, rejects = "opencv-python-headless<5"),
    .package = "reticulate"
  )
  unmet <- transforEmotion:::.te_unmet_requirements()$unmet
  expect_equal(unname(unmet["opencv-python-headless<5"]), "opencv-python 5.0.0")
})

test_that("a CUDA PyTorch gives gpu = TRUE in the rebuild advice", {
  local_fixed_python()
  installed <- complete_environment(torch = "2.14.0+cu126")
  installed$protobuf <- NULL
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(import = fake_import(installed), .package = "reticulate")
  w <- expect_warning(transforEmotion:::.te_warn_unmet_requirements(), "protobuf")
  expect_match(conditionMessage(w), "python_requirements(gpu = TRUE)", fixed = TRUE)
})

test_that("the CUDA build of PyTorch from PyPI gives gpu = TRUE", {
  # PyPI's Linux x86_64 torch has a plain version and requires the CUDA
  # libraries as nvidia-* packages
  local_fixed_python()
  installed <- complete_environment(torch = "2.11.0")
  installed$protobuf <- NULL
  requires <- list(torch = list(
    "filelock",
    'nvidia-cuda-nvrtc-cu13==13.0.48; platform_system == "Linux" and platform_machine == "x86_64"'
  ))
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(import = fake_import(installed, requires = requires),
                        .package = "reticulate")
  expect_true(transforEmotion:::.te_unmet_requirements()$gpu)
  w <- expect_warning(transforEmotion:::.te_warn_unmet_requirements(), "protobuf")
  expect_match(conditionMessage(w), "python_requirements(gpu = TRUE)", fixed = TRUE)

  # The CPU build from the PyTorch index, and torch on macOS or Windows
  for (torch in c("2.14.0+cpu", "2.14.0")) {
    installed$torch <- torch
    local_mocked_bindings(
      import = fake_import(installed, requires = list(torch = list("filelock", "sympy"))),
      .package = "reticulate"
    )
    expect_false(transforEmotion:::.te_unmet_requirements()$gpu, label = torch)
  }
})

test_that("a version that packaging cannot parse is reported, not raised", {
  installed <- complete_environment()
  installed$sentencepiece <- "abc"
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(import = fake_import(installed, invalid = "sentencepiece<1"),
                        .package = "reticulate")
  unmet <- transforEmotion:::.te_unmet_requirements()$unmet
  expect_equal(unmet, c("sentencepiece<1" = "sentencepiece abc"))
})

test_that("without PyTorch, the GPU is detected once for the PyTorch requirements", {
  installed <- complete_environment(c("rag", "audio"))
  installed$torch <- NULL
  installed$torchvision <- NULL
  calls <- 0
  local_mocked_bindings(te_should_use_gpu = function() {
    calls <<- calls + 1
    TRUE
  })
  local_mocked_bindings(import = fake_import(installed), .package = "reticulate")
  check <- transforEmotion:::.te_unmet_requirements(c("rag", "audio"))
  expect_equal(calls, 1)
  expect_true(check$gpu)
  expect_equal(unname(requirement_name(names(check$unmet))), c("torch", "torchvision"))
  expect_equal(names(check$unmet), python_requirements(gpu = TRUE)[1:2])
})

test_that("without packaging, the check falls back to presence", {
  installed <- complete_environment()
  installed$protobuf <- NULL
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(
    import = fake_import(installed, rejects = "transformers>=4.40,<4.47", packaging = FALSE),
    .package = "reticulate"
  )
  unmet <- transforEmotion:::.te_unmet_requirements()$unmet
  expect_equal(names(unmet), "protobuf>=3.20,<8")
  expect_equal(unname(unmet), "")
})

test_that("a managed environment is not checked", {
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA, HF_HUB_OFFLINE = "1")
  local_mocked_bindings(
    .te_require = function(...) invisible(TRUE),
    .te_unmet_requirements = function(...) stop("must not run")
  )
  local_mocked_bindings(import = fake_import(complete_environment()), .package = "reticulate")
  expect_message(setup_modules(download_models = FALSE), "Python environment ready")
})

# The import names of the given feature sets
feature_modules <- function(features) {
  unique(unlist(transforEmotion:::.te_feature_modules[c("core", features)], use.names = FALSE))
}

test_that("a function that needs an extra set stops when a fixed environment lacks it", {
  local_fixed_python()
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(
    import = fake_import(complete_environment(), modules = feature_modules(character())),
    .package = "reticulate"
  )
  err <- expect_error(rag("a text"), "does not have these modules")
  msg <- conditionMessage(err)
  expect_match(msg, paste0("  rag: llama_index.core, llama_index.llms.huggingface, ",
                           "llama_index.embeddings.huggingface, pypdf, rank_bm25\n"), fixed = TRUE)
  expect_match(msg, "transforEmotion does not install packages into a fixed environment",
               fixed = TRUE)
  python <- normalizePath("~/te-venv/bin/python", winslash = "/", mustWork = FALSE)
  expect_match(msg, paste0('uv pip install --python "', python, '" "llama-index-core>=0.10.30,<0.11" ',
                           '"llama-index-llms-huggingface" "llama-index-embeddings-huggingface" ',
                           '"pypdf" "rank-bm25"\n'), fixed = TRUE)
  expect_match(msg, 'python_requirements(extras = "rag")', fixed = TRUE)
  expect_match(msg, "--build-arg EXTRAS=rag,", fixed = TRUE)
  expect_no_match(msg, "GPU=1", fixed = TRUE)
  # A failed set is checked again on the next call
  expect_length(transforEmotion:::.te_py_state$fixed_features, 0)

  expect_error(download_findingemo_data(withr::local_tempdir()),
               "  findingemo: findingemo_light, termcolor\n", fixed = TRUE)
})

test_that("each extra set of a fixed environment is checked once, by find_spec()", {
  local_fixed_python()
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(
    import = fake_import(complete_environment(c("youtube", "gpu")),
                         modules = feature_modules(c("youtube", "gpu"))),
    .package = "reticulate"
  )
  specs <- character()
  expect_no_error(transforEmotion:::.te_require("youtube"))
  expect_equal(specs, "pytubefix")
  expect_equal(transforEmotion:::.te_py_state$fixed_features, "youtube")
  expect_no_error(transforEmotion:::.te_require(c("core", "youtube")))
  expect_equal(specs, "pytubefix")

  # Only the requested sets that were not found before are checked, and the
  # error names only those that are missing
  specs <- character()
  err <- expect_error(transforEmotion:::.te_require(c("youtube", "gpu", "audio", "findingemo")),
                      "does not have these modules")
  expect_equal(specs, c("bitsandbytes", "av", "findingemo_light", "termcolor"))
  msg <- conditionMessage(err)
  expect_match(msg, "\n  audio: av\n  findingemo: findingemo_light, termcolor\n", fixed = TRUE)
  expect_no_match(msg, "youtube", fixed = TRUE)
  expect_match(msg, 'python_requirements(extras = c("audio", "findingemo"))', fixed = TRUE)
  expect_match(msg, "--build-arg EXTRAS=audio,findingemo,", fixed = TRUE)
  expect_equal(transforEmotion:::.te_py_state$fixed_features, c("youtube", "gpu"))
})

test_that("the core set of a fixed environment does not start Python", {
  local_fixed_python()
  local_mocked_bindings(import = function(...) stop("Python was started"), .package = "reticulate")
  expect_no_error(transforEmotion:::.te_require("core"))
  expect_no_error(transforEmotion:::.te_require(character()))
})

test_that("a missing gpu set names the CUDA build, and a broken package its requirement", {
  local_fixed_python()
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  installed <- complete_environment(torch = "2.14.0+cu126")
  local_mocked_bindings(import = fake_import(installed, modules = feature_modules(character())),
                        .package = "reticulate")
  err <- expect_error(transforEmotion:::.te_require("gpu"), "  gpu: bitsandbytes\n", fixed = TRUE)
  expect_match(conditionMessage(err), 'python_requirements(extras = "gpu", gpu = TRUE)', fixed = TRUE)
  expect_match(conditionMessage(err), "--build-arg GPU=1 adds the gpu set", fixed = TRUE)

  # PyAV is installed, but its module does not import
  local_mocked_bindings(
    import = fake_import(complete_environment("audio"), modules = feature_modules(character())),
    .package = "reticulate"
  )
  err <- expect_error(transforEmotion:::.te_require("audio"), "  audio: av\n", fixed = TRUE)
  expect_match(conditionMessage(err), '" "av>=14,<20"\n', fixed = TRUE)
})

test_that("setup_modules() reports a missing extra set of a fixed environment", {
  # The stop of .te_require() would hide the other unmet requirements
  local_fixed_python()
  installed <- complete_environment()
  installed$protobuf <- NULL
  local_mocked_bindings(te_should_use_gpu = function() stop("torch is installed"))
  local_mocked_bindings(
    import = fake_import(installed, modules = feature_modules(character())),
    .package = "reticulate"
  )
  specs <- character()
  expect_message(
    w <- expect_warning(setup_modules(extras = "rag", download_models = FALSE),
                        "does not meet these requirements"),
    "Python environment ready"
  )
  expect_match(conditionMessage(w), "  protobuf>=3.20,<8: not installed", fixed = TRUE)
  expect_match(conditionMessage(w), "  rank-bm25: not installed", fixed = TRUE)
  expect_length(specs, 0)
})

test_that("the note about ~/.virtualenvs/r-reticulate is not shown for a fixed environment", {
  venv <- withr::local_tempdir()
  local_mocked_bindings(.te_default_venv = function() venv)
  # .onAttach() unsets RETICULATE_PYTHON and can set RETICULATE_AUTOCONFIGURE
  withr::local_envvar(RETICULATE_PYTHON = NA, RETICULATE_AUTOCONFIGURE = "FALSE")
  startup_messages <- function() {
    messages <- character()
    withCallingHandlers(
      transforEmotion:::.onAttach(dirname(venv), "transforEmotion"),
      packageStartupMessage = function(m) {
        messages <<- c(messages, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    )
    messages
  }
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA)
  expect_true(any(grepl("~/.virtualenvs/r-reticulate", startup_messages(), fixed = TRUE)))
  # The Docker image sets TRANSFOREMOTION_PYTHON
  withr::local_envvar(TRANSFOREMOTION_PYTHON = "/opt/transforEmotion/venv/bin/python")
  messages <- startup_messages()
  expect_true(any(grepl("transforEmotion (version", messages, fixed = TRUE)))
  expect_false(any(grepl("r-reticulate", messages, fixed = TRUE)))
})

test_that("the check reads the versions of a real fixed environment", {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON")), "TRANSFOREMOTION_PYTHON is not set")
  transforEmotion:::.te_require("core")
  imported <- "sorted(__import__('sys').modules)"
  before <- reticulate::py_eval(imported)
  check <- transforEmotion:::.te_unmet_requirements()
  expect_type(check$unmet, "character")
  expect_type(check$gpu, "logical")
  # Reading metadata imports none of the required packages
  new <- setdiff(reticulate::py_eval(imported), before)
  expect_false(any(c("transformers", "sentence_transformers", "timm", "cv2") %in% new))
})
