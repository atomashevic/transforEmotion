# .te_torch_device() without a GPU: .te_cuda_probe() is replaced with what
# PyTorch reports. Arch lists of the PyTorch builds, from libtorch_cuda.so:
# 2.14.0+cu126 and 2.11.0+cu130.
arch_cu126 <- c("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90")
arch_cu130 <- c("sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120")

fake_probe <- function(arch, major, minor, name = "NVIDIA GeForce RTX 5070 Ti",
                       version = "2.14.0+cu126") {
  force(arch)
  function(...) list(
    available = TRUE, version = version, arch = arch,
    devices = data.frame(index = seq_along(major) - 1L, name = name,
                         major = as.integer(major), minor = as.integer(minor),
                         stringsAsFactors = FALSE)
  )
}

# A fresh session state for each test
local_cuda_state <- function(env = parent.frame()) {
  rm(list = ls(.te_cuda_state, all.names = TRUE), envir = .te_cuda_state)
  withr::defer(rm(list = ls(.te_cuda_state, all.names = TRUE), envir = .te_cuda_state),
               envir = env)
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA, TE_HOME = NA, .local_envir = env)
}

test_that(".te_cuda_arch_supports follows the CUDA compatibility rules", {
  # Blackwell GPUs and the cu126 build: no cubin and no PTX
  expect_false(.te_cuda_arch_supports(arch_cu126, 12, 0))
  expect_false(.te_cuda_arch_supports(arch_cu126, 10, 0))
  # A cubin runs on later minor versions of the same major version
  expect_true(.te_cuda_arch_supports(arch_cu126, 8, 9))
  expect_true(.te_cuda_arch_supports(arch_cu126, 9, 0))
  expect_true(.te_cuda_arch_supports(arch_cu126, 6, 1))
  # The cu130 build runs on Blackwell and not below 7.5
  expect_true(.te_cuda_arch_supports(arch_cu130, 12, 0))
  expect_true(.te_cuda_arch_supports(arch_cu130, 12, 1))
  expect_true(.te_cuda_arch_supports(arch_cu130, 10, 0))
  expect_false(.te_cuda_arch_supports(arch_cu130, 7, 0))
  expect_false(.te_cuda_arch_supports(arch_cu130, 6, 1))
  # A cubin does not run on an earlier minor version
  expect_false(.te_cuda_arch_supports("sm_86", 8, 0))
  # PTX is compiled for any later GPU, also of a later major version
  expect_true(.te_cuda_arch_supports(c("sm_80", "compute_90"), 12, 0))
  expect_false(.te_cuda_arch_supports("compute_90", 8, 9))
  # Architecture-specific targets run only on that capability
  expect_true(.te_cuda_arch_supports("sm_90a", 9, 0))
  expect_false(.te_cuda_arch_supports(c("sm_90a", "compute_90a"), 10, 0))
  # Family-specific targets run on the same major version
  expect_true(.te_cuda_arch_supports("sm_100f", 10, 3))
  expect_false(.te_cuda_arch_supports("sm_100f", 12, 0))
  expect_false(.te_cuda_arch_supports(character(), 8, 6))
})

test_that("device = 'auto' uses the GPU when the build has kernels for it", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(arch_cu126, 8, 9, "NVIDIA GeForce RTX 4090"))
  expect_silent(expect_identical(.te_torch_device("auto"), "cuda"))
  expect_identical(.te_torch_device("cuda"), "cuda")
  expect_true(.te_cuda_supported())
})

test_that("device = 'auto' uses the CPU on a Blackwell GPU with the cu126 build, with one message", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(arch_cu126, 12, 0))
  msg <- expect_message(device <- .te_torch_device("auto"), "has no kernels")
  expect_identical(device, "cpu")
  text <- conditionMessage(msg)
  expect_match(text, "NVIDIA GeForce RTX 5070 Ti", fixed = TRUE)
  expect_match(text, "compute capability 12.0", fixed = TRUE)
  expect_match(text, "PyTorch 2.14.0+cu126", fixed = TRUE)
  expect_match(text, "sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90", fixed = TRUE)
  expect_match(text, "cu130", fixed = TRUE)
  expect_match(text, "R580", fixed = TRUE)
  expect_match(text, "uses the CPU", fixed = TRUE)
  # Once per session
  expect_silent(expect_identical(.te_torch_device("auto"), "cpu"))
  expect_false(.te_cuda_supported())
})

test_that("an explicit device = 'cuda' stops with the explanation", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(arch_cu126, 12, 0))
  err <- expect_error(.te_torch_device("cuda"), "has no kernels")
  expect_match(conditionMessage(err), "compute capability 12.0", fixed = TRUE)
  expect_match(conditionMessage(err), 'use device = "cpu"', fixed = TRUE)
  expect_error(.te_torch_device("cuda:0"), "GPU 0")
  expect_identical(.te_torch_device("cpu"), "cpu")
})

test_that("the fix names the source of the Python environment", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(arch_cu126, 12, 0))

  # Fixed environment: pip install from the cu130 index, same torch version
  withr::local_envvar(TRANSFOREMOTION_PYTHON = "/opt/env/bin/python")
  err <- expect_error(.te_torch_device("cuda"))
  expect_match(conditionMessage(err),
               "pip install torch==2.14.0 torchvision --index-url https://download.pytorch.org/whl/cu130",
               fixed = TRUE)

  # Docker image built from docker/Dockerfile
  te_home <- withr::local_tempdir()
  writeLines("gpu=1", file.path(te_home, "BUILD_INFO"))
  withr::local_envvar(TE_HOME = te_home)
  err <- expect_error(.te_torch_device("cuda"))
  expect_match(conditionMessage(err), "--build-arg GPU=1 --build-arg CUDA=cu130", fixed = TRUE)

  # Environment that transforEmotion manages
  withr::local_envvar(TRANSFOREMOTION_PYTHON = NA, TE_HOME = NA)
  err <- expect_error(.te_torch_device("cuda"))
  expect_match(conditionMessage(err), "nvidia-smi", fixed = TRUE)
  expect_match(conditionMessage(err), "setup_cache(gpu = TRUE)", fixed = TRUE)
})

test_that("a GPU below compute capability 7.5 with the cu130 build is pointed at cu126", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(arch_cu130, 7, 0, "Tesla V100-SXM2-32GB",
                                                    version = "2.14.0+cu130"))
  msg <- expect_message(expect_identical(.te_torch_device("auto"), "cpu"))
  expect_match(conditionMessage(msg), "CUDA 12.6 build of PyTorch (cu126)", fixed = TRUE)
  expect_match(conditionMessage(msg), "R525", fixed = TRUE)
  expect_match(conditionMessage(msg), "CUDA_VISIBLE_DEVICES", fixed = TRUE)
})

test_that("only GPU 0 decides device = 'auto'; an explicit cuda:N checks GPU N", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(arch_cu130, c(12, 7), c(0, 0),
                                                    c("NVIDIA GeForce RTX 5090", "Tesla V100"),
                                                    version = "2.14.0+cu130"))
  expect_identical(.te_torch_device("auto"), "cuda")
  expect_identical(.te_torch_device("cuda:0"), "cuda:0")
  expect_error(.te_torch_device("cuda:1"), "Tesla V100", fixed = TRUE)
})

test_that("without CUDA or without an arch list, the old choice holds", {
  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = function(...) list(available = FALSE, version = "2.14.0+cpu"))
  expect_identical(.te_torch_device("auto"), "cpu")
  expect_true(is.na(.te_cuda_supported()))

  local_cuda_state()
  local_mocked_bindings(.te_cuda_probe = fake_probe(character(), 12, 0))
  expect_identical(.te_torch_device("auto"), "cuda")
})

test_that("the probe is cached for the session", {
  local_cuda_state()
  calls <- 0
  probe <- fake_probe(arch_cu126, 8, 6)
  local_mocked_bindings(.te_cuda_probe = function(...) {
    calls <<- calls + 1
    probe()
  })
  .te_torch_device("auto")
  .te_torch_device("auto")
  .te_torch_device("cuda")
  expect_identical(calls, 1)
})

test_that(".te_cuda_probe reads the arch flags without torch.cuda.get_arch_list()", {
  torch <- list(
    `__version__` = "2.14.0+cu126",
    cuda = list(
      is_available = function() TRUE,
      device_count = function() 1L,
      get_device_capability = function(i) list(12L, 0L),
      get_device_name = function(i) "NVIDIA GeForce RTX 5070 Ti",
      get_arch_list = function() stop("not used")
    )
  )
  local_mocked_bindings(
    py_get_attr = function(x, name, ...) {
      if (identical(name, "_C")) return(list())
      function() paste(arch_cu126, collapse = " ")
    },
    .package = "reticulate"
  )
  probe <- .te_cuda_probe(torch)
  expect_true(probe$available)
  expect_identical(probe$arch, arch_cu126)
  expect_identical(probe$devices$major, 12L)
  expect_identical(probe$devices$name, "NVIDIA GeForce RTX 5070 Ti")
})

test_that("rag's auto_device uses the CPU on a GPU without kernels", {
  local_cuda_state()
  local_mocked_bindings(
    ensure_te_py_env = function(...) invisible(TRUE),
    get_cpu_memory = function() 64000,
    .te_cuda_probe = fake_probe(arch_cu126, 12, 0)
  )
  local_mocked_bindings(import = function(module, ...) list(), .package = "reticulate")
  expect_message(device <- auto_device("auto", "tinyllama", verbose = FALSE), "has no kernels")
  expect_identical(device, "cpu")
  expect_error(auto_device("cuda", "tinyllama", verbose = FALSE), "has no kernels")
})

test_that("rag's auto_device keeps to GPU 0 when another GPU has no kernels", {
  local_cuda_state()
  local_mocked_bindings(
    ensure_te_py_env = function(...) invisible(TRUE),
    get_gpu_memory = function(torch, device_no) 24000,
    .te_cuda_probe = fake_probe(arch_cu130, c(12, 7), c(0, 0), version = "2.14.0+cu130")
  )
  torch <- list(cuda = list(device_count = function() stop("not used")))
  local_mocked_bindings(import = function(module, ...) torch, .package = "reticulate")
  expect_identical(auto_device("auto", "tinyllama", verbose = FALSE), "cuda:0")
})
