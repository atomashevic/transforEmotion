# transforEmotion with a fixed Python environment and pre-downloaded models,
# for offline use (HPC clusters, air-gapped machines, Apptainer).
# .github/workflows/docker.yml builds it and publishes it to
# ghcr.io/atomashevic/transforemotion: the CPU image, and the CUDA image
# (GPU=1) with -cuda tags. An image does not change after its build: after a
# fix in transforEmotion or in its Python requirements, pull or build it again.
# The images are for x86_64 (linux/amd64).
#
# Build from the repository root, with Docker 23 or newer (BuildKit is needed
# for RUN --mount and RUN --network):
#   docker build -f docker/Dockerfile -t transforemotion .
#   docker build -f docker/Dockerfile --build-arg EXTRAS=rag -t transforemotion:rag .
#   docker build -f docker/Dockerfile --build-arg GPU=1 -t transforemotion:cuda .
#   docker build -f docker/Dockerfile --build-arg REVISION="$(git rev-parse HEAD)" -t transforemotion .
#
# Build arguments:
#   EXTRAS         comma-separated feature sets: rag, youtube, audio, findingemo, gpu
#                  ("rag" also downloads the rag() models, about 2.3 GB, and
#                  "audio" the speech_scores() model, about 800 MB)
#   MODELS         comma-separated extra Hugging Face model IDs to include
#   GPU            1 for the CUDA 12.6 build of PyTorch (run with --gpus all)
#   SNAPSHOT_DATE  YYYY-MM-DD: install the R packages from the Posit Package
#                  Manager snapshot of that date and resolve the Python
#                  packages with uv --exclude-newer that date. Empty: latest.
#   REVISION       git commit of the build, for the image label and BUILD_INFO
#   VERSION        version for the image label. docker.yml replaces this label
#                  with the first tag of the build (main, X.Y.Z or sha-<7>;
#                  with -cuda in the CUDA image). BUILD_INFO always has the
#                  installed package version
#
# At run time the image needs no network and no writable folder except TMPDIR
# (default /tmp). It runs as any user and from a read-only file system
# (":exec" lets PyTorch on CUDA load the kernels it compiles into /tmp):
#   docker run --rm -it --read-only --tmpfs /tmp:exec --user "$(id -u):$(id -g)" \
#     -v "$PWD":/work transforemotion
#   apptainer build transforemotion.sif docker-daemon://transforemotion:latest
#
# The image lists what it contains in /opt/transforEmotion: BUILD_INFO,
# requirements.in, requirements.lock, python-packages.txt, R-packages.txt
# and models.txt.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM rocker/r-ver:4.6.1@sha256:bafb3b9a72791eeeb810f0bc0472e74c7f4939b20b46c96cb7ebc6838b6a25ef

ARG SNAPSHOT_DATE=""

COPY --from=uv /uv /uvx /usr/local/bin/

ENV TE_HOME=/opt/transforEmotion
ENV HF_HOME=${TE_HOME}/huggingface \
    NLTK_DATA=${TE_HOME}/nltk_data \
    TRANSFOREMOTION_SFACE_MODEL=${TE_HOME}/face_recognition_sface_2021dec_int8.onnx \
    UV_PYTHON_INSTALL_DIR=${TE_HOME}/python \
    UV_PYTHON_PREFERENCE=only-managed

# R dependencies: binaries from Posit Package Manager, with the system
# libraries they need installed by pak. Only DESCRIPTION is mounted, so this
# layer is rebuilt only when DESCRIPTION changes. The pak package cache, the
# temporary folder of pak's worker process and the apt lists are removed in
# the same step, so they do not stay in the layer.
RUN --mount=type=bind,source=DESCRIPTION,target=/tmp/te-deps/DESCRIPTION \
    repo="https://p3m.dev/cran/__linux__/noble/${SNAPSHOT_DATE:-latest}" \
 && R -q -e "options(repos = c(CRAN = '${repo}')); install.packages('pak')" \
 && PKG_SYSREQS=true R -q -e "options(repos = c(CRAN = '${repo}')); \
      pak::local_install_deps('/tmp/te-deps', upgrade = FALSE, ask = FALSE)" \
 && rm -rf /root/.cache/R /tmp/Rtmp* /var/lib/apt/lists/*

# The package. A bind mount instead of COPY keeps the source tree out of the
# image layers.
RUN --mount=type=bind,source=.,target=/tmp/transforEmotion,rw \
    repo="https://p3m.dev/cran/__linux__/noble/${SNAPSHOT_DATE:-latest}" \
 && R -q -e "options(repos = c(CRAN = '${repo}')); \
      pak::local_install('/tmp/transforEmotion', upgrade = FALSE, ask = FALSE)" \
 && rm -rf /root/.cache/R /tmp/Rtmp* \
 && mkdir -p ${TE_HOME} \
 && Rscript -e 'ip <- installed.packages(lib.loc = .libPaths()[1:2]); \
      write.table(ip[order(ip[, "Package"]), c("Package", "Version", "LibPath")], \
                  file.path(Sys.getenv("TE_HOME"), "R-packages.txt"), \
                  quote = FALSE, sep = "\t", row.names = FALSE)'

# A build argument changes the cache key of every later RUN, so each one is
# declared just before the first step that reads it
ARG EXTRAS=""
ARG GPU=0

# Fixed Python environment built from the package's own requirements. uv
# cannot run from a read-only cache, so the image uses a plain virtual
# environment that transforEmotion loads through TRANSFOREMOTION_PYTHON.
# python_requirements() gives the requirements with their caps
# (requirements.in); uv resolves them to exact versions (requirements.lock),
# and the environment is installed from the lock. Both files stay in the
# image. Bytecode is compiled at build time, for the packages and for the
# standard library of the uv-managed Python: a read-only image cannot write
# __pycache__, and without it every session compiles the imported modules
# again. torch/test holds C++ test programs that Python never loads.
RUN Rscript -e ' \
      extras <- trimws(strsplit(Sys.getenv("EXTRAS"), ",")[[1]]); \
      gpu <- Sys.getenv("GPU") == "1"; \
      if (gpu) extras <- c(extras, "gpu"); \
      writeLines(transforEmotion::python_requirements(extras = extras, gpu = gpu), \
                 file.path(Sys.getenv("TE_HOME"), "requirements.in"))' \
 && uv venv --no-cache --python 3.12 ${TE_HOME}/venv \
 && uv pip compile --no-cache --no-header --only-binary :all: \
      ${SNAPSHOT_DATE:+--exclude-newer "${SNAPSHOT_DATE}"} \
      --python ${TE_HOME}/venv/bin/python \
      ${TE_HOME}/requirements.in -o ${TE_HOME}/requirements.lock \
 && UV_LINK_MODE=copy uv pip sync --no-cache --no-build --compile-bytecode \
      --python ${TE_HOME}/venv/bin/python ${TE_HOME}/requirements.lock \
 && ${TE_HOME}/venv/bin/python -m compileall -q -j 0 ${TE_HOME}/python \
 && rm -rf ${TE_HOME}/venv/lib/python3.12/site-packages/torch/test \
 && uv pip freeze --python ${TE_HOME}/venv/bin/python > ${TE_HOME}/python-packages.txt

# Point transforEmotion and reticulate at the environment once it exists.
# Apptainer passes RETICULATE_PYTHON and RETICULATE_PYTHON_ENV in from the
# host unless the image sets them. reticulate prefers either one to the
# Python that transforEmotion selects, and fails when it names a Python or a
# folder that does not exist.
ENV TRANSFOREMOTION_PYTHON=${TE_HOME}/venv/bin/python \
    RETICULATE_PYTHON=${TE_HOME}/venv/bin/python \
    RETICULATE_PYTHON_ENV=${TE_HOME}/venv

# Every Python module of the requested feature sets must import through R and
# reticulate, as transforEmotion imports them; a missing package fails the
# build. A GPU build must also have the CUDA build of PyTorch.
RUN Rscript -e ' \
      extras <- trimws(strsplit(Sys.getenv("EXTRAS"), ",")[[1]]); \
      gpu <- Sys.getenv("GPU") == "1"; \
      sets <- unique(c("core", extras, if (gpu) "gpu")); \
      modules <- unique(unlist(transforEmotion:::.te_feature_modules[sets], use.names = FALSE)); \
      transforEmotion:::.te_require("core"); \
      failed <- unlist(lapply(modules, function(m) tryCatch( \
        { reticulate::import(m); NULL }, \
        error = function(e) paste0(m, ": ", conditionMessage(e))))); \
      if (gpu && is.null(reticulate::import("torch")$version$cuda)) \
        failed <- c(failed, "torch: the CPU build is installed, but GPU=1"); \
      if (length(failed)) { \
        message("Python modules that do not import:\n", paste(failed, collapse = "\n")); \
        quit(status = 1) \
      }; \
      message("Python modules for ", paste(sets, collapse = ", "), " import: ", \
              paste(modules, collapse = ", "))'

# Default models, the models for the chosen extras and any MODELS.
# transformers 4.46 writes hub/version.txt into a non-empty cache that does
# not have it, at every import, and warns when it cannot; the file is written
# here so a read-only image does not try. models.txt records the revision of
# each model.
ARG MODELS=""
RUN Rscript -e ' \
      extras <- trimws(strsplit(Sys.getenv("EXTRAS"), ",")[[1]]); \
      models <- trimws(strsplit(Sys.getenv("MODELS"), ",")[[1]]); \
      transforEmotion::setup_modules(extras = setdiff(extras, "gpu"), models = models)' \
 && mkdir -p ${HF_HOME}/hub \
 && echo 1 > ${HF_HOME}/hub/version.txt \
 && ${TRANSFOREMOTION_PYTHON} -c 'from huggingface_hub import scan_cache_dir; \
      [print(r.repo_id, v.commit_hash, r.size_on_disk) \
       for r in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id) for v in r.revisions]' \
      > ${TE_HOME}/models.txt \
 && chmod -R a+rX ${TE_HOME}

# Run offline by default. HF_HUB_CACHE and SENTENCE_TRANSFORMERS_HOME are set
# as well as HF_HOME: Apptainer passes host variables that the image does not
# set into the container, and a host HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE or
# SENTENCE_TRANSFORMERS_HOME would hide the bundled models. A host
# TRANSFORMERS_CACHE or PYTORCH_TRANSFORMERS_CACHE also hides them from
# transformers. The image does not set these two, because transformers warns
# in every session when one is set: unset them on the host, or use
# apptainer exec --cleanenv. To use other models, pass -e HF_HUB_OFFLINE=0 and
# set HF_HUB_CACHE and SENTENCE_TRANSFORMERS_HOME to a mounted, writable
# folder: Apptainer images are read-only, and --rm deletes a download into a
# container.
ENV HF_HUB_OFFLINE=1 \
    HF_HUB_CACHE=${HF_HOME}/hub \
    SENTENCE_TRANSFORMERS_HOME=${HF_HOME}/hub

# The default text, image and sentence models must load without network
RUN --network=none Rscript -e ' \
      library(transforEmotion); \
      s <- transformer_scores("I am happy", c("joy", "anger"), \
                              transformer = "cross-encoder-distilroberta"); \
      i <- image_scores(system.file("extdata", "boris-1.png", package = "transforEmotion"), \
                        c("happy", "sad")); \
      d <- sentence_similarity("I am happy", "I feel joy"); \
      stopifnot(abs(sum(s[[1]]) - 1) < 1e-3, nrow(i) == 1, is.finite(d[1, 1]))' \
 && rm -rf /tmp/torchinductor_* /tmp/Rtmp*

# Labels replace the ones inherited from rocker/r-ver (title, authors,
# vendor, licenses, revision, version, and base.name, which rocker sets to
# ubuntu:noble). Keep base.name the same as the tag in the FROM line; the
# digest is not repeated here, so a digest update changes only the FROM line.
# In CI, docker/metadata-action sets title, description, url, source,
# licenses, version, revision and created again, and its values take
# precedence. ARGs that change with every commit come last, so they do not
# invalidate the layers above.
ARG REVISION=unknown
ARG VERSION=""
LABEL org.opencontainers.image.title="transforEmotion" \
      org.opencontainers.image.description="R package transforEmotion with a fixed Python environment and pre-downloaded models, for offline use" \
      org.opencontainers.image.authors="Aleksandar Tomašević <atomashevic@gmail.com>" \
      org.opencontainers.image.vendor="transforEmotion" \
      org.opencontainers.image.url="https://github.com/atomashevic/transforEmotion" \
      org.opencontainers.image.source="https://github.com/atomashevic/transforEmotion" \
      org.opencontainers.image.documentation="https://github.com/atomashevic/transforEmotion#docker-and-apptainer" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.base.name="docker.io/rocker/r-ver:4.6.1"

# Each value is read first, so a failing command fails the build. The uv
# cache that reticulate wrote under /root during the checks is removed.
RUN te=$(Rscript -e 'cat(format(packageVersion("transforEmotion")))') \
 && py=$(${TRANSFOREMOTION_PYTHON} -c 'import platform; print(platform.python_version())') \
 && torch=$(${TRANSFOREMOTION_PYTHON} -c 'import torch; print(torch.__version__)') \
 && uv=$(uv --version) && uv=${uv#uv } && uv=${uv%% *} \
 && printf '%s\n' \
      "transforEmotion=${te}" \
      "revision=${REVISION}" \
      "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "snapshot_date=${SNAPSHOT_DATE:-latest}" \
      "R=${R_VERSION}" \
      "python=${py}" \
      "torch=${torch}" \
      "uv=${uv}" \
      "extras=${EXTRAS}" \
      "gpu=${GPU}" \
      "models=${MODELS}" \
      > ${TE_HOME}/BUILD_INFO \
 && chmod a+r ${TE_HOME}/BUILD_INFO \
 && rm -rf /root/.cache

WORKDIR /work
CMD ["R"]
