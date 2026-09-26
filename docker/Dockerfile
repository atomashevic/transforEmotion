# transforEmotion with a fixed Python environment and pre-downloaded models,
# for offline use (HPC clusters, air-gapped machines, Apptainer).
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t transforemotion .
#   docker build -f docker/Dockerfile --build-arg EXTRAS=rag -t transforemotion:rag .
#   docker build -f docker/Dockerfile --build-arg GPU=1 -t transforemotion:cuda .
#
# Build arguments:
#   EXTRAS  comma-separated feature sets: rag, youtube, findingemo, gpu
#           ("rag" also downloads the rag() models, about 2.3 GB)
#   MODELS  comma-separated extra Hugging Face model IDs to include
#   GPU     1 for the CUDA 12.6 build of PyTorch (run with --gpus all)

ARG R_VERSION=4.6
FROM rocker/r-ver:${R_VERSION}

ARG EXTRAS=""
ARG MODELS=""
ARG GPU=0

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /usr/local/bin/

ENV TE_HOME=/opt/transforEmotion
ENV HF_HOME=${TE_HOME}/huggingface \
    NLTK_DATA=${TE_HOME}/nltk_data \
    UV_PYTHON_INSTALL_DIR=${TE_HOME}/python \
    UV_PYTHON_PREFERENCE=only-managed

# R package and its R dependencies: binaries from Posit Package Manager, with
# the system libraries they need installed by pak
COPY . /tmp/transforEmotion
RUN R -q -e 'install.packages("pak")' \
 && PKG_SYSREQS=true R -q -e 'pak::local_install("/tmp/transforEmotion", upgrade = FALSE, ask = FALSE)' \
 && rm -rf /tmp/transforEmotion

# Fixed Python environment built from the package's own requirements. uv
# cannot run from a read-only cache, so the image uses a plain virtual
# environment that transforEmotion loads through TRANSFOREMOTION_PYTHON.
RUN Rscript -e ' \
      extras <- trimws(strsplit(Sys.getenv("EXTRAS"), ",")[[1]]); \
      gpu <- Sys.getenv("GPU") == "1"; \
      if (gpu) extras <- c(extras, "gpu"); \
      writeLines(transforEmotion::python_requirements(extras = extras, gpu = gpu), "/tmp/requirements.txt")' \
 && uv venv --python 3.12 ${TE_HOME}/venv \
 && UV_LINK_MODE=copy uv pip install --no-cache --python ${TE_HOME}/venv/bin/python -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# Point transforEmotion at the environment once it exists
ENV TRANSFOREMOTION_PYTHON=${TE_HOME}/venv/bin/python

# Default models, the models for the chosen extras and any MODELS
RUN Rscript -e ' \
      extras <- trimws(strsplit(Sys.getenv("EXTRAS"), ",")[[1]]); \
      models <- trimws(strsplit(Sys.getenv("MODELS"), ",")[[1]]); \
      transforEmotion::setup_modules(extras = setdiff(extras, "gpu"), models = models)' \
 && chmod -R a+rX ${TE_HOME}

# Run offline by default; pass -e HF_HUB_OFFLINE=0 to download other models
ENV HF_HUB_OFFLINE=1

WORKDIR /work
CMD ["R"]
