### CRAN 0.1.7 | GitHub 0.1.7

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![R-CMD-check](https://github.com/atomashevic/transforEmotion/actions/workflows/r.yml/badge.svg)](https://github.com/atomashevic/transforEmotion/actions/workflows/r.yml) [![Downloads Total](https://cranlogs.r-pkg.org/badges/grand-total/transforEmotion?color=brightgreen)](https://cran.r-project.org/package=transforEmotion) [![DOI](https://img.shields.io/badge/DOI-10.5117%2FCCR2026.2.2.TOMA-blue)](https://doi.org/10.5117/CCR2026.2.2.TOMA) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa)](https://github.com/sponsors/atomashevic)

<!--[![Downloads per month](http://cranlogs.r-pkg.org/badges/transforEmotion)](https://cran.r-project.org/package=transforEmotion) [![DOI](https://zenodo.org/badge/464199787.svg)](https://zenodo.org/doi/10.5281/zenodo.10471354) -->


## transforEmotion: Sentiment Analysis for Text, Image and Video Using Transformer Models


<div style="text-align: center;">
  <img src="man/figures/logo.png" alt="Logo" width="35%" style="display: block; margin: 0 auto;">
</div>

With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion classification of text, image, and video in R — without the need for a GPU, subscriptions, or paid services, and without any manual Python setup. All data is processed locally on your machine, and nothing is sent to any external server or third-party service. This ensures full privacy for your data.

- [Installation and first run](#installation-and-first-run)
- [Google Colab](#google-colab)
- [How to run sentiment analysis on text?](#text-example)
- [How to run facial expression recognition on images?](#image-example)
- [How to run facial expression recognition on videos?](#video-example)

<!-- Implements sentiment analysis using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines. The default pipeline for text is [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base) trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/snli) (SNLI) and [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/nyu-mll/multi_nli) (MultiNLI) datasets. Using similar models, zero-shot classification transformers have demonstrated superior performance relative to other natural language processing models (Yin, Hay, & Roth, [2019](https://arxiv.org/abs/1909.00161)). All other zero-shot classification model pipelines can be implemented using their model name from https://huggingface.co/models?pipeline_tag=zero-shot-classification. -->

## Installation and First Run

Install the release from CRAN, or the development version from GitHub:

```R
install.packages("transforEmotion")

# Development version
install.packages("remotes")
remotes::install_github("atomashevic/transforEmotion")
```

You do not need to install Python. The first function that uses Python sets it up:

1. `reticulate` downloads [uv](https://docs.astral.sh/uv/), a fast Python package manager, if it is not installed.
2. uv installs Python 3.12 and the Python packages: about 500 MB, or about 3 GB on a machine with an NVIDIA GPU (the CUDA build of PyTorch).
3. The function downloads its model from Hugging Face, usually 100–600 MB.

This takes 1–3 minutes and happens once. Later R sessions reuse the cached environment and models: Python starts in a few seconds and nothing is downloaded again. Each new model you use is downloaded the first time you use it.

To do the setup before your first analysis, and download the default text, image and sentence-similarity models (about 1 GB):

```R
library(transforEmotion)
setup_modules()
```

Optional features add their Python packages the first time they are used: `rag()` adds LlamaIndex and downloads a language model (TinyLLAMA, about 2 GB), `video_scores()` adds `pytubefix` for YouTube URLs, and the FindingEmo functions add `findingemo-light`. To prepare them in advance, run `setup_modules(extras = c("rag", "youtube", "findingemo"))`.

Then run an analysis:

```R
library(transforEmotion)
transformer_scores(
  text = "I am so happy to see you today!",
  classes = c("joy", "anger", "fear")
)
#> $`I am so happy to see you today!`
#>         joy       anger        fear
#> 0.992855728 0.003367609 0.003776714
```

The scores are the model's probabilities that the text belongs to each class.

The Python environment and the models are stored in cache folders in your home directory (uv's cache and `~/.cache/huggingface`). To keep them elsewhere, for example on a cluster with a small home quota, or to work without internet access, see [Offline Use](#offline-use-hpc-clusters-and-containers).

If the setup fails, the error shows uv's output. Fix the cause (usually the network connection) and try again in a new R session: Python cannot be set up again once it has started in a session.

> [!WARNING]
> If you use the [radian](https://github.com/randy3k/radian) console (VSCode/terminal), its Python session may block first-time environment provisioning. Use the default R console for initial setup, then switch back if you prefer.

> [!NOTE]
> If you are using RStudio, please make sure that you are using the latest version of both R and RStudio.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atomashevic/transforEmotion/blob/main/notebooks/transforEmotion_colab.ipynb)

The notebook installs the package in a Colab R runtime and runs the text, image, video and RAG examples. It works on the free CPU runtime; on a GPU runtime (Runtime > Change runtime type > T4 GPU) the package installs the CUDA build of PyTorch and uses the GPU automatically. Colab deletes the Python environment and models when the runtime is deleted, so a new runtime repeats the first-run setup.

## Text Example

The example below uses item descriptions from the personality trait extraversion in the NEO-PI-R inventory found on the [IPIP](https://ipip.ori.org/newNEOFacetsKey.htm) website.

```R
# Load data
data(neo_ipip_extraversion)
```

For the example, the positively worded item descriptions will be used.


```R
# Example text
text <- neo_ipip_extraversion$friendliness[1:5]
```

Next, the text can be loaded in the function `transformer_scores()` to obtain the probability that item descriptions correspond to a certain class. The classes defined below are the facets of extraversion in the NEO-PI-R. The example text data draws from the friendliness facet.

```R
# Cross-Encoder DistilRoBERTa
transformer_scores(
 text = text,
 classes = c(
   "friendly", "gregarious", "assertive",
   "active", "excitement", "cheerful"
 )
)
```

The default transformer model is [DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base). The model is fast and accurate.

### BART
Another model that can be used is [BART](https://huggingface.co/facebook/bart-large-mnli), a much larger and more computationally intensive model (slower prediction times). The BART model tends to be more accurate but the accuracy gains above DistilRoBERTa are negotiatiable.

```R
# Facebook BART Large
transformer_scores(
 text = text,
 classes = c(
   "friendly", "gregarious", "assertive",
   "active", "excitement", "cheerful"
 ),
 transformer = "facebook-bart"
)
```

### Any Text Classification Model with a Pipeline on [huggingface](https://huggingface.co/models?pipeline_tag=zero-shot-classification)
Text classification models with a pipeline on huggingface can be used so long as there is a pipeline available for them. Below is an example of [Typeform's DistilBERT](https://huggingface.co/typeform/distilbert-base-uncased-mnli) model.

```R
# Directly from huggingface: typeform/distilbert-base-uncased-mnli
transformer_scores(
 text = text,
 classes = c(
   "friendly", "gregarious", "assertive",
   "active", "excitement", "cheerful"
 ),
 transformer = "typeform/distilbert-base-uncased-mnli"
)
```

## RAG

The `rag` function  is designed to enhance text generation using Retrieval-Augmented Generation (RAG) techniques. This function allows users to input text data or specify a path to local PDF files, which are then used to retrieve relevant documents.

Supported local LLMs include TinyLLAMA, Gemma3‑1B/4B, Qwen3‑1.7B, and Ministral‑3B via HuggingFace — no Ollama required. Specifically: `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `Qwen/Qwen3-1.7B-Instruct`, and `ministral/Ministral-3b-instruct`. The default model is TinyLLAMA for speed.

Here's an example based on the decription of this package. First, we specify the text data.

```R
text <- "With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion
        classification of text, image, and video in R, *all without the need for a GPU,
        subscriptions, paid services, or using Python. Implements sentiment analysis
        using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines.
        The default pipeline for text is
        [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base)
        trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/snli) (SNLI) and
        [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/nyu-mll/multi_nli) (MultiNLI) datasets.
        Using similar models, zero-shot classification transformers have demonstrated
        superior performance relative to other natural language processing models
        (Yin, Hay, & Roth, [2019](https://arxiv.org/abs/1909.00161)).
        All other zero-shot classification model pipelines can be implemented using their model name
        from https://huggingface.co/models?pipeline_tag=zero-shot-classification."
```

And then we run the `rag` function.

```R
 rag(text, query = "What is the use case for transforEmotion package?")
```

This code will provide the output similar to this one.

```
The use case for transforEmotion package is to use cutting-edge transformer
models forzero-shot emotion classification of text, image, and video in R,
without the need for a GPU, subscriptions, paid services, or using Python.
This package implements sentiment analysis using the Cross-Encoder's DistilRoBERTa
model trained on the Stanford Natural Language Inference (SNLI) and MultiNLI datasets.
Using similar models, zero-shot classification transformers have demonstrated
superior performance relative to other natural language processing models
(Yin, Hay, & Roth, [2019](https://arxiv.org/abs/1909.00161)).
The transforEmotion package can be used to implement these models and other
zero-shot classification model pipelines from the HuggingFace library.>
```

Supported local LLMs include TinyLLAMA, Gemma3‑1B/4B, Qwen3‑1.7B, and Ministral‑3B via HuggingFace — no Ollama required. Specifically: `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `Qwen/Qwen3-1.7B-Instruct`, and `ministral/Ministral-3b-instruct`.

> [!IMPORTANT] Gemma 3 access
> - Gemma 3 repos are gated. You must log in to Hugging Face and accept the model license on the model page (e.g., https://huggingface.co/google/gemma-3-1b-it).

> [!TIP] Hugging Face token handling
> - No long-term storage: The package never writes tokens to disk.
> - Gemma only: It first tries to download without a token; on 401/forbidden, it prompts you to paste a token and uses it once for that download, then discards it.
> - Scripts and notebooks (including Google Colab) cannot show the prompt: set the token first with `Sys.setenv(HF_TOKEN = "hf_...")`.
> - Public models: Downloads always ignore tokens to avoid unnecessary 401s from stale credentials.
> - Create a token at https://huggingface.co/settings/tokens. For Gemma 3, accept the model license on the model page first.

You can also request structured outputs for easier parsing and statistics.

```R
# JSON output (validated schema)
j <- rag(text, query = "Extract emotions present in the text", output = "json", task = "emotion")

# Tidy table output
rag(text, query = "Extract emotions present in the text", output = "table", task = "emotion")

# Helpers for parsing/flattening
as_rag_table(j)
parse_rag_json(j)
```

### RAG structured outputs (per-document)

For per-document emotion/sentiment with small local LLMs (Gemma3‑1B/4B), use the convenience wrapper:

```R
texts <- c(
  "I feel so happy and grateful today!",
  "This is frustrating and makes me angry."
)
rag_sentemo(texts, task = "emotion", output = "table", transformer = "Gemma3-1B")
```

## VAD Example

Directly predict Valence–Arousal–Dominance (VAD) with definitional labels and automatic fallbacks:

```R
texts <- c("I'm absolutely thrilled!", "I feel so helpless and sad", "This is boring")
vad_scores(texts, input_type = "text")

# Single dimension, simple labels
vad_scores(texts, input_type = "text", dimensions = "valence", label_type = "simple")
```

## Image Example

For Facial Expression Recognition (FER) task from images we use Open AI's [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) transformer model. Two input arguments are needed: the path to image and list of emotion labels.

Path can be either local or an URL. Here's an example of using a URL of Mona Lisa's image from Wikipedia.

```R

# Image URL or local filepath
image <- 'https://cdn.mos.cms.futurecdn.net/xRqbwS4odpkSQscn3jHECh-650-80.jpg'

# Array of emotion labels
emotions <- c("excitement", "happiness", "pride", "anger", "fear", "sadness", "neutral")

# Run FER with base model
image_scores(image, emotions, model = "oai-base")
```

You can define up to 10 emotions. The output is a data frame with 1 row and columns corresponding to emotions. The values are FER scores for each emotion.

If there is no face detected in the image, the output will be a 0x0 data frame.

If there are multiple faces detected in the image, by default the function will return the FER scores for the largest (focal) face. Alternatively, you can select the face on the left or the right side of the image by specifying the `face_selection` argument.

## Video Example

Video processing works by extracting frames from the video and then running the image processing function on each frame. Two input arguments are needed: the path to video and list of emotion labels.

Path can be either a local filepath or a **YouTube** URL. Support for other video hosting platforms is not yet implemented.

```R
# Video URL or local filepath
video_url <- "https://www.youtube.com/watch?v=hdYNcv-chgY&ab_channel=Conservatives"

# Array of emotion labels
emotions <- c("excitement", "happiness", "pride", "anger", "fear", "sadness", "neutral")

# Run FER on `nframes` of the video with large model
result <- video_scores(video_url, classes = emotions,
                    nframes = 10, save_video = TRUE,
                    save_frames = TRUE, video_name = 'boris-johnson',
                    start = 10, end = 120, model = "oai-large")
```

Working with videos is more computationally complex. This example extracts only 10 frames from the video and shouldn't take longer than a few minutes on an average laptop without GPU (depending on your internet connection needed to download the entire video and CLIP model). In research applications, we will usually extract 100-300 frames from the video. This can take much longer, so patience is advised while waiting for the results.

### Available Models

The `image_scores` and `video_scores` functions support different models. The available models are:

- `oai-base`: "openai/clip-vit-base-patch32" - A base model that is faster but less accurate. Requires ~2GB of RAM.
- `oai-large`: "openai/clip-vit-large-patch14" - A larger model that is more accurate but slower. Requires ~4GB of RAM.
- `eva-8B`: "BAAI/EVA-CLIP-8B-448" - A very large model that has been quantized to 4-bit precision for reduced memory usage (requires ~8GB of RAM instead of the original ~32GB).
- `jina-v2`: "jinaai/jina-clip-v2" - Another large model with high accuracy but requires more resources (~6GB of RAM).

> **Note:** The memory requirements listed above are approximate and represent the minimum RAM needed. For optimal performance, we recommend having at least 16GB of system RAM when using any of these models. If you're processing videos or multiple images in batch, more RAM might be needed. When using GPU acceleration, similar VRAM requirements apply. We recommend using 'oai-base' or 'oai-large' for most applications as they provide a good balance between accuracy and resource usage.

## Vision Model Registry (experimental)

Register custom/experimental vision models and list them for use in `image_scores()` and `video_scores()`:

```R
# Quick add of popular experimental models
setup_popular_models(c("blip-base", "align-base"))

# Show models
show_vision_models()

# Register a custom CLIP model
register_vision_model(
  name = "my-clip",
  model_id = "openai/clip-vit-base-patch32",
  architecture = "clip",
  description = "My CLIP baseline"
)
```

## GPU Support

When an NVIDIA GPU is detected on Linux or Windows, the CUDA 12.6 build of PyTorch is installed; otherwise the CPU-only build is used (on macOS, PyTorch supports Apple Silicon GPUs directly). GPU use needs only an NVIDIA driver recent enough for CUDA 12 (R525 or newer); the CUDA libraries come with PyTorch, so no CUDA Toolkit or compiler is required.

To override the detection, set an environment variable in a new R session before loading the package (or add it to `.Renviron`):

```R
Sys.setenv(TE_FORCE_CPU = "1")             # always use the CPU build
Sys.setenv(TRANSFOREMOTION_USE_GPU = "1")  # always use the CUDA build
library(transforEmotion)
```

Everything works in CPU mode, only slower.

## Offline Use: HPC Clusters and Containers

### HPC clusters

Compute nodes often have no internet access, and home folders often have small quotas. Keep the Python environment, Python itself and the models in one folder on shared project storage, prepare it once on a login node, and use it offline in jobs:

```R
# Once, on a login node (with internet access)
library(transforEmotion)
setup_cache("/project/mylab/transforEmotion-cache", gpu = TRUE)
setup_modules(extras = "rag", models = "facebook/bart-large-mnli")

# In each job script (no internet access needed)
library(transforEmotion)
setup_cache("/project/mylab/transforEmotion-cache", offline = TRUE, gpu = TRUE)
scores <- transformer_scores(text, classes)
```

Login nodes usually have no GPU, so set `gpu` explicitly, to the same value in both places: `TRUE` if the compute nodes have NVIDIA GPUs, `FALSE` otherwise. Offline jobs can use the default models plus any passed to `setup_modules(models = )`. The cache folder must stay writable.

### Docker and Apptainer

`docker/Dockerfile` builds an image with the package, a fixed Python environment and the default models, which runs offline by default:

```bash
docker build -f docker/Dockerfile -t transforemotion .                        # CPU
docker build -f docker/Dockerfile --build-arg EXTRAS=rag -t transforemotion:rag .
docker build -f docker/Dockerfile --build-arg GPU=1 -t transforemotion:cuda .  # NVIDIA
docker run --rm -it -v "$PWD":/work transforemotion
```

On clusters that run Apptainer (Singularity) instead of Docker, convert the image with `apptainer build transforemotion.sif docker-daemon://transforemotion:latest`. Apptainer images are read-only, which is why the image uses a fixed environment rather than uv's cache.

To build your own fixed environment, for example in a read-only software folder, install the output of `python_requirements()` into a Python 3.12 virtual environment and set `TRANSFOREMOTION_PYTHON` to its Python executable; transforEmotion then uses it without installing anything.

## Datasets: FindingEmo-Light

Reproducibly download and prepare the FindingEmo-Light dataset:

```R
# Download (optionally limit images for quick start)
download_findingemo_data("data/findingemo", max_images = 200, randomize = TRUE)

# Load and inspect annotations
ann <- load_findingemo_annotations("data/findingemo")
head(ann)
```

## Example Images

The example images included in this package (`inst/extdata/`) have different licensing terms:

- `trump1.jpg`, `trump2.jpg`: Official U.S. government portraits from Wikipedia. These works are copyright-free and therefore in the public domain under the terms of Title 17, Chapter 1, Section 105 of the U.S Code. Users may freely use, modify, and distribute these images for any purpose without attribution requirements.

- `boris-1.png`, `boris-2.png`: Screenshots from YouTube video (https://www.youtube.com/watch?v=hdYNcv-chgY). These images are licensed under Creative Commons Attribution license (reuse allowed). Users may use, modify, and distribute these images, but must provide appropriate attribution to the original source.

## Citation

If you use `transforEmotion` in your research, please cite:

Tomašević, A., Golino, H., & Christensen, A. P. (2026). transforEmotion: An open-source R package for emotion analysis using transformer-based generative AI models. *Computational Communication Research*, *8*(2). <https://doi.org/10.5117/CCR2026.2.2.TOMA>

```bibtex
@article{tomasevic2026transforemotion,
  title   = {{transforEmotion}: An Open-Source {R} Package for Emotion Analysis Using Transformer-Based Generative {AI} Models},
  author  = {Toma{\v{s}}evi{\'c}, Aleksandar and Golino, Hudson and Christensen, Alexander P.},
  journal = {Computational Communication Research},
  year    = {2026},
  volume  = {8},
  number  = {2},
  doi     = {10.5117/CCR2026.2.2.TOMA},
  url     = {https://www.aup-online.com/content/journals/10.5117/CCR2026.2.2.TOMA}
}
```

The same entry is available from R with `citation("transforEmotion")`.

## References

### BART

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019).
Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.
[arXiv preprint arXiv:1910.13461](https://arxiv.org/abs/1910.13461).

### RoBERTa

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019).
Roberta: A robustly optimized bert pretraining approach.
[arXiv preprint arXiv:1907.11692](https://arxiv.org/abs/1907.11692).

### CLIP

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. [arXiv preprint arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

### Comparison of Methods
Yin, W., Hay, J., & Roth, D. (2019).
Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
[arXiv preprint arXiv:1909.00161](https://arxiv.org/abs/1909.00161).
