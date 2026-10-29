### CRAN 0.1.9 | GitHub 0.1.9

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![R-CMD-check](https://github.com/atomashevic/transforEmotion/actions/workflows/r.yml/badge.svg)](https://github.com/atomashevic/transforEmotion/actions/workflows/r.yml) [![Downloads Total](https://cranlogs.r-pkg.org/badges/grand-total/transforEmotion?color=brightgreen)](https://cran.r-project.org/package=transforEmotion) [![DOI](https://img.shields.io/badge/DOI-10.5117%2FCCR2026.2.2.TOMA-blue)](https://doi.org/10.5117/CCR2026.2.2.TOMA) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa)](https://github.com/sponsors/atomashevic)

<!--[![Downloads per month](http://cranlogs.r-pkg.org/badges/transforEmotion)](https://cran.r-project.org/package=transforEmotion) [![DOI](https://zenodo.org/badge/464199787.svg)](https://zenodo.org/doi/10.5281/zenodo.10471354) -->


## transforEmotion: Transformer Emotion Analysis for Text, Image, Video, and Speech


<div style="text-align: center;">
  <img src="man/figures/logo.png" alt="Logo" width="35%" style="display: block; margin: 0 auto;">
</div>

With `transforEmotion` you can use transformer models for zero-shot emotion classification of text, images, video, and speech in R, and find faces and code them by hand — without the need for a GPU, subscriptions, or paid services, and without any manual Python setup. All data is processed locally on your machine, and nothing is sent to any external server or third-party service. This ensures full privacy for your data.

- [Installation and first run](#installation-and-first-run)
- [Google Colab](#google-colab)
- [How to run sentiment analysis on text?](#text-example)
- [How to run facial expression recognition on images?](#image-example)
- [How to run facial expression recognition on videos?](#video-example)

<!-- Implements sentiment analysis using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines. The default pipeline for text is [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base) trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/stanfordnlp/snli) (SNLI) and [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/nyu-mll/multi_nli) (MultiNLI) datasets. Using similar models, zero-shot classification transformers have demonstrated superior performance relative to other natural language processing models (Yin, Hay, & Roth, [2019](https://arxiv.org/abs/1909.00161)). All other zero-shot classification model pipelines can be implemented using their model name from https://huggingface.co/models?pipeline_tag=zero-shot-classification. -->

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

Optional features add their Python packages the first time they are used: `rag()` adds LlamaIndex and downloads a language model (TinyLLAMA, about 2 GB), `video_scores()` adds `pytubefix` for YouTube URLs, `speech_scores()` adds `av` (PyAV) to read audio, and the FindingEmo functions add `findingemo-light`. To prepare them in advance, run `setup_modules(extras = c("rag", "youtube", "audio", "findingemo"))`.

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

### Text in other languages

The default models are trained on English text. For text in other languages, give the ID of a multilingual zero-shot model, for example [`joeddav/xlm-roberta-large-xnli`](https://huggingface.co/joeddav/xlm-roberta-large-xnli) (XLM-RoBERTa large fine-tuned on MultiNLI and XNLI, 15 languages; the first call downloads about 2.2 GB). The classes can be in the language of the text.

```R
transformer_scores(
 text = "Estoy muy contento de verte hoy",
 classes = c("alegría", "ira", "miedo"),
 transformer = "joeddav/xlm-roberta-large-xnli"
)
```

transforEmotion 0.1.9 from CRAN does not install protobuf, which the tokenizer of this model needs. Install the GitHub version, or run `reticulate::py_require("protobuf>=3.20,<8")` after `library(transforEmotion)` and before the first analysis in each session.

Copy the model ID from the model page. For example, `facebook/xlm-roberta-large-xnli` does not exist. For offline use, add the model to `setup_modules(models = )` on a machine with network access. For the Docker image, see [Other models](#other-models).

## RAG

The `rag` function  is designed to enhance text generation using Retrieval-Augmented Generation (RAG) techniques. This function allows users to input text data or specify a path to local PDF files, which are then used to retrieve relevant documents.

Supported local LLMs include TinyLLAMA, Gemma3‑1B/4B, Qwen2.5‑1.5B/3B, and Ministral‑3B via HuggingFace. No Ollama is required. Specifically: `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, and `ministral/Ministral-3b-instruct`. The default model is TinyLLAMA for speed. Gemma 3 is gated (a Hugging Face token is needed) and requires transformers >= 4.50; Qwen 2.5 is not gated and works with the transformers version the package installs.

Here's an example based on the decription of this package. First, we specify the text data.

```R
text <- "With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion
        classification of text, image, and video in R, *all without the need for a GPU,
        subscriptions, paid services, or using Python. Implements sentiment analysis
        using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines.
        The default pipeline for text is
        [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base)
        trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/stanfordnlp/snli) (SNLI) and
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

Supported local LLMs include TinyLLAMA, Gemma3‑1B/4B, Qwen2.5‑1.5B/3B, and Ministral‑3B via HuggingFace. No Ollama is required. Specifically: `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, and `ministral/Ministral-3b-instruct`.

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

For per-document emotion/sentiment with small local LLMs (Gemma3‑1B/4B or Qwen2.5‑1.5B/3B), use the convenience wrapper:

```R
texts <- c(
  "I feel so happy and grateful today!",
  "This is frustrating and makes me angry."
)
rag_sentemo(texts, task = "emotion", output = "table", transformer = "Qwen2.5-3B")
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

If there is no face detected in the image, the output is one row of `NA` scores.

If there are multiple faces detected in the image, by default the function will return the FER scores for the largest (focal) face. Alternatively, you can select the face on the left or the right side of the image by specifying the `face_selection` argument, or score every face with `face_selection = "all"`.

Faces are found with OpenCV's YuNet detector. To reproduce scores from versions before 0.1.8, which used a Haar cascade, set `face_detector = "haar"`.

## Inspecting and Choosing Faces

`detect_faces()` lists every face it finds in images or in frames sampled from a video, with its box, confidence, a track id that follows it from frame to frame, and a person id shared by all faces of the same person (OpenCV's SFace face recognizer; its 10 MB model is downloaded on first use). Pick the faces to analyse with `select_faces()` or ordinary subsetting, and score them with `face_scores()`:

```r
faces <- detect_faces("interview.mp4", nframes = 50)
table(faces$person_id)

# Draw the boxes, track and person on a few frames
draw_faces(faces, max_images = 6)

# Every face of one person, across the whole video
scores <- face_scores(select_faces(faces, persons = "p1"), classes = emotions)
```

To look at the faces in your web browser instead, use `view_faces(faces)`: R opens a page that shows every image or sampled frame with its boxes, coloured by person, and a list of the people found. With `select = TRUE`, you click faces, video tracks or people and then Send to R, and `view_faces()` returns those faces:

```r
chosen <- view_faces(faces, select = TRUE)
scores <- face_scores(chosen, classes = emotions)
```

The page is served by R on your own computer only and makes no outside requests. When R runs on a remote computer, forward the port over SSH (`view_faces()` prints the command), or use `live = FALSE` to write the page as a folder to copy, and read its exported choice with `read_face_selection()`.

In a Jupyter notebook, such as Google Colab, `view_faces()` shows the page below the cell instead. Click the faces, then **Copy R code**, and run the copied line in a new cell:

```r
view_faces(faces, select = TRUE)
# then, in a new cell, the copied line, for example
chosen <- read_face_selection("te-selection:eyJzY2hlbWEiOi...", faces)
```

To code faces by hand, for example to check the model against people, `annotation_task(faces, classes = emotions, coders = c("C01", "C02"), dir = "round-1")` sets up a coding task, `annotate_faces()` opens the coding page and saves each answer as it is given, and `read_annotations()`, `annotation_agreement()` and `consensus_labels()` read the answers, measure agreement between coders and give one label per face for `evaluate_emotions()`. In a Jupyter notebook such as Google Colab, `annotate_faces()` shows the coding page below the cell, and **Copy R code** gives a line, `answers <- read_annotations(task, "te-annotations:...")`, to run in a new cell. The vignette "Checking faces and coding them by hand" walks through it.

The same tasks can hold video clips, images or texts instead of single faces, for example to review the clips a model got wrong. Each item shows sampled frames with the face boxes, the clip with its sound and its text; the reference label, the model's answer and its scores appear after the coder's own answer, followed by a question on what explains the result:

```r
emotions <- c("angry", "fearful", "happy", "sad")
items <- data.frame(id = errors$clip, media = errors$path, text = errors$description,
                    label = errors$label, prediction = errors$predicted, errors[, emotions])
review <- annotation_task(items = items, classes = emotions, coders = c("C01", "C02"), dir = "error-review")
annotate_faces(review)

answers <- read_annotations(review)
table(answers$value[answers$question == "why"])
consensus <- consensus_labels(answers)
evaluate_emotions(consensus, truth_col = "reference", pred_col = "label")  # coders vs the dataset's label
```

To try this, the package comes with a picture of two people (`system.file("extdata", "two-people.jpg", package = "transforEmotion")`) and a short clip in which both leave and one of them comes back alone (`"two-people.mp4"`).

The same table can be passed as `face_selection` to `image_scores()`, `image_scores_dir()` and `video_scores()`. Grouping faces by person processes biometric data; use `identify = FALSE` to turn it off.

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
- `jina-v2`: "jinaai/jina-clip-v2" - Another large model with high accuracy but requires more resources (~6GB of RAM). Its text encoder is multilingual, so labels can be written in languages other than English.
- `oai-base-fer`: "tanganke/clip-vit-base-patch32_fer2013" - `oai-base` with its vision encoder fine-tuned on the FER2013 facial-expression dataset. Same resources as `oai-base`.
- `oai-large-fer`: "tanganke/clip-vit-large-patch14_fer2013" - `oai-large` with its vision encoder fine-tuned on FER2013. Same resources as `oai-large`.

#### Fine-tuned vision encoders

`oai-base-fer` and `oai-large-fer` are checkpoints that hold only a fine-tuned CLIP vision encoder. transforEmotion loads the text encoder and processor from the base CLIP model named in the checkpoint's `config.json` and replaces its vision encoder, so the labels still work zero-shot. Any Hugging Face checkpoint of this kind (model type `clip_vision_model`, saved with `save_pretrained()` from a CLIP model) can be used the same way, by its ID or through `register_vision_model(architecture = "clip")`.

The fine-tuned models recognise facial expressions better than the originals, and label wording matters too. On a sample of the MAFW video dataset (100 clips per basic emotion, of which 682 had a detectable face; 10 frames per clip, `face_selection = "largest"`, clip prediction = highest mean score), accuracy for seven basic emotions was:

| Model | `"angry"`, `"happy"`, ... | `"a photo of an angry face"`, ... |
|---|---|---|
| `oai-base` | 0.27 | 0.33 |
| `oai-large` | 0.33 | 0.34 |
| `oai-base-fer` | 0.41 | 0.40 |
| `oai-large-fer` | 0.44 | 0.48 |

Chance is 0.14. Validate the model and label wording on a labelled sample from your own data before a full analysis (see `evaluate_emotions()`).

```R
emotions <- c("a photo of an angry face", "a photo of a disgusted face", "a photo of a fearful face",
              "a photo of a happy face", "a photo of a neutral face", "a photo of a sad face",
              "a photo of a surprised face")
image <- system.file("extdata", "boris-1.png", package = "transforEmotion")
image_scores(image, classes = emotions, model = "oai-large-fer")
```

> **Note:** The memory requirements listed above are approximate and represent the minimum RAM needed. For optimal performance, we recommend having at least 16GB of system RAM when using any of these models. If you're processing videos or multiple images in batch, more RAM might be needed. When using GPU acceleration, similar VRAM requirements apply. We recommend using 'oai-base' or 'oai-large' for most applications as they provide a good balance between accuracy and resource usage.

## Speech Example

`speech_scores()` splits the audio track of an audio file, a video file or a YouTube video into segments and scores each segment against the classes with a CLAP audio-text model (`laion/larger_clap_music_and_speech`), zero-shot, as `image_scores()` does with CLIP. The segments are chosen as `video_scores()` chooses frames: `nsegments`, `segment_length` (at most 10 seconds), `start`, `end` and `uniform`.

```R
emotions <- c("a person speaking angrily", "a person speaking happily",
              "a person speaking sadly", "a person speaking in a neutral voice")

# Consecutive 5-second segments from 10 s to 120 s
result <- speech_scores("interview.mp4", classes = emotions,
                        nsegments = 20, segment_length = 5, start = 10, end = 120)
```

The result has one row per segment, with `segment`, `start` and `end` (seconds in the file) and one column of scores per class.

LAION CLAP is a general audio model and separates speech emotions poorly. On RAVDESS (1440 acted English clips from 24 actors, one segment per clip, clip prediction = highest score), unweighted average recall was:

| Classes | `"angry"`, `"happy"`, ... | `"a person speaking angrily"`, ... |
|---|---|---|
| 8 emotions (chance 0.125) | 0.13 | 0.17 |
| angry, happy, neutral, sad (chance 0.25) | 0.28 | 0.33 |

Most clips were predicted as neutral. Use the scores for exploration only, and validate them on a labelled sample from your own data (see `evaluate_emotions()`).

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

When an NVIDIA GPU is detected on Linux or Windows, the CUDA 12.6 build of PyTorch is installed; otherwise the CPU-only build is used. On macOS, transforEmotion runs on the CPU, also on Apple Silicon: `device = "auto"` chooses CUDA or the CPU and never the Apple GPU (MPS). GPU use needs only an NVIDIA driver recent enough for CUDA 12 (R525 or newer); the CUDA libraries come with PyTorch, so no CUDA Toolkit or compiler is required.

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

After each update of transforEmotion, run the login-node lines once more, so that the cache gets any new Python packages. Until then, an offline job stops at its first analysis function, with uv's reason and the lines to run on the login node.

On clusters with Apptainer, you can also use the Docker image (see [Apptainer](#apptainer)). The image needs no writable cache folder. After an update of transforEmotion, pull the new image instead of preparing the cache again.

You can also run the lines above in the image, for example to add models. `setup_cache()` then keeps the Python environment of the image. It sets the model folder to `<path>/huggingface/hub`, so the models of the image are not used, and `setup_modules()` downloads the default models (about 1 GB) into the cache. Two changes are necessary on the login node. First, leave out `extras`, because the published image has only the core feature set (see [What is in the image](#what-is-in-the-image)). For example, `extras = "rag"` stops with "ModuleNotFoundError: No module named 'nltk'". Second, add `--env HF_HUB_OFFLINE=0` (`-e HF_HUB_OFFLINE=0` in Docker). The image sets `HF_HUB_OFFLINE=1`, and `setup_cache()` without `offline = TRUE` does not change it. Without the flag, `setup_modules()` prints "HF_HUB_OFFLINE is set; skipping model downloads." and downloads nothing. `gpu` has no effect in the image.

```bash
# Once, on a login node (with internet access)
apptainer exec --env HF_HUB_OFFLINE=0 transforemotion_main.sif Rscript -e \
  'library(transforEmotion); setup_cache("/project/mylab/transforEmotion-cache"); setup_modules(models = "facebook/bart-large-mnli")'
```

The job lines stay the same. The first offline job prints the transformers warning that [Other models](#other-models) describes.

### Docker and Apptainer

The Docker image holds transforEmotion, a fixed Python environment and the default models. It runs offline by default. Pull it with:

```bash
docker pull ghcr.io/atomashevic/transforemotion:main
```

The download is about 1.8 GB. The images are for x86_64 (linux/amd64). On a Mac with Apple Silicon, install transforEmotion natively. `docker run --platform linux/amd64` runs the image under emulation, which is slow and not tested.

| Tag | Content | Changes |
|---|---|---|
| `main` | the GitHub version (branch main) | at each build from main |
| `sha-<commit>` | one build of one commit; `<commit>` is the first 7 characters of the commit | when the workflow runs again for that commit |
| `X.Y.Z` | the GitHub release X.Y.Z | when the workflow runs again for that release |
| `latest` | the newest GitHub release that is not a pre-release | at each GitHub release that is not a pre-release |

The workflow makes the `X.Y.Z` and `latest` tags when a GitHub release is published, with the tag `vX.Y.Z` or `X.Y.Z`. These tags start with the 0.1.10 release. Only the digest names one build. To pin an image in a script, use the digest:

```bash
docker buildx imagetools inspect ghcr.io/atomashevic/transforemotion:main --format '{{.Manifest.Digest}}'   # prints sha256:<digest>
docker pull ghcr.io/atomashevic/transforemotion@sha256:<digest>
```

Start R with the current folder as the working folder:

```bash
docker run --rm -it -v "$PWD":/work ghcr.io/atomashevic/transforemotion:main
```

To run a script instead, add `Rscript analysis.R` after the image name. On Linux, add `--user "$(id -u):$(id -g)" -e HOME=/tmp`. Without `--user`, the files that R writes in your folder belong to root. The image also runs from a read-only file system. It needs only a writable `/tmp`:

```bash
docker run --rm -it --read-only --tmpfs /tmp:exec --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD":/work ghcr.io/atomashevic/transforemotion:main
```

#### What is in the image

- transforEmotion and the R packages it needs, on R 4.6.
- A Python 3.12 environment in `/opt/transforEmotion/venv` with the core Python packages and the CPU build of PyTorch. `TRANSFOREMOTION_PYTHON` points transforEmotion at it, so transforEmotion does not install Python packages in the image.
- The default models: `cross-encoder/nli-distilroberta-base` for `transformer_scores()`, `openai/clip-vit-base-patch32` for `image_scores()` and `video_scores()`, `sentence-transformers/all-MiniLM-L6-v2` for `sentence_similarity()`, and the face recognition model of `detect_faces()`.
- `HF_HUB_OFFLINE=1`, so the image does not connect to Hugging Face.

The image does not have the Python packages for `rag()`, `speech_scores()`, YouTube URLs and `download_findingemo_data()`. In the image, these functions stop with an error that names the missing modules. For example, `rag()` stops with:

```
Error: The Python environment that TRANSFOREMOTION_PYTHON points to does not have these modules of the feature sets that this function needs:
  rag: llama_index.core, llama_index.llms.huggingface, llama_index.embeddings.huggingface, pypdf, rank_bm25
transforEmotion does not install packages into a fixed environment. Install them with
  uv pip install --python "/opt/transforEmotion/venv/bin/python" "llama-index-core>=0.10.30,<0.11" "llama-index-llms-huggingface" "llama-index-embeddings-huggingface" "pypdf" "rank-bm25"
or rebuild the environment from python_requirements(extras = "rag"). For a Docker image, build it from docker/Dockerfile with --build-arg EXTRAS=rag. In extras and in EXTRAS, also list every other feature set that the environment needs.
```

For these features, build your own image with `EXTRAS` (see [Build your own image](#build-your-own-image)).

#### Other models

To use a model that is not in the image, build your own image with `MODELS`, or keep the models in a folder on the host. Download the models into the folder once, with network access. Then mount the folder in each run:

```bash
mkdir -p hf
# Once, with network access: the default models and the extra model
docker run --rm -v "$PWD/hf":/hf -e HF_HUB_CACHE=/hf -e SENTENCE_TRANSFORMERS_HOME=/hf \
  -e HF_HUB_OFFLINE=0 ghcr.io/atomashevic/transforemotion:main \
  Rscript -e 'transforEmotion::setup_modules(models = "joeddav/xlm-roberta-large-xnli")'

# Each later run, offline
docker run --rm -it -v "$PWD":/work -v "$PWD/hf":/hf \
  -e HF_HUB_CACHE=/hf -e SENTENCE_TRANSFORMERS_HOME=/hf \
  ghcr.io/atomashevic/transforemotion:main
```

On Linux, add `--user "$(id -u):$(id -g)" -e HOME=/tmp` to both commands, as above. Without `--user`, the files in `hf` belong to root. You then cannot delete them as your user, and a later download with `--user` into `hf` fails with "PermissionError".

`HF_HUB_CACHE` is the model folder of transformers, and `SENTENCE_TRANSFORMERS_HOME` the model folder of `sentence_similarity()`. Set both to the same folder. The folder replaces the model folder of the image. For this reason, `setup_modules()` also downloads the default models (about 1 GB) into `hf`. The first offline run prints "You are offline and the cache for model files in Transformers v4.22.0 has been updated". The models load correctly, and transformers then writes `hf/version.txt`, so later runs do not print it. If you set only `-e HF_HUB_OFFLINE=0`, transforEmotion downloads the model into the container, and `--rm` deletes the model when the container stops. With `--user`, this download fails, because only root can write to the model folder of the image. In Apptainer, it fails because the image is read-only.

#### Apptainer

On a login node, with network access:

```bash
export APPTAINER_CACHEDIR=/project/mylab/apptainer-cache  # the default is in your home folder
export APPTAINER_TMPDIR=/project/mylab/apptainer-tmp      # the default is /tmp; it must hold the unpacked image
apptainer pull docker://ghcr.io/atomashevic/transforemotion:main
```

The command makes the file `transforemotion_main.sif`. In a job:

```bash
apptainer exec transforemotion_main.sif Rscript analysis.R
```

Apptainer mounts your home folder and the current folder, so the script can read and write your files. To use other models, prepare a model folder as for Docker. On the login node, add `--env HF_HUB_CACHE=$PWD/hf --env SENTENCE_TRANSFORMERS_HOME=$PWD/hf --env HF_HUB_OFFLINE=0` and run `setup_modules(models = )` once. In jobs, add `--env HF_HUB_CACHE=$PWD/hf --env SENTENCE_TRANSFORMERS_HOME=$PWD/hf`.

Apptainer passes the variables of your shell into the container, except the variables that the image sets. The image sets, among others, `HF_HOME`, `HF_HUB_CACHE`, `SENTENCE_TRANSFORMERS_HOME`, `HF_HUB_OFFLINE`, `TRANSFOREMOTION_PYTHON`, `RETICULATE_PYTHON` and `RETICULATE_PYTHON_ENV`, so the values of your shell do not apply to them. Use `--env` to change them.

Three other variables hide the models of the image from transformers: `TRANSFORMERS_CACHE`, `PYTORCH_TRANSFORMERS_CACHE` and `PYTORCH_PRETRAINED_BERT_CACHE`. The image does not set them, because transformers warns in each session when one of them is set. If your shell sets one of them, the text and image models of the image do not load. `transformer_scores()` and `image_scores()` then stop with "OSError: We couldn't connect to 'https://huggingface.co' to load this file". `sentence_similarity()` still works. `setup_cache()` unsets these variables in its session, but it also sets the model folder to its cache (see [HPC clusters](#hpc-clusters)). To use the models of the image, unset the variables for the call, for example `env -u TRANSFORMERS_CACHE apptainer exec transforemotion_main.sif Rscript analysis.R`, or use `apptainer exec --cleanenv`.

R in the container reads `.Renviron` from the current folder or, if there is none, from your home folder. Its values take precedence over the variables of the image. If the file sets `TRANSFOREMOTION_PYTHON` or a model folder variable, add `--env R_ENVIRON_USER=/dev/null` to the call, or remove those lines. In Docker, this applies to a `.Renviron` in the folder that you mount on `/work`.

#### An image never changes

An image keeps the transforEmotion version, the Python packages and the models of its build. An update of transforEmotion on your computer does not change the image. To get a fix, pull the image again (`docker pull`, or `apptainer pull --force`), or build it again.

For example, images built before the fix for issue #46 do not have the Python package protobuf. In these images, `transformer_scores()` fails for models such as `joeddav/xlm-roberta-large-xnli` with "requires the protobuf library". To check an image:

```bash
docker run --rm IMAGE /opt/transforEmotion/venv/bin/python -c "import google.protobuf"
apptainer exec IMAGE.sif /opt/transforEmotion/venv/bin/python -c "import google.protobuf"
```

If the command prints nothing, the image has protobuf. If it prints `ModuleNotFoundError`, pull or build a new image. If you cannot replace the image, see `?python_requirements`.

#### What a given image holds

```bash
docker run --rm ghcr.io/atomashevic/transforemotion:main cat /opt/transforEmotion/BUILD_INFO
docker run --rm ghcr.io/atomashevic/transforemotion:main cat /opt/transforEmotion/python-packages.txt
docker run --rm ghcr.io/atomashevic/transforemotion:main cat /opt/transforEmotion/models.txt
```

`BUILD_INFO` gives the package version, the transforEmotion commit of the build (`revision`), the build date, the versions of R, Python, PyTorch and uv, and the build arguments `SNAPSHOT_DATE`, `EXTRAS`, `GPU` and `MODELS`. `python-packages.txt` lists each Python package with its version. It lists PyTorch and torchvision by the URL of the wheel, and `BUILD_INFO` gives the PyTorch version. `models.txt` lists each model with its revision and its size in bytes. The label `org.opencontainers.image.revision` also gives the commit. In Apptainer, use `apptainer exec transforemotion_main.sif cat /opt/transforEmotion/BUILD_INFO`.

`packageVersion("transforEmotion")` does not identify an image, because builds of the GitHub version report the number of the last CRAN release. Images built from an older `docker/Dockerfile` have the labels of the base image, rocker/r-ver, and no `BUILD_INFO`. For these images, use `docker run --rm IMAGE uv pip list --python /opt/transforEmotion/venv/bin/python`.

#### Build your own image

Build from the repository root, with Docker 23 or newer:

```bash
docker build -f docker/Dockerfile -t transforemotion .                        # CPU
docker build -f docker/Dockerfile --build-arg EXTRAS=rag -t transforemotion:rag .
docker build -f docker/Dockerfile --build-arg GPU=1 -t transforemotion:cuda .  # NVIDIA
docker run --rm -it -v "$PWD":/work transforemotion
```

`EXTRAS` adds optional feature sets, separated by commas: `rag`, `youtube`, `audio`, `findingemo` and `gpu`. `rag` also adds the models of `rag()` (about 2.3 GB), and `audio` the model of `speech_scores()` (about 800 MB). `MODELS` adds Hugging Face models, separated by commas, for example `--build-arg MODELS=joeddav/xlm-roberta-large-xnli`. `GPU=1` installs the CUDA 12.6 build of PyTorch. `--build-arg REVISION="$(git rev-parse HEAD)"` records the commit in `BUILD_INFO`.

To use a local image in Apptainer, convert it with `apptainer build transforemotion.sif docker-daemon://transforemotion:latest`. Apptainer images are read-only, which is why the image uses a fixed environment rather than uv's cache. On an arm64 machine, the build gives an arm64 image. This image also holds the CUDA 13 libraries of PyTorch (about 2.5 GB to download), because transforEmotion selects the CPU build of PyTorch only on x86_64.

To build your own fixed environment, for example in a read-only software folder, install the output of `python_requirements()` into a Python 3.12 virtual environment. Then set `TRANSFOREMOTION_PYTHON` to its Python executable, and transforEmotion uses it without installing anything. Rebuild the fixed environment after you update transforEmotion, because a new version can add Python packages (see `?python_requirements`).

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

- `two-people.jpg`, `two-people.mp4`: made from `boris-1.png`, `boris-2.png` and `trump1.jpg` by `data-raw/example-media.py`, under the same terms as those images (credit the YouTube video above for the Boris Johnson frames).

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
  url     = {https://doi.org/10.5117/CCR2026.2.2.TOMA}
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
