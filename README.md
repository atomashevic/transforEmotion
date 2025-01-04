### CRAN 0.1.5 | GitHub 0.1.5

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![R-CMD-check](https://github.com/atomashevic/transforEmotion/actions/workflows/r.yml/badge.svg)](https://github.com/atomashevic/transforEmotion/actions/workflows/r.yml) [![Downloads Total](https://cranlogs.r-pkg.org/badges/grand-total/transforEmotion?color=brightgreen)](https://cran.r-project.org/package=transforEmotion) 

<!--[![Downloads per month](http://cranlogs.r-pkg.org/badges/transforEmotion)](https://cran.r-project.org/package=transforEmotion) [![DOI](https://zenodo.org/badge/464199787.svg)](https://zenodo.org/doi/10.5281/zenodo.10471354) -->


## transforEmotion: Sentiment Analysis for Text, Image and Video Using Transformer Models


<div style="text-align: center;">
  <img src="man/figures/logo.png" alt="Logo" width="35%" style="display: block; margin: 0 auto;">
</div>

With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion classification of text, image, and video in R, *all without the need for a GPU, subscriptions, paid services, or using Python*.

- [How to install the package?](#how-to-install)
- [How to run sentiment analysis on text?](#text-example)
- [How to run facial expression recognition on images?](#image-example)
- [How to run facial expression recognition on videos?](#video-example)

<!-- Implements sentiment analysis using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines. The default pipeline for text is [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base) trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/snli) (SNLI) and [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/multi_nli) (MultiNLI) datasets. Using similar models, zero-shot classification transformers have demonstrated superior performance relative to other natural language processing models (Yin, Hay, & Roth, [2019](https://arxiv.org/abs/1909.00161)). All other zero-shot classification model pipelines can be implemented using their model name from https://huggingface.co/models?pipeline_tag=zero-shot-classification. -->

## How to Install

You can find the latest stable version on [CRAN](https://cran.r-project.org/package=transforEmotion). Install it in R with:

```R
install.packages("transforEmotion")
```

If you want to use the latest development version, you can install it from GitHub using the `devtools` package.

```R
if(!"devtools" %in% row.names(installed.packages())){
  install.packages("devtools")
}

devtools::install_github("atomashevic/transforEmotion")
```

After installing the package, load it in R.

```R
# Load package
library(transforEmotion)
```

After loading package **for the first time**, you need to setup the Python virtual environment. This will download the necessary Python packages and models. This step can take a few minutes but it is only required once after installing the package on a new system.

```R
# Run Python setup
setup_miniconda()
```

> [!WARNING]
> If you using [radian](https://github.com/randy3k/radian) console in VSCode or in a terminal emulator, you won't be able to set up the transforEmotion package. Radian is written in Python and (in most cases) already runs in your default Python environment. This prevents transforEmotion package from setting up the new virtual environment and installing the correct versions of necessary Python packages. Switch to default R console and everything should work fine.


## Text Example

Next load some data with text for analysis. The example below uses item descriptions from the personality trait extraversion in the NEO-PI-R inventory found on the [IPIP](https://ipip.ori.org/newNEOFacetsKey.htm) website.

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

The rag function supports various large language models (LLMs), including TinyLLAMA, LLAMA-2, Mistral-7B, Orca-2, and Phi-2, each offering different levels of computational efficiency and quality. The default model is TinyLLAMA, which is the fastest model.

Here's an example based on the decription of this package. First, we specify the text data.

```R
text <- "With `transforEmotion` you can use cutting-edge transformer models for zero-shot emotion
        classification of text, image, and video in R, *all without the need for a GPU,
        subscriptions, paid services, or using Python. Implements sentiment analysis
        using [huggingface](https://huggingface.co/) transformer zero-shot classification model pipelines.
        The default pipeline for text is
        [Cross-Encoder's DistilRoBERTa](https://huggingface.co/cross-encoder/nli-distilroberta-base)
        trained on the [Stanford Natural Language Inference](https://huggingface.co/datasets/snli) (SNLI) and
        [Multi-Genre Natural Language Inference](https://huggingface.co/datasets/multi_nli) (MultiNLI) datasets.
        Using similar models, zero-shot classification transformers have demonstrated superior performance
        relative to other natural language processing models
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

- `oai-base`: "openai/clip-vit-base-patch32" - A base model that is faster but less accurate.
- `oai-large`: "openai/clip-vit-large-patch14" - A larger model that is more accurate but slower.
- `eva-8B`: "BAAI/EVA-CLIP-8B-448" - A very large model that requires significant HDD space and RAM.
- `jina-v2`: "jinaai/jina-clip-v2" - Another large model with high accuracy but requires more resources.

> **Note:** The larger models like `eva-8B` and `jina-v2` may take a lot of HDD space and need a lot of RAM to run efficiently. Choose the model based on your requirements for speed, accuracy, and available system resources.

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

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. [arXiv preprint arXiv:2103.00020](http://arxiv.org/abs/2103.00020)

### Comparison of Methods
Yin, W., Hay, J., & Roth, D. (2019).
Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach.
[arXiv preprint arXiv:1909.00161](https://arxiv.org/abs/1909.00161).
