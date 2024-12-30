## Goal of the tutorial

The goal of the tutorial is to demonstrate the functionalities of `transforEmotion` R package for sentiment analysis, facial expression recognition and RAG.
This is an R package that uses the power of HuggingFace `transformers` library in Python
to run model inference and perform various tasks. The tutorial will go through two
examples from the MAFW dataset. MAFW is a dataset which has large number of short videos which
are annotated (with the dominant emotion) and also have a detailed description of the
video using emotionally neutral language. The tutorial will also go through all
functionalities of the package and demostrate the use of three different models. The aim
of the tutorial is to make these tasks easy to understand, follow and apply by
communication science researches. All code can be run locally, on a CPU in reasonable
time, without need to pay for anything or run anything on a remote server. This fosters
reproducible, open science with privacy in mind, making this package usable in
communications research in situations where privacy and confidentiality is paramount.

## Call for papers

Overview 

Generative AI (GenAI) is revolutionizing the communication landscape, enabling machines to generate diverse forms of content such as text, audio, and multimedia. These advances hold immense potential for reshaping how we understand media, as they facilitate faster, more personalized, and scalable content production. However, they also raise significant questions about the ethical implications, accuracy, and societal impacts of machine-generated content. As a result, human-machine communication is emerging as a critical area of inquiry that explores how humans interact with AI-generated content, how GenAI tools can be harnessed to achieve desired societal outcomes, and how these systems shape broader communication ecosystems.

In communication research, GenAI introduces new opportunities both as a subject of study and as a methodological tool. Researchers are increasingly exploring how AI-driven content influences public discourse, media production, and audience engagement. At the same time, GenAI is being integrated into research workflows, offering innovative computational solutions for data generation and analysis. This special issue of Computational Communication Researchwill focus on the diverse ways in which GenAI intersects with communication science, from its theoretical implications to its practical applications in research.

We invite submissions on a wide range of computational communication research topics, including studies on AI’s role in media, public discourse, news production, and other communication phenomena. All submissions must employ computational methods to answer communication research questions.

1. Research Articles: We seek manuscripts that make substantive contributions to communication science by employing computational techniques to study GenAI, either as a communication phenomenon or as a research tool for understanding communication processes and human-AI or AI-AI interactions. Two broad sub-themes emerge under this category:

a. Generative AI as the Object of Research: Manuscripts in this sub-theme should apply computational techniques to investigate GenAI as a communication technology. Some examples include studies that:

Analyze large datasets of machine-generated content.

Apply network analysis to explore how GenAI impacts information diffusion.

Study how AI-generated media affects public discourse and engagement.

Compare AI and human interactions in various platforms, focusing on trust, information exchange, and relationship building.

Evaluate the effectiveness of GenAI tools for misinformation detection.

These articles should use computational methods to analyze the influence of AI-generated content on communication systems and focus on its societal implications.

b. Generative AI as a Methodological Tool: Manuscripts in this sub-theme should demonstrate how computational techniques involving GenAI are used to conduct empirical communication research. Submissions need to go beyond the simple one-shot use of GenAI for dataset generation, content coding, stimuli creation, or agent simulation, by 1) integrating innovative and creative uses of GenAI, or 2) complementing the coding process with robust validation measures, or 3) illuminating potential issues such as biases, hallucinations, etc. Some examples include studies that:

Use GenAI to generate synthetic datasets for communication research.

Use GenAI to annotate or classify communication data, including multimodal data.

Use GenAI to create experimental stimuli.

Use GenAI to develop agents that simulate human behavior.

We also welcome meta-analyses and synthesizing papers that aggregate and assess existing computational research on GenAI’s methodological potential in communication studies, with attention to these AI methodologies’ accuracy, scalability, and reliability.

2. Software Demonstration Articles: Besides traditional research articles, we invite Software Demonstration Articles that introduce new tools, platforms, or software designed to incorporate GenAI into communication research workflows. These articles should offer practical tutorials or demonstrations of software that computational researchers can adopt to integrate GenAI into their empirical work. The emphasis is on practical, open-source tools and frameworks that make it easier for other computational researchers to implement AI-driven methods in their own studies.

Submissions that explore the integration of GenAI with other computational methods and address the ethical considerations surrounding these technologies (such as bias, transparency, replicability, and societal impact) are highly encouraged. We also encourage submissions representing diverse geographical localities, cultural considerations, and disciplinary traditions.

Computational Communication Research is an open-access journal, free for both authors and readers. All accepted papers will be published under a Creative Commons Attribution license.

Submission Guidelines

All special issue extended abstracts should be submitted to the journal’s online system at https://journal.computationalcommunication.org/submission, to the section: 'Special Issue: Generative AI'

Extended abstracts are limited to 1,500 words (excluding the title page, references, tables, and figures) and should follow this format:

Title Page: Include a title, author(s), and affiliation(s), and specify the type of submission (Research Article – GenAI as Object/Tool or Software Demonstration Article).

Proposed Research (up to 1,000 words): For Research Articles, include a review of relevant literature, research questions, hypotheses, and methods. For Software Demonstration Articles, describe the software or tool, the problem it solves, and provide details about its functionality and code availability.

Timeline and Milestones (up to 100 words): Provide a clear outline of when data will be collected and analyzed and when the paper will be drafted.

Contribution to Social Science Research (up to 400 words): Outline how the proposed research or software will contribute to broader social science scholarship, particularly within the domain of computational communication research.

## Deadlines

-  Extended abstracts due: December 31, 2024

-  Decisions on extended abstracts: January 31, 2025

-  Full papers for accepted abstracts due: April 30, 2025

-  Expected publication of final articles: November 2025


## Proposed Research

**Title**: *transforEmotion: An Open-Source R Package for Emotion Analysis Using Transformer-Based Generative AI Models*

**Type of Submission**: Software Demonstration Article

---

**Abstract**

This software demonstration article introduces **`transforEmotion`**, an open-source R package designed to facilitate emotion analysis in communication research using state-of-the-art **transformer-based** generative AI models. Leveraging the HuggingFace `transformers` library in Python, `transforEmotion` provides an accessible interface for sentiment analysis, facial expression recognition, and retrieval-augmented generation (RAG) within the R environment. The package aims to empower communication scientists by allowing them to perform complex natural language processing (NLP) and computer vision tasks locally on standard CPUs, without the need for expensive hardware or proprietary software.

**Problem Statement**

In the era of Generative AI (GenAI), communication researchers face challenges in analyzing the vast amounts of multimedia data generated across various platforms. Traditional tools often require significant computational resources or advanced programming skills, creating barriers for researchers aiming to incorporate AI-driven methods into their work. There is a pressing need for accessible, open-source tools that utilize transformer models to perform sophisticated analyses while ensuring data privacy and reproducibility.

**Software Description**

`transforEmotion` addresses these challenges by integrating powerful transformer-based AI models into a user-friendly R package. The key functionalities include:

1. **Sentiment Analysis**: Utilizing pre-trained transformer models, the package can analyze textual data to compute emotion scores across various categories. Functions like `transformer_scores()` enable researchers to quantify emotional content in text, aiding in the analysis of social media posts, news articles, and other communication mediums.

2. **Facial Expression Recognition**: The package processes video files to extract frames and analyze facial expressions using transformer-based computer vision models. The `video_scores()` function leverages models such as OpenAI's CLIP (ViT-L/14), BAAI's EVA-CLIP-18B, and Jina AI's Jina-CLIP-v2 to calculate emotion scores from visual data, supporting studies on non-verbal communication and media effects. Notably, Jina-CLIP-v2 extends capabilities beyond English, enabling analysis in multiple languages.

3. **Zero-Shot Learning and Arbitrary Label Sets**: By harnessing the zero-shot capabilities of transformer models, `transforEmotion` allows researchers to test and analyze emotions using an arbitrary set of labels without the need for additional training data. In our application, we utilize 11 emotion labels, demonstrating the flexibility to adapt to various research needs.

4. **Retrieval-Augmented Generation (RAG)**: Incorporating RAG techniques, the `rag()` function enables users to generate context-aware responses based on input queries and relevant documents. This feature enhances qualitative analyses by providing AI-generated insights grounded in specific textual data.

**Functionality Demonstration**

**MAFW Dataset Application**

We demonstrate the package's capabilities using the **Multi-modal Affective Facial Expression in the Wild (MAFW)** dataset, a large-scale, multi-modal database designed for dynamic facial expression recognition. MAFW comprises over 10,000 video clips sourced from diverse cultural and thematic backgrounds, encompassing various genres such as drama, comedy, and interviews. The dataset includes 11 single-expression categories and 32 compound-expression classes, each annotated multiple times to ensure reliability. Rich annotations accompany each clip, including expression distribution vectors, bilingual emotional descriptions, facial landmarks, and gender information. By applying `transforEmotion` to this dataset, we showcase how researchers can perform comprehensive emotion analysis across different media formats and languages.

- **Video Analysis**: Two videos representing "happy" and "angry" emotions were processed. The `video_scores()` function extracted frames and computed mean emotion scores for specified labels. The analysis successfully identified the predominant emotions, with the "happy" video showing the highest mean score for "happy" and the "angry" video for "angry."

- **Textual Description Analysis**: The associated video descriptions were analyzed using `transformer_scores()`. The textual analysis mirrored the video results, with the "happy" description yielding a high score for "happy" and the "angry" description showing elevated scores for "angry," "contemptuous," and "anxious."

- **RAG Application**: Using the `rag()` function, we generated AI-driven interpretations of the emotional expressions based on the video descriptions and a specific query. The model provided coherent responses that aligned with the actual emotional content of the videos.

**Code Availability**

The `transforEmotion` package is available on GitHub under an open-source license, promoting transparency and collaboration. Comprehensive documentation, including tutorials and examples, is provided to facilitate adoption and encourage contributions from the research community.

---

## Timeline and Milestones

- **By January 31, 2025**: Complete the drafting of the full software demonstration article, incorporating feedback from the extended abstract and refining analyses, including demonstrations with multiple transformer models and multilingual capabilities.
- **February – March 2025**: Conduct additional validation studies and user testing to strengthen the software's reliability and usability across different models and languages.
- **By March 31, 2025**: Submit the full manuscript for peer review, addressing all guidelines specified by the journal.
- **April 2025**: Revise the manuscript based on peer review feedback and prepare for final submission.
- **By April 30, 2025**: Finalize and submit the revised manuscript for publication.

---

## Contribution to Social Science Research

The `transforEmotion` package substantially contributes to social science research by lowering the barriers to integrating advanced transformer-based AI methodologies into communication studies. Its accessible interface allows researchers without extensive programming backgrounds to perform sophisticated emotion analysis on textual and visual data, including multilingual content. By leveraging models capable of zero-shot learning and handling arbitrary label sets, the package provides flexibility and adaptability for various research contexts.

The inclusion of multiple transformer models, such as OpenAI's CLIP, BAAI's EVA-CLIP-18B, and Jina AI's Jina-CLIP-v2, enhances the package's capabilities, enabling analysis beyond English and supporting cross-cultural studies. This fosters a more inclusive approach to communication research, accommodating diverse linguistic and cultural backgrounds.

By facilitating local computation on standard hardware, `transforEmotion` ensures data privacy and aligns with ethical research practices, particularly important when handling sensitive or confidential communication data.

The package promotes interdisciplinary collaboration by bridging computational methods and communication theory, allowing researchers to explore new dimensions of human-machine communication, media influence, and audience engagement.

The ability to utilize zero-shot learning with arbitrary label sets empowers researchers to tailor their analyses to specific emotions or constructs of interest, without the need for extensive model retraining. This flexibility supports innovative research designs and contributes to the advancement of emotion analysis methodologies.

Moreover, `transforEmotion` supports reproducible and open science. By providing open-source code and comprehensive documentation, it encourages transparency and allows other researchers to validate, replicate, or build upon the work. This aligns with the broader goals of computational communication research to enhance methodological rigor and fosters a collaborative community focused on advancing the understanding of communication phenomena in the age of GenAI.

In summary, `transforEmotion` serves as a valuable tool that enhances research capabilities, supports ethical standards, and contributes to the theoretical and practical advancement of computational communication research, particularly in the utilization of transformer-based models for emotion analysis.

---


