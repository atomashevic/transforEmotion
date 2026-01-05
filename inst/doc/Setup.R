## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#"
)

## ----load, eval = TRUE, echo = FALSE, comment = NA, warning = FALSE, message = FALSE----
library(transforEmotion)

## ----transformers_scores, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE, message = FALSE----
# 
# library(transforEmotion)
# 
# # Setup Python
# 
# setup_miniconda()
# 
# # Load data
# 
# data(neo_ipip_extraversion)
# 
# # Example text
# 
# text <- neo_ipip_extraversion$friendliness[1:5] # positively worded items only
# 
# # Run transformer function
# 
# transformer_scores(
#     text = text,
#     classes = c(
#       "friendly", "gregarious", "assertive",
#       "active", "excitement", "cheerful"
#     )
# )

## ----transformers_scores_output, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE, message = FALSE----
# $`make friends easily`
#   friendly gregarious  assertive     active excitement   cheerful
#      0.579      0.075      0.070      0.071      0.050      0.155
# 
# $`warm up quickly to others`
#   friendly gregarious  assertive     active excitement   cheerful
#      0.151      0.063      0.232      0.242      0.152      0.160
# 
# $`feel comfortable around people`
#   friendly gregarious  assertive     active excitement   cheerful
#      0.726      0.044      0.053      0.042      0.020      0.115
# 
# $`act comfortably around people`
#   friendly gregarious  assertive     active excitement   cheerful
#      0.524      0.062      0.109      0.183      0.019      0.103
# 
# $`cheer people up`
#   friendly gregarious  assertive     active excitement   cheerful
#      0.071      0.131      0.156      0.190      0.362      0.089

## ----image_scores, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE, message = FALSE----
# 
# image <- ""https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
# 
# emotions <- c("excitement", "happiness", "pride", "anger", "fear", "sadness", "neutral")
# 
# image_scores(image, emotions)

## ----image_scores_output, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE----
#   excitement  happiness     pride      anger       fear   sadness   neutral
# 1 0.02142187 0.02024468 0.0604699 0.04037686 0.03273294 0.1061871 0.7185667

## ----image_scores_nas, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE, message = FALSE----
# 
# image <- "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Adelie_penguins_in_the_South_Shetland_Islands.jpg/640px-Adelie_penguins_in_the_South_Shetland_Islands.jpg"
# 
# image_scores(image, emotions)

## ----image_scores_nas_output, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE----
# No face found in the image
# data frame with 0 columns and 0 rows

## ----video_scores, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE, message = FALSE----
# 
# video_url <- "https://www.youtube.com/watch?v=hdYNcv-chgY&ab_channel=Conservatives"
# 
# emotions <- c("excitement", "happiness", "pride", "anger", "fear", "sadness", "neutral")
# 
# result <- video_scores(video_url, classes = emotions,
#                     nframes = 10, save_video = TRUE,
#                     save_frames = TRUE, video_name = 'boris-johnson',
#                     start = 10, end = 120)
# 
# head(result)

## ----video_scores_output, eval = FALSE, echo = TRUE, comment = NA, warning = FALSE----
#   excitement  happiness     pride      anger       fear   sadness   neutral
# 1 0.08960483 0.006041054 0.05632496 0.2259102 0.2781007 0.1757137 0.1683045
# 2 0.11524552 0.011083936 0.08131301 0.1672127 0.3321840 0.1652457 0.1277151
# 3 0.09541881 0.007240616 0.05629114 0.1665660 0.3410282 0.1952039 0.1382514
# 4 0.09860725 0.011296707 0.07909032 0.1693194 0.3010349 0.1759851 0.1646665
# 5 0.08856109 0.007197607 0.07237346 0.2261922 0.3237688 0.1515539 0.1303529
# 6 0.10022306 0.011431777 0.09256416 0.1467394 0.3202718 0.1574203 0.1713494

