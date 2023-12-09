# NEW R SESSION

roxygen2::roxygenise()
devtools::build()
remove.packages("transforEmotion")
devtools::install(upgrade = "never")

########################

library(transforEmotion)

reticulate::conda_remove("transforEmotion")

transforEmotion:: setup_miniconda()

transforEmotion:::conda_check()

transforEmotion:::check_python_libs()

##### IMAGE TEST

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Donald_Trump_official_portrait.jpg/330px-Donald_Trump_official_portrait.jpg"

classes = c("anger", "disgust", "fear", "happinness", "sadness", "surprise", "neutral")

transforEmotion::image_scores(url, classes) # done

video = "https://www.youtube.com/watch?v=720O_yBLrTs&ab_channel=DonaldJTrump"

transforEmotion::video_scores(video, classes, nframes = 10)

