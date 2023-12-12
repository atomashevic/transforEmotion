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

reticulate::import("pandas")
##### IMAGE TEST

url = "https://www.thoughtsinvinyl.com/Images/Medium/gr127jpg20140924091458.jpg"

classes = c("anger", "disgust", "fear", "happinness", "sadness", "surprise", "neutral")

transforEmotion::image_scores(url, classes) # done

video = "https://www.youtube.com/watch?v=720O_yBLrTs&ab_channel=DonaldJTrump"

transforEmotion::video_scores(video, classes)

Sys.setenv(RETICULATE_PYTHON_ENV =  "transforEmotion")
library(reticulate)
