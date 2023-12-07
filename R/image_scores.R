#' Calculate image scores based on OpenAI CLIP model
#'
#' This function takes an image file and a vector of classes as input and calculates the scores for each class using the OpenAI CLIP model.
#' Primary use of the function is to calculate FER scores - Facial Expession Detectection of emotions based on detected facial expression in images. In case there are more than one face in the image, the function will return the scores of the face selected using the face_selection parameter.
#' If there is no face in the image, the function will return NA for all classes.
#' Function uses reticulate to call the Python functions in the image.py file. If you run this package/function for the first time it will take some time for the package to setup a functioning Python virtual enviroment in the background. This includes installing Python libraries for facial recognition and emotion detection in text, images and video. Please be patient.
#'
#' @param image The path to the image file or URL of the image.
#' @param classes A character vector of classes to classify the image into.
#' @param face_selection The method to select the face in the image. Can be "largest" or "left" or "right". Default is "largest" and will select the largest face in the image. "left" and "right" will select the face on the far left or the far right side of the image. Face_selection method is irrelevant if there is only one face in the image.
#' @return A data frame containing the scores for each class.
#'
#' @examples
#' \donttest{boris_image = "inst/extdata/boris-1.png"
#' image_scores(boris_image, c("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"))}
#' 
#' @importFrom reticulate source_python
#' @export
#' 


image_scores <- function(image, classes, face_selection = "largest"){
  if (!(reticulate::condaenv_exists("transforEmotion"))){
    print("Creating and switching to transforEmotion virtual Python environment...")
    Sys.sleep(1)
    setup_miniconda()
  } else
  {
    reticulate::use_condaenv("transforEmotion", required = FALSE)
  }
  if (!reticulate::py_module_available("transformers") &!reticulate::py_module_available("face_recognition")){
    print("Some Python libraries are not available in the transforEmotion conda environment. We are going to set them up. We need them for facial recognition and emotion detection in text, images and video. This may take a while, please be patient.")
    Sys.sleep(1)
    setup_modules()
  }
  
  source_python(system.file("python", "image.py", package = "transforEmotion"))
  
  # check if image has image file extension
  if(!grepl("\\.(jpg|jpeg|png|bmp)$", image)){
    stop("Image file name must have an image file extension: jpg, jpeg, png, bmp")
  }
  # if not url check if file exists
  if(!grepl("^http", image)){
    if(!file.exists(image)){
      stop("Image file does not exist. If path is an URL make sure it includes http:// or https://")
    }
  }
  # check if classes is a character vector
  if(!is.character(classes)){
    stop("Classes must be a character vector.")
  }
  # check if we have at least 2 classes
  if(length(classes) < 2){
    stop("Classes must have at least 2 elements.")
  }

  # check if face_selection is valid
  if(!face_selection %in% c("largest", "left", "right")){
    stop("Argument face_selection must be one of: largest, left, right")
  }
  # if(!exists("text_embeds_openai", envir = .GlobalEnv)){
  #   print("Downloading and preparing OpenAI CLIP text embeddings...")
  #   text_embeds_openai <<- get_text_embeds(labels = classes)
  # }
  # if (! "model_openai" %in% py$globals()) {
  #    print("Downloading and preparing OpenAI CLIP model and generating text embeddings. \n Please be patient, this may take a while...")
  # }
  result <- classify_openai(image = image, labels = classes, face = face_selection)
  result <- as.data.frame(result)
  return(result)
}