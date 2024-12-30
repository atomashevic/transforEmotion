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
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>
#' @importFrom reticulate source_python
#' @importFrom reticulate py
#' @export


image_scores <- function(image, classes, face_selection = "largest", model = "openai/clip-vit-large-patch14") {
  if (!conda_check()){
      stop("Python environment 'transforEmotion' is not available. Please run setup_miniconda() to install it.")
    
  }
  else {
    reticulate::use_condaenv("transforEmotion", required = FALSE)
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
  # Check if model is valid
  valid_models <- c("openai/clip-vit-large-patch14", "BAAI/EVA-CLIP-14B", "jinaai/jina-clip-v2")
  if (!model %in% valid_models) {
    stop(paste("Invalid model specified. Allowed models are:", paste(valid_models, collapse = ", ")))
  }
  
  result <- reticulate::py$classify_image(image = image, labels = classes, face = face_selection, model_name = model)
  result <- as.data.frame(result)
  return(result)
}

#
