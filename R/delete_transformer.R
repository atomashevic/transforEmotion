#' Delete a Transformer Model
#'
#' @description Large language models can be quite large and, when stored locally,
#' can take up a lot of space on your computer. The direct paths to where the
#' models are on your computer is not necessarily intuitive.
#'
#' This function quickly identifies the models on your computer and
#' informs you which ones can be deleted from it to open up storage space
#'
#' @param model_name Character vector.
#' If no model is provided, then a list of models that are locally stored on the
#' computer are printed
#'
#' @param delete Boolean (length = 1).
#' Should model skip delete question?
#' Defaults to \code{FALSE}.
#' Set to \code{TRUE} for less interactive deletion
#'
#' @return Returns list of models or confirmed deletion
#'
#' @author Alexander P. Christensen <alexpaulchristensen@gmail.com>
#'
#' @examples
#' if(interactive()){
#'   delete_transformer()
#' }
#' 
#' @importFrom methods is
#'
#' @export
#'
# Retrieval-augmented generation
# Updated 27.01.2024
delete_transformer <- function(model_name, delete = FALSE)
{

  # Check for model name
  if(missing(model_name)){
    model_name <- NULL
  }

  # Use the transformers library from Python
  transformers <- reticulate::import("transformers")

  # Get the default cache directory
  model_cache <- transformers$file_utils$default_cache_path

  # List models
  model_list <- list.files(model_cache)

  # Show models only
  model_list <- model_list[grepl("models--", model_list)]

  # Remove "models--"
  models <- gsub("models--", "", model_list)

  # Print models
  if(is.null(model_name)){
    print(models)
  }else{

    # Check that model_name is in models
    if(model_name %in% models){

      # Get value
      model_index <- which(model_name == models)

      # Check for quick deletion
      if(delete){

        # Remove model
        unlink(
          paste0(model_cache, "/", model_list[model_index]),
          recursive = TRUE
        )

        # Confirm deletion
        message(
          paste0(model_name, " has been removed from your computer.")
        )

      }else{

        # Print ask
        answer <- tolower(
          readline(
            prompt = paste0(
              "Are you sure you want to delete, ", model_name,
              ", off of your computer? (y/N) "
            )
          )
        )

        # Get possible answers
        possible_answers <- c("y", "n", "yes", "no")

        # Convert answer
        while(!answer %in% possible_answers){

          # Send message
          message("Answer not accepted.")

          # Try again...
          answer <- tolower(
            readline(
              prompt = paste0(
                "Are you sure you want to delete, ", model_name,
                ", off of your computer? (y/N) "
              )
            )
          )

        }

        # Check for deletion
        if(answer == "y" || answer == "yes"){

          # Remove model
          unlink(
            paste0(model_cache, "/", model_list[model_index]),
            recursive = TRUE
          )

          # Confirm deletion
          message(
            paste0(model_name, " has been removed from your computer.")
          )

        }else{

          # Confirm abort
          message(
            paste0(model_name, " was not removed from your computer.")
          )

        }

      }

    }else{
      message(
        paste0(
          model_name, ", was not found. Please double check that ",
          "model is present by typing `delete_transformer()`"
        )
      )

    }

  }

}
