#' User-Friendly Vision Model Management Functions
#'
#' @description
#' High-level functions for managing vision models in transforEmotion,
#' providing an easy interface for extending the package with custom models.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>

#' Add a Custom Vision Model
#'
#' @description
#' User-friendly wrapper for registering custom vision models with automatic
#' validation and helpful error messages.
#'
#' @param name A short, memorable name for your model (e.g., "my-emotion-model")
#' @param model_id HuggingFace model identifier or path to local model directory
#' @param description Optional description of the model and its purpose
#' @param architecture Model architecture type. Currently supported:
#'   \itemize{
#'     \item \code{"clip"}: Standard CLIP models (most compatible)
#'     \item \code{"clip-custom"}: CLIP variants needing special handling
#'     \item \code{"blip"}: BLIP models (caption-likelihood scoring)
#'     \item \code{"align"}: ALIGN dual-encoder models (direct similarity)
#'   }
#' @param test_labels Optional character vector to test the model immediately
#' @param force Logical indicating whether to overwrite existing model with same name
#'
#' @return Invisibly returns TRUE if successful
#' @export
#'
#' @examples
#' \dontrun{
#' # Add a fine-tuned CLIP model for emotion recognition
#' add_vision_model(
#'   name = "emotion-clip",
#'   model_id = "openai/clip-vit-large-patch14",
#'   description = "Large CLIP model for better emotion recognition",
#'   test_labels = c("happy", "sad", "angry"),
#'   force = TRUE
#' )
#'
#' # Add a local model
#' add_vision_model(
#'   name = "my-local-model",
#'   model_id = "/path/to/my/model",
#'   description = "My custom fine-tuned model"
#' )
#'
#' # Add experimental BLIP model
#' add_vision_model(
#'   name = "blip-base",
#'   model_id = "Salesforce/blip-image-captioning-base",
#'   architecture = "blip",
#'   description = "BLIP model for image captioning"
#' )
#'
#' # Now use any of them in analysis
#' result <- image_scores("photo.jpg", c("happy", "sad"), model = "emotion-clip")
#' batch_results <- image_scores_dir("photos/", c("positive", "negative"), 
#'                                  model = "my-local-model")
#' }
add_vision_model <- function(name,
                             model_id,
                             description = NULL,
                             architecture = "clip",
                             test_labels = NULL,
                             force = FALSE) {
  # Input validation
  if (!is.character(name) || length(name) != 1 || name == "") {
    stop("'name' must be a non-empty character string")
  }

  if (!is.character(model_id) || length(model_id) != 1 || model_id == "") {
    stop("'model_id' must be a non-empty character string")
  }

  # Check if name already exists
  if (is_vision_model_registered(name) && !force) {
    stop(
      "Model name '", name, "' already exists. Use force=TRUE to overwrite, ",
      "or choose a different name."
    )
  }

  # Validate architecture
  valid_architectures <- c("clip", "clip-custom", "blip", "align")
  if (!architecture %in% valid_architectures) {
    stop("'architecture' must be one of: ", paste(valid_architectures, collapse = ", "))
  }

  # Check if it's a local path (absolute path or relative path starting with ./ or ../)
  is_local <- grepl("^/", model_id) || grepl("^\\./", model_id) || grepl("^\\.\\./", model_id) || 
              (!grepl("^https?://", model_id) && !grepl("/", model_id))
  
  # More precise: local if it looks like a filesystem path, not a HuggingFace ID
  is_huggingface <- grepl("^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", model_id)
  is_local <- !is_huggingface && !grepl("^https?://", model_id)
  
  if (is_local && !dir.exists(model_id)) {
    stop("Local model directory does not exist: ", model_id)
  }

  # Generate description if not provided
  if (is.null(description)) {
    if (is_local) {
      description <- paste("Local model:", basename(model_id))
    } else {
      description <- paste("HuggingFace model:", model_id)
    }
  }

  # Register the model
  tryCatch(
    {
      register_vision_model(
        name = name,
        model_id = model_id,
        architecture = architecture,
        description = description,
        requires_special_handling = architecture == "clip-custom"
      )
    },
    error = function(e) {
      stop("Failed to register model: ", e$message)
    }
  )

  # Test the model if test labels provided
  if (!is.null(test_labels)) {
    message("\\nTesting model with provided labels...")
    # List of fallback test image URLs (most reliable first)
    test_image_urls <- c(
      "https://cdn2.psychologytoday.com/assets/styles/max_800/public/center/2020-07/shutterstock_653372512.jpg",
      "https://cdn.britannica.com/24/189624-050-F3C5BAA9/Mona-Lisa-oil-wood-panel-Leonardo-da.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Mona_Lisa.jpg/960px-Mona_Lisa.jpg"
    )
    
    test_image_url <- NULL
    
    # Try each URL until we find one that works
    for (url in test_image_urls) {
      url_check <- tryCatch({
        if (requireNamespace("httr", quietly = TRUE)) {
          response <- httr::HEAD(url, timeout = 5)
          if (httr::status_code(response) < 400) {
            url
          } else {
            NULL
          }
        } else {
          # Fallback if httr not available - try to download directly
          temp_file <- tempfile()
          result <- tryCatch({
            download.file(url, temp_file, mode = "wb", quiet = TRUE, timeout = 10)
            url
          }, error = function(e) NULL)
          if (file.exists(temp_file)) file.remove(temp_file)
          result
        }
      }, error = function(e) NULL)
      
      if (!is.null(url_check)) {
        test_image_url <- url_check
        break
      }
    }
    
    if (is.null(test_image_url)) {
      message("Cannot reach any test image URLs. Skipping model test.")
      message("You can test the model manually with: image_scores('your_image.jpg', test_labels, model = '", name, "')")
      return(invisible(TRUE))
    }
    message("Testing model '", name, "' with labels: ", paste(test_labels, collapse = ", "))
    message("Using test image: ", basename(test_image_url))
    
    tryCatch({
      test_result <- image_scores(test_image_url, test_labels, face_selection = "none", model = name)
      
      # Validate the results
      if (is.data.frame(test_result) && nrow(test_result) == 1) {
        expected_cols <- test_labels
        if (all(expected_cols %in% names(test_result))) {
          message("✓ Model test completed successfully!")
          message("Sample results:")
          print(test_result)
          
          # Check if probabilities sum to approximately 1
          prob_sum <- sum(as.numeric(test_result[1, test_labels]), na.rm = TRUE)
          if (abs(prob_sum - 1.0) < 0.01) {
            message("✓ Probability scores are properly normalized")
          } else {
            message("⚠ Warning: Probability scores sum to ", round(prob_sum, 3), " (expected ~1.0)")
          }
        } else {
          warning("Model test produced unexpected output format")
        }
      } else {
        warning("Model test produced unexpected result structure")
      }
    }, error = function(e) {
      warning("Model test failed: ", e$message)
      message("This could indicate:")
      message("- Model architecture incompatibility")
      message("- Missing dependencies") 
      message("- Network connectivity issues")
      message("Try testing manually once dependencies are resolved.")
    })
  }

  message("\\nModel '", name, "' successfully added!")
  message("You can now use it with: model = '", name, "'")
  message("\\nTo see all available models, use: list_vision_models()")

  invisible(TRUE)
}

#' Show Available Vision Models
#'
#' @description
#' Display all available vision models in a user-friendly format with
#' additional details and usage hints.
#'
#' @param show_details Logical indicating whether to show detailed information
#' @param filter_by Optional character vector to filter by architecture type
#'
#' @return Invisibly returns the models data.frame
#' @export
#'
#' @examples
#' # Show all models
#' show_vision_models()
#'
#' # Show only CLIP models
#' show_vision_models(filter_by = "clip")
#'
#' # Show detailed information
#' show_vision_models(show_details = TRUE)
show_vision_models <- function(show_details = FALSE, filter_by = NULL) {
  models_df <- tryCatch(
    {
      list_vision_models(architecture_filter = filter_by, verbose = show_details)
    },
    error = function(e) {
      message("Error retrieving models: ", e$message)
      return(data.frame())
    }
  )

  if (nrow(models_df) == 0) {
    message("No vision models found.")
    if (!is.null(filter_by)) {
      message("Try removing the filter or registering models with register_vision_model()")
    }
    return(invisible(models_df))
  }

  if (show_details) {
    # Detailed view
    message("\\n=== Available Vision Models (Detailed) ===")
    for (i in seq_len(length(models_df))) {
      model <- models_df[[i]]
      message("\\n", i, ". ", model$name)
      message("   Model ID: ", model$model_id)
      message("   Architecture: ", model$architecture)
      message("   Description: ", model$description)
      if (model$requires_special_handling) {
        message("   Note: Requires special handling")
      }
    }
  } else {
    # Simple table view
    message("\\n=== Available Vision Models ===")
    print(models_df)

    if (nrow(models_df) > 0) {
      message("\\nUsage: Use the 'name' column values with image_scores() or video_scores()")
      message(
        "Example: image_scores('photo.jpg', c('happy', 'sad'), model = '",
        models_df$name[1], "')"
      )
      message("\\nFor detailed information: show_vision_models(show_details = TRUE)")
    }
  }

  invisible(models_df)
}

#' Remove a Vision Model
#'
#' @description
#' Remove a custom vision model from the registry. Built-in models cannot be removed.
#'
#' @param name Name of the model to remove
#' @param confirm Logical indicating whether to show confirmation prompt
#'
#' @return Invisibly returns TRUE if successful
#' @export
#'
#' @examples
#' \dontrun{
#' # Remove a custom model
#' remove_vision_model("my-custom-model")
#'
#' # Remove without confirmation prompt
#' remove_vision_model("my-custom-model", confirm = FALSE)
#' }
remove_vision_model <- function(name, confirm = TRUE) {
  if (!is.character(name) || length(name) != 1) {
    stop("'name' must be a single character string")
  }

  # Check if model exists
  if (!is_vision_model_registered(name)) {
    available_models <- tryCatch(
      {
        models_df <- list_vision_models()
        models_df$name
      },
      error = function(e) character(0)
    )

    stop(
      "Model '", name, "' not found in registry.\\n",
      if (length(available_models) > 0) {
        paste("Available models:", paste(available_models, collapse = ", "))
      } else {
        "No models currently registered."
      }
    )
  }

  # Prevent removal of built-in models
  builtin_models <- c("oai-base", "oai-large", "eva-8B", "jina-v2")
  if (name %in% builtin_models) {
    stop("Cannot remove built-in model '", name, "'. Only custom models can be removed.")
  }

  # Get model info for confirmation
  model_config <- tryCatch(get_vision_model_config(name), error = function(e) NULL)

  if (confirm && !is.null(model_config)) {
    message("Model to remove:")
    message("  Name: ", model_config$name)
    message("  Description: ", model_config$description)
    message("  Model ID: ", model_config$model_id)

    response <- readline("Are you sure you want to remove this model? (y/N): ")
    if (!tolower(response) %in% c("y", "yes")) {
      message("Model removal cancelled.")
      return(invisible(FALSE))
    }
  }

  # Remove the model
  success <- tryCatch(
    {
      remove_vision_model(name, confirm = FALSE) # Call the registry function
      TRUE
    },
    error = function(e) {
      message("Error removing model: ", e$message)
      FALSE
    }
  )

  if (success) {
    message("Model '", name, "' successfully removed from registry.")
  }

  invisible(success)
}

#' Quick Setup for Popular Models
#'
#' @description
#' Convenience function to quickly add popular vision models with pre-configured settings.
#'
#' @param models Character vector of model shortcuts to add. Available options:
#'   \itemize{
#'     \item \code{"blip-base"}: BLIP base model for image captioning and VQA
#'     \item \code{"blip-large"}: BLIP large model for better performance
#'     \item \code{"align-base"}: ALIGN base model for image-text alignment
#'   }
#'
#' @return Invisibly returns TRUE if all models added successfully
#' @export
#'
#' @examples
#' \dontrun{
#' # Add BLIP models for image captioning
#' setup_popular_models("blip-base")
#'
#' # Add multiple experimental models at once
#' setup_popular_models(c("blip-base", "blip-large", "align-base"))
#'
#' # Then use them in your analysis
#' list_vision_models()  # See all available models
#' result <- image_scores("image.jpg", c("happy", "sad"), model = "blip-base")
#' }
setup_popular_models <- function(models) {
  if (!is.character(models) || length(models) == 0) {
    stop("'models' must be a non-empty character vector")
  }

  # Popular model configurations
  popular_configs <- list(
    "blip-base" = list(
      model_id = "Salesforce/blip-image-captioning-base",
      description = "BLIP base model for image captioning and visual question answering",
      architecture = "blip"
    ),
    "blip-large" = list(
      model_id = "Salesforce/blip-image-captioning-large",
      description = "BLIP large model for improved image understanding",
      architecture = "blip"
    ),
    "align-base" = list(
      model_id = "kakaobrain/align-base",
      description = "ALIGN base model for image-text alignment",
      architecture = "align"
    )
  )

  # Validate requested models
  unknown_models <- setdiff(models, names(popular_configs))
  if (length(unknown_models) > 0) {
    stop(
      "Unknown popular models: ", paste(unknown_models, collapse = ", "), "\\n",
      "Available options: ", paste(names(popular_configs), collapse = ", ")
    )
  }

  message("Setting up popular vision models...")
  success_count <- 0

  for (model_name in models) {
    config <- popular_configs[[model_name]]

    tryCatch(
      {
        register_vision_model(
          name = model_name,
          model_id = config$model_id,
          architecture = config$architecture,
          description = config$description
        )
        success_count <- success_count + 1
      },
      error = function(e) {
        warning("Failed to setup model '", model_name, "': ", e$message)
      }
    )
  }

  message(
    "\\nSetup complete! Successfully added ", success_count, " out of ",
    length(models), " models."
  )

  if (success_count > 0) {
    message("\\nTo see all available models: show_vision_models()")
    message("Note: These models are experimental and may require additional setup.")
  }

  invisible(success_count == length(models))
}
