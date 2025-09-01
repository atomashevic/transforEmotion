#' Vision Model Registry for transforEmotion Package
#'
#' @description
#' Central registry system for managing vision models in transforEmotion.
#' Provides extensible architecture allowing users to register custom vision models
#' beyond the default CLIP-based models.
#'
#' @details
#' The registry maintains a list of available vision models with their configurations.
#' Each model entry includes the HuggingFace model ID, architecture type, and metadata
#' needed for proper initialization and processing.
#'
#' @author Aleksandar Tomasevic <atomashevic@gmail.com>

# Initialize the vision model registry as a package environment variable
.vision_model_registry <- new.env(parent = emptyenv())

#' Register a Vision Model
#'
#' @description
#' Register a new vision model in the transforEmotion registry, making it available
#' for use with image_scores(), video_scores(), and related functions.
#'
#' @param name A short name/alias for the model (e.g., "my-custom-clip")
#' @param model_id The HuggingFace model identifier or path to local model
#' @param architecture The model architecture type. Currently supported:
#'   \itemize{
#'     \item \code{"clip"}: Standard CLIP dual-encoder models (default)
#'     \item \code{"clip-custom"}: CLIP variants requiring special handling
#'     \item \code{"blip"}: BLIP captioning/VQA models (supported via BLIP adapter)
#'     \item \code{"align"}: ALIGN dual-encoder models (supported via ALIGN adapter)
#'   }
#' @param description Optional description of the model
#' @param preprocessing_config Optional list of preprocessing parameters
#' @param requires_special_handling Logical indicating if the model needs
#'   custom processing beyond standard CLIP pipeline
#'
#' @return Invisibly returns TRUE if registration successful
#' @export
#'
#' @examples
#' \dontrun{
#' # Register a custom CLIP model
#' register_vision_model(
#'   name = "my-emotion-clip",
#'   model_id = "j-hartmann/emotion-english-distilroberta-base",
#'   architecture = "clip",
#'   description = "Custom CLIP fine-tuned on emotion datasets"
#' )
#'
#' # Register a local model
#' register_vision_model(
#'   name = "local-clip",
#'   model_id = "/path/to/local/model",
#'   architecture = "clip",
#'   description = "Locally stored fine-tuned model"
#' )
#'
#' # Register experimental BLIP model
#' register_vision_model(
#'   name = "blip-caption",
#'   model_id = "Salesforce/blip-image-captioning-base",
#'   architecture = "blip",
#'   description = "BLIP model for image captioning"
#' )
#' }
register_vision_model <- function(name,
                                  model_id,
                                  architecture = "clip",
                                  description = NULL,
                                  preprocessing_config = NULL,
                                  requires_special_handling = FALSE) {
  
  # Validate inputs
  if (!is.character(name) || length(name) != 1 || name == "") {
    stop("'name' must be a non-empty character string")
  }
  
  if (!is.character(model_id) || length(model_id) != 1 || model_id == "") {
    stop("'model_id' must be a non-empty character string")
  }
  
  valid_architectures <- c("clip", "clip-custom", "blip", "align")
  if (!architecture %in% valid_architectures) {
    stop("'architecture' must be one of: ", paste(valid_architectures, collapse = ", "))
  }
  
  # Warn if overwriting existing model
  if (name %in% names(.vision_model_registry)) {
    warning("Overwriting existing model registration: ", name)
  }
  
  # Create model configuration
  model_config <- list(
    name = name,
    model_id = model_id,
    architecture = architecture,
    description = description %||% paste("Vision model:", model_id),
    preprocessing_config = preprocessing_config,
    requires_special_handling = requires_special_handling,
    registered_at = Sys.time()
  )
  
  # Register the model
  .vision_model_registry[[name]] <- model_config
  
  # Only show message for custom models, not built-ins during package loading
  builtin_models <- c("oai-base", "oai-large", "eva-8B", "jina-v2")
  if (!name %in% builtin_models) {
    message("Successfully registered vision model '", name, "' -> ", model_id)
  }
  invisible(TRUE)
}

#' List Available Vision Models
#'
#' @description
#' List all vision models currently available in the transforEmotion registry.
#'
#' @param include_builtin Logical indicating whether to include built-in models
#'   (default: TRUE)
#' @param architecture_filter Optional character vector to filter by architecture type
#' @param verbose Logical indicating whether to show detailed information
#'   (default: FALSE)
#'
#' @return A data.frame with model information, or detailed list if verbose=TRUE
#' @export
#'
#' @examples
#' # List all available models
#' list_vision_models()
#'
#' # List only CLIP models
#' list_vision_models(architecture_filter = "clip")
#'
#' # Get detailed information
#' list_vision_models(verbose = TRUE)
#'
#' # See what models are available for image analysis
#' models <- list_vision_models()
#' print(paste("Available models:", paste(models$name, collapse = ", ")))
list_vision_models <- function(include_builtin = TRUE,
                               architecture_filter = NULL,
                               verbose = FALSE) {
  
  # Get all registered models
  model_names <- names(.vision_model_registry)
  
  if (length(model_names) == 0) {
    message("No models registered. Use register_vision_model() to add models.")
    return(data.frame())
  }
  
  # Filter by architecture if specified
  models_info <- lapply(model_names, function(name) {
    config <- .vision_model_registry[[name]]
    if (!is.null(architecture_filter) && !config$architecture %in% architecture_filter) {
      return(NULL)
    }
    config
  })
  
  # Remove NULL entries (filtered out)
  models_info <- models_info[!sapply(models_info, is.null)]
  
  if (length(models_info) == 0) {
    message("No models found matching the specified criteria.")
    return(data.frame())
  }
  
  if (verbose) {
    return(models_info)
  }
  
  # Create summary data.frame
  summary_df <- data.frame(
    name = sapply(models_info, function(x) x$name),
    model_id = sapply(models_info, function(x) x$model_id),
    architecture = sapply(models_info, function(x) x$architecture),
    description = sapply(models_info, function(x) x$description),
    stringsAsFactors = FALSE
  )
  
  rownames(summary_df) <- NULL
  return(summary_df)
}

#' Get Vision Model Configuration
#'
#' @description
#' Retrieve the configuration for a specific vision model from the registry.
#'
#' @param name The name/alias of the registered model
#'
#' @return A list with model configuration, or NULL if model not found
#' @export
get_vision_model_config <- function(name) {
  if (!is.character(name) || length(name) != 1) {
    stop("'name' must be a single character string")
  }
  
  config <- .vision_model_registry[[name]]
  
  if (is.null(config)) {
    available_models <- names(.vision_model_registry)
    stop("Model '", name, "' not found in registry.\n",
         "Available models: ", paste(available_models, collapse = ", "))
  }
  
  return(config)
}

#' Remove Vision Model from Registry
#'
#' @description
#' Remove a vision model from the transforEmotion registry.
#'
#' @param name The name/alias of the model to remove
#' @param confirm Logical indicating whether to show confirmation prompt
#'   (default: TRUE)
#'
#' @return Invisibly returns TRUE if removal successful
#' @export
remove_vision_model <- function(name, confirm = TRUE) {
  if (!is.character(name) || length(name) != 1) {
    stop("'name' must be a single character string")
  }
  
  if (!name %in% names(.vision_model_registry)) {
    stop("Model '", name, "' not found in registry")
  }
  
  if (confirm) {
    response <- readline(paste0("Remove model '", name, "' from registry? (y/N): "))
    if (!tolower(response) %in% c("y", "yes")) {
      message("Model removal cancelled")
      return(invisible(FALSE))
    }
  }
  
  rm(list = name, envir = .vision_model_registry)
  message("Successfully removed model '", name, "' from registry")
  invisible(TRUE)
}

#' Check if Vision Model is Registered
#'
#' @description
#' Check if a vision model is available in the registry.
#'
#' @param name The name/alias of the model to check
#'
#' @return Logical indicating if the model is registered
#' @export
is_vision_model_registered <- function(name) {
  if (!is.character(name) || length(name) != 1) {
    return(FALSE)
  }
  
  return(name %in% names(.vision_model_registry))
}

#' Initialize Built-in Vision Models
#'
#' @description
#' Register the default/built-in vision models that come with transforEmotion.
#' This function is automatically called when the package is loaded.
#'
#' @return Invisibly returns TRUE
#' @export
.init_builtin_models <- function() {
  # Register built-in CLIP models
  register_vision_model(
    name = "oai-base",
    model_id = "openai/clip-vit-base-patch32",
    architecture = "clip",
    description = "OpenAI CLIP ViT-Base/32 - General purpose vision-language model"
  )
  
  register_vision_model(
    name = "oai-large",
    model_id = "openai/clip-vit-large-patch14",
    architecture = "clip",
    description = "OpenAI CLIP ViT-Large/14 - Higher capacity vision-language model"
  )
  
  register_vision_model(
    name = "eva-8B",
    model_id = "BAAI/EVA-CLIP-8B-448",
    architecture = "clip-custom",
    description = "EVA CLIP 8B - Large-scale CLIP model with quantization support",
    requires_special_handling = TRUE
  )
  
  register_vision_model(
    name = "jina-v2",
    model_id = "jinaai/jina-clip-v2",
    architecture = "clip-custom",
    description = "Jina CLIP v2 - Optimized CLIP variant",
    requires_special_handling = TRUE
  )
  
  invisible(TRUE)
}

# Utility function for NULL coalescing
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}
