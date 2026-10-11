# Shared by test_detect_faces.R and test_view_faces.R

# A faces table as detect_faces() returns it: two images, three faces
fake_faces <- function(sources = c("a.jpg", "b.jpg")) {
  out <- data.frame(
    source = sources[c(1, 1, 2)], frame = NA_integer_, frame_index = NA_integer_,
    time = NA_real_, face = c(1L, 2L, 1L), track_id = NA_integer_,
    person_id = c("p1", "p2", "p1"), x = c(10, 200, 50), y = 20,
    width = c(50, 80, 60), height = c(50, 80, 60), confidence = 0.9,
    detector = "yunet", image_width = 400L, image_height = 300L,
    stringsAsFactors = FALSE
  )
  for (lm in .te_landmark_columns) out[[lm]] <- 1
  attr(out, "media") <- data.frame(source = sources, n_faces = c(2L, 1L))
  attr(out, "settings") <- list(type = "images", face_detector = "yunet")
  class(out) <- c("te_faces", "data.frame")
  out
}

# Tests that run the te_faces Python module need TRANSFOREMOTION_PYTHON to
# point at a Python environment with OpenCV
skip_without_face_python <- function() {
  skip_on_cran()
  skip_if(!nzchar(Sys.getenv("TRANSFOREMOTION_PYTHON")),
          "TRANSFOREMOTION_PYTHON is not set")
  ensure_te_py_env()
  skip_if_not(reticulate::py_module_available("cv2"), "OpenCV is not available")
}

fixture <- function(name) system.file("extdata", name, package = "transforEmotion")
