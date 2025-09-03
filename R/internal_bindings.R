# Internal adapter bindings for testability
# These wrappers allow testthat::with_mocked_bindings to replace
# base/system and namespaced calls when needed during tests.

#' @noRd
system2 <- function(...)
{
  base::system2(...)
}

