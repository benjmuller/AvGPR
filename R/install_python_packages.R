#' Installs python packages
#'
#' @export
install_python_packages <- function() {
  reticulate::py_install("numpy", pip=TRUE)
  reticulate::py_install("GPy", pip=TRUE)
  reticulate::py_install("sklearn", pip=TRUE)
  reticulate::py_install("warnings", pip=TRUE)
}
