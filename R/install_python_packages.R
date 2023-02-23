#' Installs python packages
#'
#' @export
install_python_packages <- function() {
  reticulate::py_install("numpy", pip=TRUE)
  reticulate::py_install("GPy", pip=TRUE)
  reticulate::py_install("scikit-learn", pip=TRUE)
}
