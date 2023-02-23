#' Creates installation of python environment
#'
#' @export
create_python_environment <- function() {
  env <- "venv-AvGPR"
  reticulate::conda_create(env)
  reticulate::py_install("numpy", pip=TRUE)
  reticulate::py_install("GPy", pip=TRUE)
  reticulate::py_install("scikit-learn", pip=TRUE)

  reticulate::use_condaenv(env)
}
