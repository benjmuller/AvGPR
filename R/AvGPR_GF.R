#' Calculates the weighting statistic
#'
#' @param y_pred vector of mean predicted values
#' @param y_true vector of true values
#' @param var vector of variances
#' @param bool if 'var' is standard deviation
#' @return weighting statistic
#' @examples
#' y_pred <- c(1, 2, 4.1, 5.2)
#' y_true <- c(1.5, 2.5, 3.5, 4.5)
#' var <- c(0.6, 0.8, 1, 1.1)
#' AvGPR_GF(y_pred, y_true, var, sd = TRUE)
#'
#' @export
AvGPR_GF <- function(y_pred, y_true, var, sd = FALSE) {
  residual <- abs(y_pred - y_true)
  if (sd) {
    return(sum((var - residual) ^ 2 + residual ^ 2))
  }
  else {
    return(sum((sqrt(var) - residual) ^ 2 + residual ^ 2))
  }
}
