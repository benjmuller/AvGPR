#' Calculates the model weights
#'
#' @param values vector of the weighting statistic
#' @return Model weights
#' @examples AvGPR_weight(c(12,24,21,16,11))
#'
#' @export
AvGPR_weight <- function(values) {
  total <- sum(values)
  return((1 - values / total) / 4)
}
