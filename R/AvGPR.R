#' Calculates an average Gaussian Process
#'
#' @param design Design data frame
#' @param response Response data frame
#' @param test Test data frame
#' @param nWeight Number of weight calculation simulations
#' @return Object of R3 class average Gaussian Process
#' @examples
#' n <- 100
#' X <- seq(0, 4 * pi, length.out = n)
#' Y <- 2 * sin(X)
#' XX <- seq(1,10, length.out=15)
#' AvGPR(data.frame(X), data.frame(Y), data.frame(X=XX))
#'
#' @export
AvGPR <- function(design, response, test, nWeight = 3) {

  ##-------------------------------------------------------##
  ################# Testing args and kwargs #################

  ## test design + response + test are data.frame
  if(!is.data.frame(design)) {
    stop("The design must be a data frame.",
         call. = FALSE)
  }

  if(!is.data.frame(response)) {
    stop("The response must be a data frame.",
         call. = FALSE)
  }

  if(!is.data.frame(test)) {
    stop("The test must be a data frame.",
         call. = FALSE)
  }

  ## testing design + response have the same number of rows
  if(nrow(design) != nrow(response)) {
    stop("The design and the response do not have the same number of rows.",
         call. = FALSE)
  }

  ## testing that response has 1 column
  if(ncol(response) != 1) {
    stop(paste("The number of columns in the response is ",
               ncol(response), ", this should be 1.", sep=""),
         call. = FALSE)
  }

  ## testing design and test have same number of columns
  if(ncol(design) != ncol(test)) {
    stop(paste("The number of columns in the design and test dataframes are ",
           ncol(design), " and ", ncol(test), ". These should be equal.", sep=""),
        call. = FALSE)
  }

  ## testing nWeight is intger

  if(!is.numeric(nWeight)) {
    stop(paste("nWeight is ", nWeight, ". This should be an integer.", sep=""),
         call. = FALSE)
  }

  ## Checking libraries are installed
  if (!requireNamespace("DiceKriging", quietly = TRUE)) {
    stop(
      "Package \"DiceKriging\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if (!requireNamespace("laGP", quietly = TRUE)) {
    stop(
      "Package \"laGP\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if (!requireNamespace("GauPro", quietly = TRUE)) {
    stop(
      "Package \"GauPro\" must be installed to use this function.",
      call. = FALSE
    )
  }

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop(
      "Package \"reticulate\" must be installed to use this function.",
      call. = FALSE
    )
  }


  ## Loading Python environment
  reticulate::source_python(system.file("python/GP_test_Python.py", package="AvGPR"))

  ##-------------------------------------------------------##
  ###################### Scaling data #######################

  for (i in 1:ncol(design)) {
    max_d <- max(design[,i]); min_d <- min(design[,i])
    design[,i] <- (design[,i] - min_d) / (max_d - min_d)
    test[,i] <- (test[,i] - min_d) / (max_d - min_d)
  }

  mu <- mean(response[,1]); sigma <- sd(response[,1])
  response[,1] <- (response[,1] - mu) / sigma

  ##-------------------------------------------------------##
  ##################### Calculating GP ######################

  initwarn <- getOption("warn")
  options(warn = -1)

  # TODO: if-else for matrix and df inputs
  design_mat <- as.matrix(design)
  response_mat <- as.matrix(response)
  test_mat <- as.matrix(test)
  response_vec <- as.vector(t(response))

  cat("Calculating Gaussian Processes...\n\n")

  ## DiceKriging ##
  gp_Dice <- DiceKriging::km(~., design, response, control = c(trace = FALSE))
  pred_Dice <- predict(gp_Dice, test, type = "SK")
  cat("Predicted: 1 / 5...\n")

  ## lagp ##
  pred_laGP <- laGP::aGP(design, response_vec, test,
                   end = min(50, nrow(design) - 1), verb = 0)
  cat("Predicted: 2 / 5...\n")

  ## GauPro ##
  invisible(capture.output({gp_GauPro <- GauPro::GauPro(design_mat, response_mat, nug.est = FALSE)}, type = "message"))
  pred_GauPro <- predict(gp_GauPro, test_mat, se.fit = TRUE, verbose = 0)

  cat("Predicted: 3 / 5...\n")

  ## GPy ##
  pred_GPy <- gp_GPy(design_mat, response_mat, test_mat)
  pred_GPy_mean <- as.vector(pred_GPy[[1]])
  pred_GPy_var <- as.vector(pred_GPy[[2]])

  cat("Predicted: 4 / 5...\n")

  ## sklearn ##
  pred_sklearn <- gp_sklearn(design_mat, response_mat, test_mat)
  pred_sklearn_mean <- as.vector(pred_sklearn[[1]])
  pred_sklearn_sd <- as.vector(pred_sklearn[[2]])

  cat("Predicted: 5 / 5...\n\n")

  ##-------------------------------------------------------##
  ################## Calculating weights ####################
  cat("Calculating weights...\n\n")

  GF_Dice <- 0; GF_laGP <- 0; GF_GauPro <- 0
  GF_GPy <- 0; GF_sklearn <- 0

  for (i in 1:nWeight) {
    n_design <- nrow(design)
    sample_size <- ceiling(n_design / 15)
    sample_points <- sample.int(n_design, sample_size, replace = TRUE)

    design_W <- design[-sample_points, , drop=FALSE]
    response_W <- response[-sample_points, , drop=FALSE]
    test_W <- design[sample_points, , drop=FALSE]
    test_true_W <- response[sample_points, , drop=FALSE]
    design_mat_W <- as.matrix(design_W)
    response_mat_W <- as.matrix(response_W)
    test_mat_W <- as.matrix(test_W)
    response_vec_W <- as.vector(t(response_W))

    ## DiceKriging ##
    gp_Dice_W <- DiceKriging::km(~., design_W, response_W, control = c(trace = FALSE))
    pred_Dice_W <- predict(gp_Dice_W, test_W, type = "SK")

    ## laGP ##
    pred_laGP_W <- laGP::aGP(design_W, response_vec_W, test_W,
                     end = min(50, nrow(design_W) - 1), verb = 0)

    ## GauPro ##
    invisible(capture.output({gp_GauPro_W <- GauPro::GauPro(design_mat_W, response_mat_W, nug.est = FALSE)}, type = "message"))
    pred_GauPro_W <- predict(gp_GauPro_W, test_mat_W, se.fit = TRUE, verbose = 0)

    ## GPy ##
    pred_GPy_W <- gp_GPy(design_mat_W, response_mat_W, test_mat_W)
    pred_GPy_mean_W <- as.vector(pred_GPy_W[[1]])
    pred_GPy_var_W <- as.vector(pred_GPy_W[[2]])

    ## sklearn ##
    pred_sklearn_W <- gp_sklearn(design_mat_W, response_mat_W, test_mat_W)
    pred_sklearn_mean_W <- as.vector(pred_sklearn_W[[1]])
    pred_sklearn_sd_W <- as.vector(pred_sklearn_W[[2]])

    ## Finding "Goodness of fit" ##
    GF_Dice <- GF_Dice + AvGPR_GF(pred_Dice_W$mean, test_true_W, pred_Dice_W$sd, sd = TRUE)
    GF_laGP <- GF_laGP + AvGPR_GF(pred_laGP_W$mean, test_true_W, pred_laGP_W$var)
    GF_GauPro <- GF_GauPro + AvGPR_GF(pred_GauPro_W$mean, test_true_W, pred_GauPro$s2)
    GF_GPy <- GF_GPy + AvGPR_GF(pred_GPy_mean_W, test_true_W, pred_GPy_var_W)
    GF_sklearn <- GF_sklearn + AvGPR_GF(pred_sklearn_mean_W, test_true_W, pred_sklearn_sd_W, sd = TRUE)
    cat("Simulation: ", i, "/", nWeight, "...\n")
  }
  cat("\n")

  ## Calculating weights ##
  weights <- AvGPR_weight(c(GF_Dice, GF_laGP, GF_GauPro, GF_GPy, GF_sklearn))
  Dice_weight <- weights[1]; laGP_weight <- weights[2]
  GauPro_weight <- weights[3]; GPy_weight <- weights[4]
  sklearn_weight <- weights[5]

  cat("Calculated weights:\n",
      "DiceKriging:\t", Dice_weight, "\n",
      "laGP:\t\t", laGP_weight, "\n",
      "GauPro:\t", GauPro_weight, "\n",
      "GPy:\t\t", GPy_weight, "\n",
      "sklearn:\t", sklearn_weight,"\n\n")

  options(warn = initwarn)

  ##-------------------------------------------------------##
  ################ Calculating Average model ################
  cat("Calculating Average model...\n\n")

  ## Mean ##
  AvGPR_mean <- Dice_weight * pred_Dice$mean + laGP_weight * pred_laGP$mean +
    GauPro_weight * pred_GauPro$mean + GPy_weight * pred_GPy_mean +
    sklearn_weight * pred_sklearn_mean

  ## Variance ##
  AvGPR_var <- (Dice_weight ^ 2) * (pred_Dice$sd ^ 2) + (laGP_weight ^ 2) *
    pred_laGP$var + (GauPro_weight ^ 2) * (pred_GauPro$se ^ 2) +
    (GPy_weight ^ 2 ) * pred_GPy_var + (sklearn_weight ^ 2) * (pred_sklearn_sd ^ 2)

  ##-------------------------------------------------------##
  ##################### Re-scaling data #####################
  AvGPR_mean <- AvGPR_mean * sigma + mu
  AvGPR_var <- AvGPR_var * (sigma ^ 2)

  ##-------------------------------------------------------##
  ################### Returning gp Object ###################
  avgpr <- list(mean=AvGPR_mean, var=AvGPR_var, weights=weights)
  class(avgpr) <- "avgpr"
  return(avgpr)
}
