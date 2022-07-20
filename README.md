
<!-- README.md is generated from README.Rmd. Please edit that file -->

# AvGPR

<!-- badges: start -->
<!-- badges: end -->

AvGPR is a package that calculates a weighted average Gaussian Process
regression model over 5 implementations from packages in both R and
Python.

## Installation

You can install the development version of AvGPR from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("benjmuller/AvGPR")
```

This package requires the python modules ‘numpy’, ‘GPy’, ‘sklearn’, and
‘warnings’ in the python virtual environment. These modules can be
installed onto the current python virtual environment with:

``` r
install_python_packages()
```

Alternatively, the models can be installed on a new python environment
with:

``` r
create_python_environment()
```

## Example

This is a basic example of a simple GPR using the AvGPR function:

``` r
library(AvGPR)
# create_python_environment()
n <- 15
X <- seq(0, 4 * pi, length.out = n)
Y <- 2 * sin(X)
XX <- seq(0, 4 * pi, length.out=100)
AvGPR(data.frame(X=X), data.frame(Y=Y), data.frame(X=XX))
```
