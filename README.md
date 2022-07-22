
<!-- README.md is generated from README.Rmd. Please edit that file -->

# AvGPR

<!-- badges: start -->
<!-- badges: end -->

AvGPR is a package that calculates a weighted average Gaussian Process
regression model over 5 implementations from packages in both R and
Python. It uses cross validation to optimise the weights to select the
regression models that are best capturing the data.

## Installation

You can install the development version of AvGPR from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("benjmuller/AvGPR")
```

This package requires the python modules ‘numpy’, ‘GPy’, ‘sklearn’, and
‘warnings’ in the python virtual environment. These modules can be
installed onto the current reticulate python virtual environment with:

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
set.seed(1)
n <- 15
X <- runif(n, min = 0, max = 4 * pi)
Y <- 2 * sin(X)
XX <- seq(0, 4 * pi, length.out=100)
AvGPR(data.frame(X=X), data.frame(Y=Y), data.frame(X=XX))
```

# Gaussian Process Tutorial

## Introduction

This document is an introduction to Gaussian Process regression (GPR)
where it will discuss; how it works, an intuitive way to think about it
and show some code for real-world examples.

## Background

A Gaussian process is a stochastic process that is used to emulate an
event that has normally distributed random variables which are indexed
over time or space. Gaussian processes are formed by the joint
distribution of these normal random variables. Gaussian Processes can be
used as a non-linear multivariate interpolation tool. This process is
called Gaussian Process regression or Kriging. Using this method, we can
write multi-output prediction problems in the following parametric form:

<center>

![\\hat{y}(x)=m(x)+Z(x),](https://latex.codecogs.com/svg.image?%5Chat%7By%7D%28x%29%3Dm%28x%29%2BZ%28x%29%2C "\hat{y}(x)=m(x)+Z(x),")

</center>

where
![\\hat{y}](https://latex.codecogs.com/svg.image?%5Chat%7By%7D "\hat{y}")
is the simulated output,
![m](https://latex.codecogs.com/svg.image?m "m") is the mean function
and ![Z](https://latex.codecogs.com/svg.image?Z "Z") is zero-mean
Gaussian Process. The mean function is a “best fit” parametric function
of the independent variables. The zero-mean Gaussian Process is
specified by a covariance function. There are many choices of covariance
functions that better suit different types of events being emulated. A
list of some standard functions can be found here
(<https://www.cs.toronto.edu/~duvenaud/cookbook/>).

## Idea behind Gaussian Processes

A way to understand Gaussian processes, in particular, getting the
parametric from the non-parametric form, is by considering a bivariate
normal:

<center>

<figure>
<img src="tools/fig1.jpeg" style="width:40.0%"
alt="A contour plot of a bivariate normal with marginal means 0, marginal variances 2 and covariances 1." />
<figcaption aria-hidden="true">A contour plot of a bivariate normal with
marginal means 0, marginal variances 2 and covariances 1.</figcaption>
</figure>

</center>

<br> The normal distributions that construct the bivariate normal have
covariance 1, hence the conditioning of an observation (data point)
gives information about what possible values the other random variable
can take. Consider the observation
![Y_0=0](https://latex.codecogs.com/svg.image?Y_0%3D0 "Y_0=0") and
plotting some random samples of possible values for the conditional
distribution
![Y_1\|Y_0=0](https://latex.codecogs.com/svg.image?Y_1%7CY_0%3D0 "Y_1|Y_0=0"):

<center>

<img src="tools/fig2.jpeg" style="width:50.0%" />

</center>

<br>

Consider expanding this notion to a 10-variate normal distribution where
we fix ![Y_0=0](https://latex.codecogs.com/svg.image?Y_0%3D0 "Y_0=0")
with
![Y_i \\sim N(0,2)](https://latex.codecogs.com/svg.image?Y_i%20%5Csim%20N%280%2C2%29 "Y_i \sim N(0,2)")
with correlations that are specified by the squared exponential
covariance function (SE):

<center>

<img src="tools/fig3.jpeg" style="width:50.0%;height:20.0%" />

</center>

<br>

Now suppose we condition the observations
![Y_0=0](https://latex.codecogs.com/svg.image?Y_0%3D0 "Y_0=0"),
![Y_6=0.3](https://latex.codecogs.com/svg.image?Y_6%3D0.3 "Y_6=0.3") and
![Y_9=-0.4](https://latex.codecogs.com/svg.image?Y_9%3D-0.4 "Y_9=-0.4")
and plot samples of possible values:

<center>

<figure>
<img src="tools/fig4.jpeg" style="width:50.0%"
alt="LHS: Random samples. RHS: Random samples overlaid onto one set of axes." />
<figcaption aria-hidden="true">LHS: Random samples. RHS: Random samples
overlaid onto one set of axes.</figcaption>
</figure>

</center>

<br>

The families of curves that join the points are subsets to the family of
solutions which are compatible with the imposed values. If we take a
large number of these sample curves, we can see that they will appear to
give regions of confidence. It is clear that at this stage, this becomes
a nonlinear regression problem. So, setting the mean and the covariance
function to be parametric functions that define the covariance in the
discrete multivariate case. This gives us the construction of the
Gaussian Process regression.

<center>

<img src="tools/fig5.jpeg" style="width:40.0%" />

</center>

## 1-Dimensional Example

In this section we will go through a one dimensional example of Gaussian
process regression. The data is created from a random sample of 8 points
on the sine function in the domain
![\[-2 \\pi, 2 \\pi\]](https://latex.codecogs.com/svg.image?%5B-2%20%5Cpi%2C%202%20%5Cpi%5D "[-2 \pi, 2 \pi]").
The randomly generated values that will be used in this example are:

<center>

![
-4.572  \\quad -1.788 \\quad-0.761 \\quad 0.095 \\quad 1.727 \\quad 3.585 \\quad 4.181 \\quad 5.454
](https://latex.codecogs.com/svg.image?%0A-4.572%20%20%5Cquad%20-1.788%20%5Cquad-0.761%20%5Cquad%200.095%20%5Cquad%201.727%20%5Cquad%203.585%20%5Cquad%204.181%20%5Cquad%205.454%0A "
-4.572  \quad -1.788 \quad-0.761 \quad 0.095 \quad 1.727 \quad 3.585 \quad 4.181 \quad 5.454
")

</center>

The covariance function that this example will be using is the squared
exponential function (SE). This has the form

<center>

![k(x_i, x_j)=\\sigma ^2 \\exp \\left(\\frac{-1}{2l^2}\\sum^d\_{k=1}(x\_{i,k}-x\_{j,k})^2 \\right)](https://latex.codecogs.com/svg.image?k%28x_i%2C%20x_j%29%3D%5Csigma%20%5E2%20%5Cexp%20%5Cleft%28%5Cfrac%7B-1%7D%7B2l%5E2%7D%5Csum%5Ed_%7Bk%3D1%7D%28x_%7Bi%2Ck%7D-x_%7Bj%2Ck%7D%29%5E2%20%5Cright%29 "k(x_i, x_j)=\sigma ^2 \exp \left(\frac{-1}{2l^2}\sum^d_{k=1}(x_{i,k}-x_{j,k})^2 \right)")

</center>

<br>

Using the samples and the chosen covariance function, this gives the
covariance matrix:

``` r
library(plgp)

kernel <- function(x1, x2, sigma = 1, l = 1) {
  matrix.dist <- distance(x1,x2)
  kern <- (sigma ^ 2) * exp(- matrix.dist / (2 * l ^ 2))
  return(kern)
}
```

<br>

![
K=
\\begin{pmatrix}
  1.0000 & 0.0208 & 0.0007 & 1.8635 \\times 10^{-5} & 2.4219\\times 10^{-9} & 3.5625\\times 10^{-15} & 2.3081\\times 10^{-17} & 1.4867\\times 10^{-22} \\\\
  0.0208 & 1.0000 & 0.5902 & 0.1699 & 0.0021 & 5.3845\\times 10^{-7} & 1.8335\\times 10^{-8} & 4.0867\\times 10^{-12} \\\\
  0.0007 & 0.5902 & 1.0000 & 0.6933 & 0.0453 & 7.9174\\times 10^{-5} & 4.9720\\times 10^{-6} & 4.0966\\times 10^{-9} \\\\
  1.8635\\times 10^{-5} & 0.1699 & 0.6933 & 1.0000 & 0.2640 & 0.0023 & 0.0002 & 5.8046\\times 10^{-7} \\\\
  2.4219\\times 10^{-9} & 0.0021 & 0.0453 & 0.2640 & 1.0000 & 0.1780 & 0.0492 & 0.0010 \\\\
  3.5625\\times 10^{-15} & 5.3845\\times 10^{-7} & 7.9174\\times 10^{-5} & 0.0023 & 0.1780 & 1.0000 & 0.8373 & 0.1744 \\\\
  2.3081\\times 10^{-17} & 1.8335\\times 10^{-18} & 4.9720\\times 10^{-6} & 0.0002 & 0.0492 & 0.8373 & 1.0000 & 0.4447 \\\\
  1.4867\\times 10^{-22} & 4.0867\\times 10^{-12} & 4.0966\\times 10^{-9} & 5.8046\\times 10^{-7} & 0.0010& 0.1744 & 0.4447 & 1.0000
\\end{pmatrix} 
](https://latex.codecogs.com/svg.image?%0AK%3D%0A%5Cbegin%7Bpmatrix%7D%0A%20%201.0000%20%26%200.0208%20%26%200.0007%20%26%201.8635%20%5Ctimes%2010%5E%7B-5%7D%20%26%202.4219%5Ctimes%2010%5E%7B-9%7D%20%26%203.5625%5Ctimes%2010%5E%7B-15%7D%20%26%202.3081%5Ctimes%2010%5E%7B-17%7D%20%26%201.4867%5Ctimes%2010%5E%7B-22%7D%20%5C%5C%0A%20%200.0208%20%26%201.0000%20%26%200.5902%20%26%200.1699%20%26%200.0021%20%26%205.3845%5Ctimes%2010%5E%7B-7%7D%20%26%201.8335%5Ctimes%2010%5E%7B-8%7D%20%26%204.0867%5Ctimes%2010%5E%7B-12%7D%20%5C%5C%0A%20%200.0007%20%26%200.5902%20%26%201.0000%20%26%200.6933%20%26%200.0453%20%26%207.9174%5Ctimes%2010%5E%7B-5%7D%20%26%204.9720%5Ctimes%2010%5E%7B-6%7D%20%26%204.0966%5Ctimes%2010%5E%7B-9%7D%20%5C%5C%0A%20%201.8635%5Ctimes%2010%5E%7B-5%7D%20%26%200.1699%20%26%200.6933%20%26%201.0000%20%26%200.2640%20%26%200.0023%20%26%200.0002%20%26%205.8046%5Ctimes%2010%5E%7B-7%7D%20%5C%5C%0A%20%202.4219%5Ctimes%2010%5E%7B-9%7D%20%26%200.0021%20%26%200.0453%20%26%200.2640%20%26%201.0000%20%26%200.1780%20%26%200.0492%20%26%200.0010%20%5C%5C%0A%20%203.5625%5Ctimes%2010%5E%7B-15%7D%20%26%205.3845%5Ctimes%2010%5E%7B-7%7D%20%26%207.9174%5Ctimes%2010%5E%7B-5%7D%20%26%200.0023%20%26%200.1780%20%26%201.0000%20%26%200.8373%20%26%200.1744%20%5C%5C%0A%20%202.3081%5Ctimes%2010%5E%7B-17%7D%20%26%201.8335%5Ctimes%2010%5E%7B-18%7D%20%26%204.9720%5Ctimes%2010%5E%7B-6%7D%20%26%200.0002%20%26%200.0492%20%26%200.8373%20%26%201.0000%20%26%200.4447%20%5C%5C%0A%20%201.4867%5Ctimes%2010%5E%7B-22%7D%20%26%204.0867%5Ctimes%2010%5E%7B-12%7D%20%26%204.0966%5Ctimes%2010%5E%7B-9%7D%20%26%205.8046%5Ctimes%2010%5E%7B-7%7D%20%26%200.0010%26%200.1744%20%26%200.4447%20%26%201.0000%0A%5Cend%7Bpmatrix%7D%20%0A "
K=
\begin{pmatrix}
  1.0000 & 0.0208 & 0.0007 & 1.8635 \times 10^{-5} & 2.4219\times 10^{-9} & 3.5625\times 10^{-15} & 2.3081\times 10^{-17} & 1.4867\times 10^{-22} \\
  0.0208 & 1.0000 & 0.5902 & 0.1699 & 0.0021 & 5.3845\times 10^{-7} & 1.8335\times 10^{-8} & 4.0867\times 10^{-12} \\
  0.0007 & 0.5902 & 1.0000 & 0.6933 & 0.0453 & 7.9174\times 10^{-5} & 4.9720\times 10^{-6} & 4.0966\times 10^{-9} \\
  1.8635\times 10^{-5} & 0.1699 & 0.6933 & 1.0000 & 0.2640 & 0.0023 & 0.0002 & 5.8046\times 10^{-7} \\
  2.4219\times 10^{-9} & 0.0021 & 0.0453 & 0.2640 & 1.0000 & 0.1780 & 0.0492 & 0.0010 \\
  3.5625\times 10^{-15} & 5.3845\times 10^{-7} & 7.9174\times 10^{-5} & 0.0023 & 0.1780 & 1.0000 & 0.8373 & 0.1744 \\
  2.3081\times 10^{-17} & 1.8335\times 10^{-18} & 4.9720\times 10^{-6} & 0.0002 & 0.0492 & 0.8373 & 1.0000 & 0.4447 \\
  1.4867\times 10^{-22} & 4.0867\times 10^{-12} & 4.0966\times 10^{-9} & 5.8046\times 10^{-7} & 0.0010& 0.1744 & 0.4447 & 1.0000
\end{pmatrix} 
")

<br>

Here we will plot a coloured gradient to give a visual representation of
the correlations between each of the random variables. This is
beneficial when the matrix is large, as there are too many values to
reliably interpolate the behaviors.

<center>

<img src="tools/fig6.jpeg" style="width:40.0%;height:20.0%" />

</center>

Using the covariance function to construct covariance matrices from
combinations of the observed data and data we want to predict, we can
calculate the posterior distribution. The posterior mean vector and
covariance matrix are given (respectively) by:

<center>

![
\\mu\_\* = K^T\_\*K^{-1}y \\\\
\\Sigma\_\* = K\_{\*\*} - K^T\_\*K^{-1}K\_\*
](https://latex.codecogs.com/svg.image?%0A%5Cmu_%2A%20%3D%20K%5ET_%2AK%5E%7B-1%7Dy%20%5C%5C%0A%5CSigma_%2A%20%3D%20K_%7B%2A%2A%7D%20-%20K%5ET_%2AK%5E%7B-1%7DK_%2A%0A "
\mu_* = K^T_*K^{-1}y \\
\Sigma_* = K_{**} - K^T_*K^{-1}K_*
")

</center>

where ![y](https://latex.codecogs.com/svg.image?y "y") are the outputs
from the observed data, ![K](https://latex.codecogs.com/svg.image?K "K")
is the kernel for the observed data,
![K\_\*](https://latex.codecogs.com/svg.image?K_%2A "K_*") is the kernel
for the observed data against the data we want to predict and
![K\_{\*\*}](https://latex.codecogs.com/svg.image?K_%7B%2A%2A%7D "K_{**}")
is the kernel for the data we want to predict.

<br>

``` r
posterior <- function(xvals.s, x.train, y.train, sigma = sigma, l = l, sigma.y = 1e-8) {
  n <- nrow(x.train); m <- nrow(xvals.s)
  k <- kernel(x.train, x.train, sigma = sigma, l = l) + sigma.y ^ 2 * diag(n)
  k.s <- kernel(x.train, xvals.s, sigma = sigma, l = l)
  k.ss <- kernel(xvals.s, xvals.s, sigma = sigma, l = l) + 1e-8 ^ 2 * diag(m)
  k.inv <- solve(k)
  y.train <- data.matrix(y.train)
  colnames(y.train) <- NULL
  mu <- t(k.s) %*% k.inv %*% y.train
  cov <-  k.ss - t(k.s) %*% k.inv %*% k.s
  
  return(list(mu, cov))
}
```

<br>

Using the predictive distribution, we can plot the Gaussian process by
choosing the data we want to predict to be a large sample of points of
the domain:

<center>

<img src="tools/fig7.jpeg" style="width:40.0%" />

</center>

## 2-Dimensional Example

In this example we will be using the Branin function which has the form:
![a(x_2 - bx_2^2+cx_1-r)^2+s(1-t)\\cos(x_1)+s](https://latex.codecogs.com/svg.image?a%28x_2%20-%20bx_2%5E2%2Bcx_1-r%29%5E2%2Bs%281-t%29%5Ccos%28x_1%29%2Bs "a(x_2 - bx_2^2+cx_1-r)^2+s(1-t)\cos(x_1)+s").
We will be using the parameters
![a=1](https://latex.codecogs.com/svg.image?a%3D1 "a=1"),
![b=\\frac{5.1}{4 \\pi ^ 2}](https://latex.codecogs.com/svg.image?b%3D%5Cfrac%7B5.1%7D%7B4%20%5Cpi%20%5E%202%7D "b=\frac{5.1}{4 \pi ^ 2}"),
![c=\\frac{5}{\\pi}](https://latex.codecogs.com/svg.image?c%3D%5Cfrac%7B5%7D%7B%5Cpi%7D "c=\frac{5}{\pi}"),
![r=6](https://latex.codecogs.com/svg.image?r%3D6 "r=6"),
![s=10](https://latex.codecogs.com/svg.image?s%3D10 "s=10") and
![t=\\frac{1}{8\\pi}](https://latex.codecogs.com/svg.image?t%3D%5Cfrac%7B1%7D%7B8%5Cpi%7D "t=\frac{1}{8\pi}").
The range of values
![x_1](https://latex.codecogs.com/svg.image?x_1 "x_1") and
![x_2](https://latex.codecogs.com/svg.image?x_2 "x_2") can take are
![x_1 \\in \[-5,10\]](https://latex.codecogs.com/svg.image?x_1%20%5Cin%20%5B-5%2C10%5D "x_1 \in [-5,10]")
and
![x_2 \\in \[0,15\]](https://latex.codecogs.com/svg.image?x_2%20%5Cin%20%5B0%2C15%5D "x_2 \in [0,15]").
This example will use 40 random samples as training data.

Since the covariance matrix has dimensions 40x40, it is difficult to
understand the behavior by looking at the correlation coefficients. So
we will plot a coloured gradient for the correlations. We will be using
the same SE function to calculate the covariance coefficients. Plotting
the covariance gradient:

<center>

<img src="tools/fig8.jpeg" style="width:40.0%" />

</center>

Using the same matrix multiplication as for the one-dimensional case, we
can calculate the predictive distribution for this Gaussian process.
Using this we can plot the GP:

``` r
library(rgl)
plotgp_2D <- function(x1, x2, x.train, y.train, sigma = 1, l = 1, xlimit = c(), ylimit = c(), zlimit = c()) {
  xvals <- expand.grid(x1, x2)
  mu <- posterior(xvals, x.train, y.train, sigma = sigma, l = l)[[1]]
  
  zlim <- range(mu)
  zlen <- zlim[2] - zlim[1] + 1
  colorlut <- hcl.colors(zlen, palette = "Blues")
  col <- colorlut[ mu - zlim[1] + 1 ]
  persp3d(x=x1, y=x2, z=mu, color=col, zlab = "y(x1, x2)")
}

x1 <- seq(from = -5, to = 10, by = 0.2)
x2 <- seq(from = 0, to = 15, by = 0.2)
plotgp_2D(x1, x2, x.train, y.train, sigma = 1, l = 2)
```

<center>

<img src="tools/fig9.png" style="width:40.0%" />

</center>

## Higher dimensional Examples

In this section we have some example functions for higher dimensional
Gaussian Process regression.

### Borehole function (8-dimensional)

The borehole function models water flowing through a borehole. The
equation has the form:

<center>

![
f(x)=\\frac{2\\pi T_u(H_u-H_l)}{ln\\left(\\frac{r}{r_w}\\right)\\left(1+\\frac{2LT-u}{ln\\left(\\frac{r}{r_w}\\right)r_w^2K_w} + \\frac{T_u}{T_l} \\right)} .
](https://latex.codecogs.com/svg.image?%0Af%28x%29%3D%5Cfrac%7B2%5Cpi%20T_u%28H_u-H_l%29%7D%7Bln%5Cleft%28%5Cfrac%7Br%7D%7Br_w%7D%5Cright%29%5Cleft%281%2B%5Cfrac%7B2LT-u%7D%7Bln%5Cleft%28%5Cfrac%7Br%7D%7Br_w%7D%5Cright%29r_w%5E2K_w%7D%20%2B%20%5Cfrac%7BT_u%7D%7BT_l%7D%20%5Cright%29%7D%20.%0A "
f(x)=\frac{2\pi T_u(H_u-H_l)}{ln\left(\frac{r}{r_w}\right)\left(1+\frac{2LT-u}{ln\left(\frac{r}{r_w}\right)r_w^2K_w} + \frac{T_u}{T_l} \right)} .
")

</center>

### OTL Circuit Function (6-dimensional)

The OTL Circuit function models an output transformerless push-pull
circuit. The equation has the form:

<center>

![
V_m(x)=\\frac{(V\_{b1} + 0.74)\\beta (R\_{x2}+9)}{\\beta(R\_{c2}+9)+R_f} + \\frac{11.35R_f}{\\beta (R\_{c2}+9)+R_f} + \\frac{0.74R_f\\beta(R\_{c2}+9)}{(\\beta(R\_{c2}+9)+R_f)R\_{c1}}, \\\\ \\text{ where }
V\_{b1}=\\frac{12R\_{b2}}{R\_{b1}+R\_{b2}}.
](https://latex.codecogs.com/svg.image?%0AV_m%28x%29%3D%5Cfrac%7B%28V_%7Bb1%7D%20%2B%200.74%29%5Cbeta%20%28R_%7Bx2%7D%2B9%29%7D%7B%5Cbeta%28R_%7Bc2%7D%2B9%29%2BR_f%7D%20%2B%20%5Cfrac%7B11.35R_f%7D%7B%5Cbeta%20%28R_%7Bc2%7D%2B9%29%2BR_f%7D%20%2B%20%5Cfrac%7B0.74R_f%5Cbeta%28R_%7Bc2%7D%2B9%29%7D%7B%28%5Cbeta%28R_%7Bc2%7D%2B9%29%2BR_f%29R_%7Bc1%7D%7D%2C%20%5C%5C%20%5Ctext%7B%20where%20%7D%0AV_%7Bb1%7D%3D%5Cfrac%7B12R_%7Bb2%7D%7D%7BR_%7Bb1%7D%2BR_%7Bb2%7D%7D.%0A "
V_m(x)=\frac{(V_{b1} + 0.74)\beta (R_{x2}+9)}{\beta(R_{c2}+9)+R_f} + \frac{11.35R_f}{\beta (R_{c2}+9)+R_f} + \frac{0.74R_f\beta(R_{c2}+9)}{(\beta(R_{c2}+9)+R_f)R_{c1}}, \\ \text{ where }
V_{b1}=\frac{12R_{b2}}{R_{b1}+R_{b2}}.
")

</center>

### Piston Simulation Function (7-dimensional)

The piston simulation function models the circular motion of a piston
within a cylinder. The equation has the form:

<center>

![
C(x) = 2 \\pi \\sqrt{\\frac{M}{k+S^2\\frac{P_0V_0}{T_0}\\frac{T_a}{V^2}}},\\\\ \\text{ where }
V = \\frac{S}{2k}\\left( \\sqrt{A^2 + 4k \\frac{P_0V_9}{T_0}T_a} - A  \\right), \\\\
A=P_0S + 19.62M - \\frac{kV_0}{S}.
](https://latex.codecogs.com/svg.image?%0AC%28x%29%20%3D%202%20%5Cpi%20%5Csqrt%7B%5Cfrac%7BM%7D%7Bk%2BS%5E2%5Cfrac%7BP_0V_0%7D%7BT_0%7D%5Cfrac%7BT_a%7D%7BV%5E2%7D%7D%7D%2C%5C%5C%20%5Ctext%7B%20where%20%7D%0AV%20%3D%20%5Cfrac%7BS%7D%7B2k%7D%5Cleft%28%20%5Csqrt%7BA%5E2%20%2B%204k%20%5Cfrac%7BP_0V_9%7D%7BT_0%7DT_a%7D%20-%20A%20%20%5Cright%29%2C%20%5C%5C%0AA%3DP_0S%20%2B%2019.62M%20-%20%5Cfrac%7BkV_0%7D%7BS%7D.%0A "
C(x) = 2 \pi \sqrt{\frac{M}{k+S^2\frac{P_0V_0}{T_0}\frac{T_a}{V^2}}},\\ \text{ where }
V = \frac{S}{2k}\left( \sqrt{A^2 + 4k \frac{P_0V_9}{T_0}T_a} - A  \right), \\
A=P_0S + 19.62M - \frac{kV_0}{S}.
")

</center>

### Robot Arm Function (8-dimensional)

The robot arm function models the position of a 4 segment robotic arm.
The equation has the form:

<center>

![ 
f(x) = (u^2 + v^2)^{0.5}, \\\\ \\text{ where } 
u= \\sum^4\_{i=1}L_i\\cos\\left(\\sum^i\_{j=1} \\theta_j\\right), \\\\
v= \\sum^4\_{i=1}L_i\\sin\\left(\\sum^i\_{j=1} \\theta_j\\right).
](https://latex.codecogs.com/svg.image?%20%0Af%28x%29%20%3D%20%28u%5E2%20%2B%20v%5E2%29%5E%7B0.5%7D%2C%20%5C%5C%20%5Ctext%7B%20where%20%7D%20%0Au%3D%20%5Csum%5E4_%7Bi%3D1%7DL_i%5Ccos%5Cleft%28%5Csum%5Ei_%7Bj%3D1%7D%20%5Ctheta_j%5Cright%29%2C%20%5C%5C%0Av%3D%20%5Csum%5E4_%7Bi%3D1%7DL_i%5Csin%5Cleft%28%5Csum%5Ei_%7Bj%3D1%7D%20%5Ctheta_j%5Cright%29.%0A " 
f(x) = (u^2 + v^2)^{0.5}, \\ \text{ where } 
u= \sum^4_{i=1}L_i\cos\left(\sum^i_{j=1} \theta_j\right), \\
v= \sum^4_{i=1}L_i\sin\left(\sum^i_{j=1} \theta_j\right).
")

</center>

## AvGPR

AvGPR (Average Gaussian Process Regression) is an R package that uses
weighted model averaging to calculate an average Gaussian Process
regression model. It does this by emulating 5 different GPR packages
then through cross validation, it calculates a “goodness of fit”
parameter which is used to find the weight for each model. This is an
advantageous method of regression as it gains the benefits associated
with model averaging. The measure of fit statistic that is used in this
package is:

<center>

![
\\sum^n\_{i=1}\[(\\sigma_i - \\epsilon_i) ^ 2 + \\epsilon_i^2\],
](https://latex.codecogs.com/svg.image?%0A%5Csum%5En_%7Bi%3D1%7D%5B%28%5Csigma_i%20-%20%5Cepsilon_i%29%20%5E%202%20%2B%20%5Cepsilon_i%5E2%5D%2C%0A "
\sum^n_{i=1}[(\sigma_i - \epsilon_i) ^ 2 + \epsilon_i^2],
")

</center>

where ![i](https://latex.codecogs.com/svg.image?i "i") denotes the
predicted point,
![\\sigma_i](https://latex.codecogs.com/svg.image?%5Csigma_i "\sigma_i")
is the variance of i, and
![\\epsilon_i](https://latex.codecogs.com/svg.image?%5Cepsilon_i "\epsilon_i")
is the residual. Through analysis and experimentation, this statistic
has shown to provide a better representation of the fit than RSS and
normalised RSS.

Here is an example plot from an AvGPR regression:

<center>

<img src="tools/fig10.jpeg" style="width:40.0%" />

</center>

<https://github.com/benjmuller/AvGPR>

<img src="SheffieldLogo.jpg" width=200/>
