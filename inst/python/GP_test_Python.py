import numpy as np
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import warnings


def gp_GPy(design, response, newdata):
  warnings.filterwarnings("ignore")
  dim = len(design[1])
  kern = GPy.kern.Matern52(dim)
  gp = GPy.models.GPRegression(design, response, kern)
  gp.optimize()
  predicted = gp.predict(Xnew = newdata)
  return(predicted)


def gp_sklearn(design, response, newdata):
  warnings.filterwarnings("ignore")
  kernel = Matern(length_scale=1.0, nu=1.5)
  gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
  gp.fit(design, response)
  predict = gp.predict(newdata, return_std=True)
  return(predict)
