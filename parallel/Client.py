from IPython import parallel
import os, sys
from sklearn import *
# from LibFMClassifier import *
from XGBoostClassifier import *
from VowpalWabbit import *
from misc import *
sys.path.append('lib/elm')
from elm import *

class Client():
  def __init__(self, dataset_names, cache_datasets=False):
    self.dataset_names = dataset_names
    self.cache_datasets = cache_datasets        
    self.results = []

  def run(self):
    engines = self._get_engines()
    classifiers = self.get_all_classifier_combos()
    for dataset in self.dataset_names:
      for clf in self.classifiers:
        self.results += self.engines.map(do_cv, clf, {}, dataset, self.cache_datasets)
    return self.results

  def _get_engines(self):
    rc = parallel.Client()
    view = rc[:]
    return view

  def _get_all_classifier_combos(self):    
    return [
      ensemble.AdaBoostClassifier(),
    
      ensemble.AdaBoostRegressor(),
      ensemble.AdaBoostRegressor(linear_model.LogisticRegression()),
      ensemble.AdaBoostRegressor(linear_model.LinearRegression()),

      ensemble.BaggingClassifier(),
      ensemble.BaggingClassifier(linear_model.LogisticRegression()),
      ensemble.BaggingClassifier(linear_model.LinearRegression()),

      ensemble.BaggingRegressor(),
      ensemble.BaggingRegressor(linear_model.LogisticRegression()),
      ensemble.BaggingRegressor(linear_model.LinearRegression()),

      ensemble.ExtraTreesClassifier(100),
      ensemble.ExtraTreesRegressor(100),
      ensemble.GradientBoostingClassifier(),
      ensemble.GradientBoostingRegressor(),
      ensemble.RandomForestClassifier(100),
      ensemble.RandomTreesEmbedding(100),
      ensemble.RandomForestRegressor(100),

      gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1),
      isotonic.IsotonicRegression(),

      linear_model.ARDRegression(200),
      linear_model.BayesianRidge(),
      linear_model.ElasticNet(),
      linear_model.ElasticNetCV(),
      linear_model.Lars(),
      linear_model.LarsCV(),
      linear_model.Lasso(),
      linear_model.LassoCV(),
      linear_model.LassoLars(),
      linear_model.LassoLarsCV(),
      linear_model.LassoLarsIC(),
      linear_model.LinearRegression(),
      linear_model.LogisticRegression(),
      linear_model.MultiTaskLasso(),
      linear_model.MultiTaskElasticNet(),
      linear_model.MultiTaskLassoCV(),
      linear_model.MultiTaskElasticNetCV(),
      linear_model.OrthogonalMatchingPursuit(),
      linear_model.OrthogonalMatchingPursuitCV(),
      linear_model.PassiveAggressiveClassifier(),
      linear_model.PassiveAggressiveRegressor(),
      linear_model.Perceptron(),
      linear_model.RandomizedLasso(),
      linear_model.RandomizedLogisticRegression(),
      linear_model.RANSACRegressor(),
      linear_model.Ridge(),
      linear_model.RidgeClassifier(),
      linear_model.RidgeClassifierCV(),
      linear_model.RidgeCV(),
      linear_model.SGDClassifier(),
      linear_model.SGDRegressor(),

      naive_bayes.GaussianNB(),
      naive_bayes.MultinomialNB(),
      naive_bayes.BernoulliNB(),

      neighbors.NearestNeighbors(3),
      neighbors.NearestNeighbors(10),
      neighbors.NearestNeighbors(50),
      neighbors.NearestNeighbors(100),
      neighbors.KNeighborsClassifier(3),
      neighbors.KNeighborsClassifier(10),
      neighbors.KNeighborsClassifier(50),
      neighbors.KNeighborsClassifier(100),
      neighbors.RadiusNeighborsClassifier(),
      neighbors.KNeighborsRegressor(3),
      neighbors.KNeighborsRegressor(10),
      neighbors.KNeighborsRegressor(50),
      neighbors.KNeighborsRegressor(100),
      neighbors.RadiusNeighborsRegressor(),
      neighbors.NearestCentroid(),
      neighbors.KernelDensity(),

      neural_network.BernoulliRBM(),

      semi_supervised.LabelPropagation(),
      semi_supervised.LabelSpreading(),

      svm.SVC(),
      svm.LinearSVC(),
      svm.NuSVC(),
      svm.SVR(),
      svm.NuSVR(),

      tree.DecisionTreeClassifier(),
      tree.DecisionTreeRegressor(),
      tree.ExtraTreeClassifier(),
      tree.ExtraTreeRegressor(),

      # LibFMClassifier(),
      # LibFMClassifier(task='regression'),

      VowpalWabbitClassifier(),
      VowpalWabbitRegressor(),

      XGBoostClassifier(booster='gbtree'),
      XGBoostClassifier(booster='gblinear'),

      GenELMRegressor(),
      GenELMClassifier(),
    ]

