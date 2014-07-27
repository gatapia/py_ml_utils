
import inspect, warnings, sklearn, psutil
from sklearn import cluster, covariance, \
  decomposition, ensemble, feature_extraction, feature_selection, \
  gaussian_process, isotonic, kernel_approximation, lda, learning_curve, \
  linear_model, manifold, mixture, multiclass, naive_bayes, \
  neighbors, neural_network, cross_decomposition, preprocessing, \
  qda, random_projection, semi_supervised, svm, tree, datasets

def get_python_processes():
  def is_python_process(p):
    try: return 'python' in p.name
    except: return false
  return len([p for p in psutil.get_process_list() if is_python_process])

def get_classifiers(module):
  ignores = ['MemmapingPool', 'PicklingPool']
  classifiers = []
  X, y = sklearn.datasets.make_regression(20, 5)
  for name, cls in inspect.getmembers(module):                
    if name in ignores: continue

    if inspect.ismodule(cls):       
      if cls.__name__.startswith('_') or \
          cls.__name__.endswith('_') or \
          not cls.__name__.startswith('sklearn'): continue
      classifiers += get_classifiers(cls)      

    if inspect.isclass(cls):             
      if '_' in name or name[0].islower(): continue
      if cls.__module__.startswith('_') or \
          cls.__module__.endswith('_') or \
          not cls.__module__.startswith('sklearn'): continue
      
      pre_processes_length = get_python_processes()
      full_name = cls.__module__ + '.' + cls.__name__

      try: cls().fit(X, y).predict(X)
      except: cls = None

      post_processes_length = get_python_processes()
      diff = post_processes_length - pre_processes_length
      if diff > 1: raise Exception('After[%s] Processes increased by: %s' % (full_name, diff))

      if cls: classifiers.append(cls)
  return classifiers

def test_all_classifiers():
  boston_data = datasets.load_boston()
  X = boston_data['data']
  y = boston_data['target']
  for classifier in get_classifiers(sklearn):
    print 'testing classifier: ', classifier
    score = sklearn.cross_validation.cross_val_score(classifier(), X, y)
    print 'classifier:', classifier, 'score:', score

test_all_classifiers()