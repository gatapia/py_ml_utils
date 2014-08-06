
import inspect, warnings, sklearn, psutil, numpy, re
import numpy as np

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
      if diff > 1: raise Exception('After[%s] Processes increased by: %s' % \
          (full_name, diff))

      if cls: classifiers.append(cls)
  return classifiers

def test_all_classifiers(classifiers):
  boston_data = datasets.load_boston()
  X = boston_data['data']
  y = boston_data['target']
  best = (0, None)
  for classifier in classifiers:
    print 'testing classifier: ', classifier
    try:
      scores = sklearn.cross_validation.cross_val_score(classifier(), X, y)
      score = numpy.mean(scores)
      if (score > best[0]): best = (score, classifier)
      print 'classifier:', classifier, 'score:', score
    except:
      print 'error testing classifier:', classifier
  print 'Best classifier is: ', best[1], 'Score: ', best[0]

def parse_classifier_meta(classifier):
  doc = classifier.__doc__
  lines = filter(None, [s.strip() for s in re.sub('-+', '\n', doc).split('\n')])
  args = []
  started = False
  curr_arg = None
  for l in lines:
    if not started and l == 'Parameters': started = True
    elif started and l == 'See Also': break
    elif started:
      if ':' in l: 
        name_type = map(lambda s: s.strip(), l.split(':'))
        curr_arg = { 'name': name_type[0], 'type': name_type[1], 'description': '' }
        args.append(curr_arg)
      elif l:
        if not curr_arg: print 'invalid line [%s] doc: %s' % (l, doc)
        curr_arg['description'] += l
  return {'classifier': classifier, 'args': args }

def parse_float_type(t):
  q = '.* ([0-9.]+) \< .* \< ([0-9.]+)'
  r = re.search(q, t)
  if r: return np.linspace(float(r.group(1)), float(r.group(2)), 100)
  return np.linspace(-100, 100, 100)

def parse_range_type(t):
  matches = re.findall('([A-z0-9\.]+)', t, re.DOTALL)
  matches = [None if m == 'None' else m for m in matches]
  matches = [True if m == 'True' else m for m in matches]
  matches = [False if m == 'False' else m for m in matches]
  l = list(set(matches))
  l.sort()
  return l

def parse_string_type(t, d):
  d = d.replace('\n', ' ')
  matches = re.findall('[\'"]([A-z0-9]+)[\'"]', t + ' ' + d, re.DOTALL)    
  l = list(set(matches))
  l.sort()
  return l

def get_val_for_type(name, t, desc):
  ignores_names = ['base_estimator', 'class_weight']

  if name in ignores_names or not(t) or 'array' in t: return None
  if t.startswith('class') or t.startswith('ref') \
    or t.startswith('meth') or t.startswith('callable'): return None
  if name.startswith('_') or name.endswith('_'): return None

  if (t.startswith('bool') or t.startswith('Bool')): return [True, False]
  if (t.startswith('float')): return parse_float_type(t)  
  if (t.startswith('int')): return range(0, 2000, 10)
  if (t.startswith('str')): return parse_string_type(t, desc)
  if (t.startswith('{')): return parse_range_type(t)  
  if (t.startswith('double')): return np.linspace(-100, 100, 100)
  return None

def test_classifier_with_arg_customisation(meta):
  clf = meta['classifier']
  gs_args = {}
  for a in meta['args']:    
    vals = get_val_for_type(a['name'], a['type'], a['description'])
    if vals != None: gs_args[a['name']] = vals
  if (1==2 and len(gs_args) > 0):
    boston_data = datasets.load_boston()
    X = boston_data['data']
    y = boston_data['target']
    do_gs(clf(), X, y, gs_args)



if __name__ == '__main__':
  # classifiers = get_classifiers(sklearn)
  ## test_all_classifiers(classifiers)
  # metas = [parse_classifier_meta(clf) for clf in classifiers]
  ignore = [test_classifier_with_arg_customisation(m) for m in metas]
