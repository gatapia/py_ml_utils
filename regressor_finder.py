'''
Regression Finder
'''
regressors = load_regressors_meta_()

def find_optimal_regressor(X, y):
  scores = map(get_regressor_score_, regressors)
  optimal = max(scores, key=lambda s: s.score)
  print '\n\n\n\nOptimal Regressior: %s Score: {0:.3f} Args: %s\n\n\n\n' % \
    (optimal.name, optimal.score, optimal.args)
  return optimal

def load_regressors_meta_():


def get_regressor_score_(meta, X, y):
  name = meta.name
  args = parse_args_(meta.args)
  clf = get_class_(name)
  gs = do_gs(clf, np.copy(X), y, args)
  gs.best_params_
  gs.best_score_
  return {'name': name, 'score': gs.best_score_, 'args': gs.best_params_}

def parse_args_(meta_args):
  args = {}
  if not(meta_args): return args
  for ma in meta_args: args[ma.name] = parse_arg_range_(ma.value)
  return args

def parse_arg_range_(meta_value_str):


def get_class_(name):
  parts = name.split('.')
  module = ".".join(parts[:-1])
  m = __import__( module )
  for comp in parts[1:]:
      m = getattr(m, comp)            
  return m
