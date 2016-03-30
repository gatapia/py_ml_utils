from keras.models import *
from keras.optimizers import *
import time

def fit_evaluate(nn, X, y, 
    final_hodlout_split=.05, validation_split=.075, 
    epochs=3, batch_size=128, 
    loss='binary_crossentropy', optimizer='rmsprop',
    compile_args={}, fit_args={}):
  np.random.seed(0)
  X_train, y_train, X_test, y_test = splitter(X, y, 1. - final_hodlout_split)  
  if isinstance(nn, Graph):
    g_loss = {}
    for n in nn.outputs.keys(): g_loss[n] = loss
    loss = g_loss

    if type(X_train) is np.ndarray: X_train, X_test = [X_train], [X_test]
    if type(X_train) is not list: raise Exception('for Graph models X and y must be lists or single numpy array')
    X_train_d, X_test_d = {}, {}    
    for idx, inp in enumerate(nn.inputs.keys()):
      X_train_d[inp], X_test_d[inp] = X_train[idx], X_test[idx]

    if type(y_train) is np.ndarray: y_train, y_test = [y_train], [y_test]
    if type(y_train) is not list: raise Exception('for Graph models X and y must be lists or single numpy array')

    for idx, outp in enumerate(nn.outputs.keys()):
      X_train_d[outp], X_test_d[outp] = y_train[idx], y_test[idx]

    X_train, X_test = X_train_d, X_test_d
    del X_train_d, X_test_d

  compile_time = time.time()
  compile_args['loss'] = loss
  compile_args['optimizer'] = optimizer
  nn.compile(**compile_args)  
  compile_time = time.time() - compile_time

  fit_time = time.time()
  if isinstance(nn, Graph): 
    history = nn.fit(X_train, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split, **fit_args)
  else: 
    history = nn.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split, show_accuracy=True, **fit_args)
  fit_time = time.time() - fit_time

  eval_time = time.time()
  if final_hodlout_split > 0.:
    holdout_score, accuracy = (nn.evaluate(X_test, batch_size=batch_size), 0) if isinstance(nn, Graph) else \
        nn.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
  else: holdout_score = 0
  eval_time = time.time() - eval_time

  try:
    best = np.min(history.history['val_loss'])
    best_epoch = np.argmin(history.history['val_loss'])
    print '\nEvaluation Summary:\nCompile Time: %.1fs\nTrain Time: %.1fs\nHoldout Eval Time: %.1fs\nBest Epoch: %d\nBest Validation Score: %.5f\nHoldout Score: %.5f\nHoldout Accuracy: %.5f' % (compile_time, fit_time, eval_time, best_epoch, best, holdout_score, accuracy)
  except Exception as e:
    print 'Error getting best: ', e
  return (holdout_score, history)


def splitter(X, y, train_split):
  if not type(X) is dict and hasattr(X, 'values'): X = X.values
  if hasattr(y, 'values'): y = y.values
  
  def _len(arr):
    if type(arr) is dict: return len(arr.itervalues().next())
    elif type(arr) is list: return len(arr[0])
    else: return len(arr)
  
  exp_length = _len(y)
  train_split = int(exp_length * train_split)
  def _spl(arr):
    if type(arr) is dict:    
      arr_train, arr_test = {}, {}
      for key, arr in arr.iteritems(): 
        if len(arr) != exp_length: raise Exception('X, y, length mismatch')
        arr_train[key], arr_test[key] = arr[:train_split], arr[train_split:]
        return arr_train, arr_test
    elif type(arr) is list: 
      if len([x for x in arr if len(x) != exp_length]) > 0: raise Exception('X, y, length mismatch')
      return [x[:train_split] for x in arr], [x[train_split:] for x in arr]
    else: 
      if len(arr) != exp_length: raise Exception('X, y, length mismatch')
      return arr[:train_split], arr[train_split:]

  X_train, X_test = _spl(X)
  y_train, y_test = _spl(y)
  return (X_train, y_train, X_test, y_test)
