from keras.models import *
from keras.optimizers import *
from time import time

def fit_evaluate(nn, X, y, 
    final_hodlout_split=.05, validation_split=.075, 
    epochs=3, batch_size=128, 
    loss='binary_crossentropy', optimizer='rmsprop',
    fit_args={}):
  np.random.seed(0)
  X_train, y_train, X_test, y_test = splitter(X, y, 1. - final_hodlout_split)  
  if isinstance(nn, Graph):
    g_loss = {}
    for n in nn.outputs.keys(): g_loss[n] = loss
    loss = g_loss

    X_train[nn.outputs.keys()[0]] = y_train
    X_test[nn.outputs.keys()[0]] = y_test

  compile_time = time()
  nn.compile(loss=loss, optimizer=optimizer)  
  compile_time = time() - compile_time

  fit_time = time()
  if isinstance(nn, Graph): history = nn.fit(X_train, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split, **fit_args)
  else: history = nn.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split, **fit_args)
  fit_time = time() - fit_time

  eval_time = time()
  if final_hodlout_split > 0.:
    holdout_score = nn.evaluate(X_test, batch_size=batch_size) if isinstance(nn, Graph) else \
        nn.evaluate(X_test, y_test, batch_size=batch_size)
  else: holdout_score = 0
  eval_time = time() - eval_time

  best = np.min(history.history['val_loss'])
  best_epoch = np.argmin(history.history['val_loss'])
  print '\nEvaluation Summary:\nCompile Time: %.1fs\nTrain Time: %.1fs\nHoldout Eval Time: %.1fs\nBest Epoch: %d\nBest Validation Score: %.5f\nHoldout Score: %.5f' % (compile_time, fit_time, eval_time, best_epoch, best, holdout_score)
  return (holdout_score, history)

def splitter(X, y, train_split):
  if not type(X) is dict and hasattr(X, 'values'): X = X.values
  if hasattr(y, 'values'): y = y.values

  split = int(len(y) * train_split)
  if type(X) is dict:    
    X_train, X_test = {}, {}
    for key, x in X.iteritems(): 
      X_train[key], X_test[key] = x[:split], x[split:]
  elif type(X) is list: 
    X_train, X_test = [x[:split] for x in X], [x[split:] for x in X]
  else: 
    X_train, X_test = X[:split], X[split:]

  y_train, y_test = y[:split], y[split:]
  return (X_train, y_train, X_test, y_test)
