from keras.models import *

def fit_evaluate(nn, X, y, 
    final_hodlout_split=.05, validation_split=.075, 
    epochs=3, batch_size=128, 
    loss='binary_crossentropy', optimizer='rmsprop'):
  np.random.seed(0)
  X_train, y_train, X_test, y_test = splitter(X, y, 1. - final_hodlout_split)
  is_graph = type(nn) is Graph
  if is_graph:
    g_loss = {}
    for n in nn.outputs.keys(): g_loss[n] = loss
    loss = g_loss

    X_train[nn.outputs.keys()[0]] = y_train
    X_test[nn.outputs.keys()[0]] = y_test

  nn.compile(loss=loss, optimizer=optimizer)  
  if is_graph: history = nn.fit(X_train, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split)
  else: history = nn.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size, validation_split=validation_split)

  if final_hodlout_split > 0.:
    score = nn.evaluate(X_test, batch_size=batch_size) if is_graph else \
        nn.evaluate(X_test, y_test, batch_size=batch_size)
  else: score = 0
  return (score, history)

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
