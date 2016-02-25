from keras.models import *

def fit_evaluate(nn, X, y, train_split=.95, epochs=3, batch_size=128, 
    loss='binary_crossentropy', optimizer='rmsprop'):
  np.random.seed(0)
  X_train, y_train, X_test, y_test = splitter(X, y, train_split)

  nn.compile(loss=loss, optimizer=optimizer)
  nn.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size)
  score = nn.evaluate(X_test, y_test, batch_size=batch_size)
  print 'score:', score
  return score

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
