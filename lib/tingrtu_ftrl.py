from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from pickle import dump, load
import gzip
from sys import argv, stdout, stderr
import random
import argparse


##############################################################################
# class, function, generator definitions #####################################
##############################################################################

class ftrl_proximal(object):
  ''' Our main algorithm: Follow the regularized leader - proximal

    In short,
    this is an adaptive-learning-rate sparse logistic-regression with
    efficient L1-L2-regularization
  '''

  def __init__(self, alpha, beta, L1, L2, D, 
         interaction=False, dropout = 1.0, sparse = False):
    # parameters
    self.alpha = alpha
    self.beta = beta
    self.L1 = L1
    self.L2 = L2

    # feature related parameters
    self.D = D
    self.interaction = interaction
    self.dropout = dropout    

    # model
    # n: squared sum of past gradients
    # z: weights
    # w: lazy weights
    self.n = [0.] * D
    self.z = [0.] * D
    
    if sparse:
      self.w = {}
    else:
      self.w = [0.] * D  # use this for execution speed up

  def _indices(self, x):
    ''' A helper generator that yields the indices in x

      The purpose of this generator is to make the following
      code a bit cleaner when doing feature interaction.
    '''

    for i in x:
      yield i

    if self.interaction:
      D = self.D
      L = len(x)
      for i in xrange(1, L):  # skip bias term, so we start at 1
        for j in xrange(i+1, L):
          # one-hot encode interactions with hash trick
          yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

  def predict(self, x, dropped = None):
    ''' Get probability estimation on x

      INPUT:
        x: features

      OUTPUT:
        probability of p(y = 1 | x; w)
    '''
    # params
    dropout = self.dropout

    # model
    w = self.w
    # wTx is the inner product of w and x
    wTx = 0.
    for j, i in enumerate(self._indices(x)):
      
      if dropped != None and dropped[j]:
        continue
       
      wTx += w[i]
    
    if dropped != None: wTx /= dropout 

    # bounded sigmoid function, this is the probability estimation
    return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

  def update(self, x, y):
    ''' Update model using x, p, y

      INPUT:
        x: feature, a list of indices
        p: probability prediction of our model
        y: answer

      MODIFIES:
        self.n: increase by squared gradient
        self.z: weights
    '''

    # parameters
    alpha = self.alpha
    beta = self.beta
    L1 = self.L1
    L2 = self.L2

    # model
    n = self.n
    z = self.z
    w = self.w  # no need to change this, it won't gain anything
    dropout = self.dropout

    ind = [ i for i in self._indices(x)]
    
    if dropout == 1:
      dropped = None
    else:
      dropped = [random.random() > dropout for i in xrange(0,len(ind))]
    
    p = self.predict(x, dropped)

    # gradient under logloss
    g = p - y

    # update z and n
    for j, i in enumerate(ind):

      # implement dropout as overfitting prevention
      if dropped != None and dropped[j]: continue

      sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
      z[i] += g - sigma * w[i]
      n[i] += g * g
      
      sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

      # build w on the fly using z and n, hence the name - lazy weights -
      if sign * z[i] <= L1:
        # w[i] vanishes due to L1 regularization
        w[i] = 0.
      else:
        # apply prediction time L1, L2 regularization to z and get w
        w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)



def logloss(p, y):
  ''' FUNCTION: Bounded logloss

    INPUT:
      p: our prediction
      y: real answer

    OUTPUT:
      logarithmic loss of p given y
  '''

  p = max(min(p, 1. - 10e-15), 10e-15)
  return -log(p) if y == 1. else -log(1. - p)

def data(f_train, D, columns):
  ''' GENERATOR: Apply hash-trick to the original csv row
           and for simplicity, we one-hot-encode everything

    INPUT:
      path: path to training or testing file
      D: the max index that we can hash to

    YIELDS:
      x: a list of hashed and one-hot-encoded 'indices'
         we only need the index since all values are either 0 or 1
      y: y = 1 if positive example else negative
  '''
  positive_counts = 0
  for t, row in enumerate(DictReader(f_train)):    
    y = 0.
    if 'y' in row:
      if row['y'].startswith('1'):
        y = 1.
        positive_counts += 1
      del row['y']
 
    # build x
    x = [0]  # 0 is the index of the bias term
    for key in row:
      if key not in columns: continue
      
      value = row[key]

      # one-hot encode everything with hash trick
      index = abs(hash(key + '_' + value)) % D
      x.append(index)

    yield t, x, y, positive_counts


##############################################################################
# start training #############################################################
##############################################################################


def myargs():
  
  parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                   description = 
""" 
Perform training and prediction based on FTRL Optimal algorithm, with dropout added.
\nUsage is via:
\n
\n\t* Training:
\n
\n\t\tpypy fastd.py train -t <train set> -o <output model> --<various parameters>
\n
\n\t* Predicting:
\n
\n\t\tpypy fastd.py predict --test <test set> -i <input model> -p <output predictions>
\n
""")
  parser.add_argument('action', type=str,
            help='action to perform: train   / predict')
  parser.add_argument('-t', "--train", default = "/dev/stdin")
  parser.add_argument('--test', default = "/dev/stdin")
  parser.add_argument('-p', "--predictions", default = "/dev/stdout")
  parser.add_argument("-o", "--outmodel")
  parser.add_argument("-i", "--inmodel")
  parser.add_argument('--alpha', default = 0.015, type = float)
  parser.add_argument('--beta', default = 2, type = float)
  parser.add_argument('--L1', default = 0, type = float)
  parser.add_argument('--L2', default = 0, type = float)
  parser.add_argument('--dropout', default = 0.8, type = float)
  parser.add_argument('--bits', default = 23, type = int)
  parser.add_argument('--n_epochs', default = 1, type = int)
  parser.add_argument('--holdout', default = 100, type = int)
  parser.add_argument('--seed', default = 0, type = int)
  parser.add_argument("--interactions", action = "store_true")
  parser.add_argument("--sparse", action = "store_true")
  parser.add_argument("-v", '--verbose', default = 3, type = int)
  parser.add_argument("-c", '--columns', default = '', type = str)
  
  args = parser.parse_args()
  if args.verbose > 1:
    for v in vars(args).keys():
      stderr.write("%s => %s\n" % (v, str(vars(args)[v])))
  args.columns = args.columns.split('|;|')
  return args


def write_learner(learner, model_file, args):
   with open(model_file, 'wb') as f: dump((args, learner), f)

def load_learner(model_file):
  with open(model_file, 'rb') as f: (p, learner) = load(f)
  
  return learner
  

def train_learner(train, args):

  if args.verbose > 1:
    stderr.write("Learning from %s\n" % train)

  if train[-3:] == ".gz":
     f_train = gzip.open(train, "rb")
  else:
     f_train = open(train)
      
  start = datetime.now()
   
  D = 2**args.bits
  holdout = args.holdout
  
    # initialize ourselves a learner
  learner = ftrl_proximal(args.alpha, args.beta, 
               args.L1, args.L2, D, 
               interaction = args.interactions,
               dropout = args.dropout,
               sparse = args.sparse)
      
    # start training
  for e in xrange(args.n_epochs):
     loss = 0.
     count = 0
     next_report = 10000
     c = 0
     
     if train != "/dev/stdin": f_train.seek(0,0)

     pc = 0
     for t, x, y, pc in data(f_train, D, args.columns):
       # data is a generator
       #  t: just a instance counter
       #  x: features
       #  y: label
      
       # step 1, get prediction from learner
      
       if t % holdout == 0:
        # step 2-1, calculate holdout validation loss
        #       we do not train with the holdout data so that our
        #       validation loss is an accurate estimation of
        #       the out-of-sample error
        p = learner.predict(x)
        loss += logloss(p, y)
        count += 1
       else:
         # step 2-2, update learner with label information
         learner.update(x, y)
      
       c += 1 
       if args.verbose > 2 and c >= next_report:
         stderr.write(' %s\tencountered: %d/%d\tcurrent logloss: %f\n' % (
           datetime.now(), c, t, loss/count))
         next_report *= 2

     if count != 0:
       stderr.write('Epoch %d finished, %d/%d samples per pass, holdout logloss: %f, elapsed time: %s\n, positives: %d ' % (
          e, c, t, loss/count, str(datetime.now() - start), pc))
     else:
       stderr.write('Epoch %d finished, %d/%d samples per pass, suspicious holdout logloss: %f/%f, elapsed time: %s\n, positives: %d ' % (
          e, c, t, loss, count, str(datetime.now() - start), pc))

  f_train.close()
  return learner
  

def predict_learner(learner, test, predictions_file, args):
  
  if args.verbose > 1:
    stderr.write("Predicting to %s\n" % predictions_file)      
    
  D = learner.D
  predictions = []
  
  if test[-3:] == ".gz": f_test = gzip.open(test, "rb")
  else: f_test = open(test, "r")

  pc = 0
  for t, x, y, pc in data(f_test, D, args.columns):
    predictions.append('%.5f' % learner.predict(x))  
  f_test.close()

  if predictions_file[-3:] == ".gz": f = gzip.open(predictions_file, "wb")
  else: f = open(predictions_file, "wb")
  
  f.write('\n'.join(predictions))  
  f.close()


def main_fast_dropout():
  
  args = myargs()
  
  learner = None
    
  if args.action in ["train", "train_predict"]:
    random.seed(args.seed)
    learner = train_learner(args.train, args)
    if args.outmodel != None:
      write_learner(learner, args.outmodel, args)
      
  if args.action in ["predict", "train_predict"]:
    random.seed(args.seed)
    if learner == None:
      learner = load_learner(args.inmodel)
    predict_learner(learner, args.test, args.predictions, args)
    
  return learner    
    
if __name__ == "__main__":
  main_fast_dropout()
