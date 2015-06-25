
from __future__ import absolute_import
import os, sys, subprocess, shlex, tempfile, time, sklearn.base, math
import numpy as np
import pandas as pd
from pandas_extensions import * 

_vw_default_path = 'utils/lib/vw'

class _VW(sklearn.base.BaseEstimator):
  def __init__(self,
         logger=None,
         vw=_vw_default_path,
         moniker='vowpal_wabbit',
         name=None,
         bits=None,
         loss=None,
         passes=10,         
         silent=False,
         l1=None,
         l2=None,
         learning_rate=None,
         quadratic=None,
         audit=None,
         power_t=None,
         adaptive=False,
         working_dir=None,
         decay_learning_rate=None,
         initial_t=None,
         minibatch=None,
         total=None,
         node=None,
         unique_id=None,
         span_server=None,
         bfgs=None,
         oaa=None,
         ect=None,
         csoaa=None,
         wap=None,
         cb=None,         
         cb_type=None,
         old_model=None,
         incremental=False,
         mem=None,
         nn=None,         
         invariant=False,
         normalized=False,
         sgd=False,
         ignore=None,
         columns=None
         ):
    self.logger = logger
    self.vw = vw
    self.moniker = moniker
    self.name = name
    self.bits = bits
    self.loss = loss
    self.passes = passes    
    self.silent = silent
    self.l1 = l1
    self.l2 = l2
    self.learning_rate = learning_rate
    self.quadratic = quadratic
    self.audit = audit
    self.power_t = power_t
    self.adaptive = adaptive
    self.working_dir = working_dir
    self.decay_learning_rate = decay_learning_rate
    self.initial_t = initial_t
    self.minibatch = minibatch
    self.total = total
    self.node = node
    self.unique_id = unique_id
    self.span_server = span_server
    self.bfgs = bfgs
    self.oaa = oaa
    self.ect = ect
    self.csoaa=csoaa
    self.wap = wap
    self.cb = cb
    self.cb_type = cb_type
    self.old_model = old_model
    self.incremental = incremental
    self.mem = mem
    self.nn = nn    
    self.invariant = invariant
    self.normalized = normalized
    self.sgd = sgd
    self.ignore = ignore
    self.columns = columns
    if hasattr(self.columns, 'tolist'): self.columns = self.columns.tolist()


  def fit(self, X, y=None):
    if type(X) is np.ndarray: 
      if self.columns is None: raise Exception('VowpalWabbit requires columns be set')      
      X = pd.DataFrame(X, columns=self.columns)
    if type(X) is pd.DataFrame: 
      X = X.to_vw(y)        
    self.vw_ = VW(
      logger=self.logger,
      vw=self.vw,
      moniker=self.moniker,
      name=self.name,
      bits=self.bits,
      loss=self.loss,
      passes=self.passes,      
      silent=self.silent,
      l1=self.l1,
      l2=self.l2,
      learning_rate=self.learning_rate,
      quadratic=self.quadratic,
      audit=self.audit,
      power_t=self.power_t,
      adaptive=self.adaptive,
      working_dir=self.working_dir,
      decay_learning_rate=self.decay_learning_rate,
      initial_t=self.initial_t,
      minibatch=self.minibatch,
      total=self.total,
      node=self.node,
      unique_id=self.unique_id,
      span_server=self.span_server,
      bfgs=self.bfgs,
      oaa=self.oaa,
      ect=self.ect,
      csoaa=self.csoaa,
      wap=self.wap,
      cb=self.cb,
      cb_type=self.cb_type,
      old_model=self.old_model,
      incremental=self.incremental,
      mem=self.mem,
      nn=self.nn,      
      invariant=self.invariant,
      normalized=self.normalized,
      sgd=self.sgd,
      ignore=self.ignore
    )
    self.vw_.training(X)
    return self

  def predict(self, X):    
    if type(X) is np.ndarray: 
      if self.columns is None: raise Exception('VowpalWabbit requires columns be set')      
      X = pd.DataFrame(X, columns=self.columns)
    if type(X) is pd.DataFrame: X = X.to_vw()    

    self.vw_.predicting(X)
    raw = self.vw_.read_predictions_()
    return np.asarray(list(raw))

  def predict_proba(self, X):    
    if type(X) is np.ndarray: 
      if self.columns is None: raise Exception('VowpalWabbit requires columns be set')      
      X = pd.DataFrame(X, columns=self.columns)
    if type(X) is pd.DataFrame: X = X.to_vw()    

    self.vw_.predicting(X)
    preds = list(self.vw_.read_predictions_())
    def sig(p):
      if p < -100: return 0
      return 1 / (1 + math.exp(-p))
    predictions = np.asarray(map(sig, preds))
    return np.vstack([1 - predictions, predictions]).T

class VowpalWabbitRegressor(sklearn.base.RegressorMixin, _VW):
  pass

class VowpalWabbitClassifier(sklearn.base.ClassifierMixin, _VW):
  pass

class VW:
  def __init__(self,
         logger=None,
         vw=_vw_default_path,
         moniker=None,
         name=None,
         bits=None,
         loss=None,
         passes=None,         
         silent=False,
         l1=None,
         l2=None,
         learning_rate=None,
         quadratic=None,
         cubic=None,
         audit=None,
         power_t=None,
         adaptive=False,
         working_dir=None,
         decay_learning_rate=None,
         initial_t=None,
         lda=None,
         lda_D=None,
         lda_rho=None,
         lda_alpha=None,
         minibatch=None,
         total=None,
         node=None,
         unique_id=None,
         span_server=None,
         bfgs=None,
         oaa=None,
         ect=None,
         csoaa=None,
         wap=None,
         cb=None,
         cb_type=None,
         old_model=None,
         incremental=False,
         mem=None,
         nn=None,
         holdout_off=None,
         no_model=None,         
         invariant=False,
         normalized=False,
         sgd=False,
         ignore=None,
         **kwargs):
    assert moniker and passes
        
    self.node = node
    self.total = total
    self.unique_id = unique_id
    self.span_server = span_server
    if self.node is not None:
      assert self.total is not None
      assert self.unique_id is not None
      assert self.span_server is not None

    if name is None:
      self.handle = '%s' % moniker
    else:
      self.handle = '%s.%s' % (moniker, name)

    if self.node is not None:
      self.handle = "%s.%d" % (self.handle, self.node)

    if old_model is None:
      self.filename = '%s.model' % self.handle
      self.incremental = False
    else:
      self.filename = old_model
      self.incremental = True

    self.name = name
    self.bits = bits
    self.loss = loss
    self.vw = vw
    self.l1 = l1
    self.l2 = l2
    self.learning_rate = learning_rate    
    self.silent = silent
    self.passes = passes
    self.quadratic = quadratic
    self.cubic = cubic
    self.power_t = power_t
    self.adaptive = adaptive
    self.decay_learning_rate = decay_learning_rate
    self.audit = audit
    self.initial_t = initial_t
    self.sgd = sgd
    self.lda = lda
    self.lda_D = lda_D
    self.lda_rho = lda_rho
    self.lda_alpha = lda_alpha
    self.minibatch = minibatch
    self.oaa = oaa
    self.ect=ect
    self.csoaa=csoaa
    self.wap=wap
    self.cb=cb
    self.cb_type=cb_type
    self.bfgs = bfgs
    self.mem = mem
    self.nn = nn
    self.holdout_off = holdout_off
    self.no_model = no_model    
    self.invariant = invariant
    self.normalized = normalized
    self.sgd = sgd
    self.ignore = ignore
    
    self.tmpdir = 'tmpfiles'
    if not os.path.isdir(self.tmpdir): os.mkdir(self.tmpdir)

    # Do some sanity checking for compatability between models
    if self.lda:
      assert not self.l1
      assert not self.l1
      assert not self.l2
      assert not self.loss
      assert not self.adaptive
      assert not self.oaa
      assert not self.csoaa
      assert not self.wap
      assert not self.cb
      assert not self.cb_type
      assert not self.ect
      assert not self.bfgs
    else:
      assert not self.lda_D
      assert not self.lda_rho
      assert not self.lda_alpha
      assert not self.minibatch

    if self.sgd:
      assert not self.adaptive
      assert not self.invariant
      assert not self.normalized

    self.working_directory = working_dir or os.getcwd()

  def vw_base_command(self, base, is_train):
    l = base
    if self.no_model is None: l.append('-f %s' % self.get_model_file())
    if self.bits is not None: l.append('-b %d' % self.bits)
    if self.learning_rate is not None: l.append('--learning_rate=%f' % self.learning_rate)
    if self.l1 is not None: l.append('--l1=%f' % self.l1)
    if self.l2 is not None: l.append('--l2=%f' % self.l2)
    if self.initial_t is not None: l.append('--initial_t=%f' % self.initial_t)
    if self.quadratic is not None: l.append('-q %s' % self.quadratic)
    if self.cubic is not None: l.append('--cubic %s' % self.cubic)
    if self.power_t is not None: l.append('--power_t=%f' % self.power_t)
    if self.loss is not None: l.append('--loss_function=%s' % self.loss)
    if self.decay_learning_rate is not None: l.append('--decay_learning_rate=%f' % self.decay_learning_rate)    
    if self.lda is not None: l.append('--lda=%d' % self.lda)
    if self.lda_D is not None: l.append('--lda_D=%d' % self.lda_D)
    if self.lda_rho is not None: l.append('--lda_rho=%f' % self.lda_rho)
    if self.lda_alpha is not None: l.append('--lda_alpha=%f' % self.lda_alpha)
    if self.minibatch is not None: l.append('--minibatch=%d' % self.minibatch)
    if is_train:
      if self.oaa is not None: l.append('--oaa=%d' % self.oaa)
      if self.ect is not None: l.append('--ect=%d' % self.ect)
      if self.csoaa is not None: l.append('--csoaa=%d' % self.csoaa)
      if self.wap is not None: l.append('--wap=%d' % self.wap)
      if self.cb is not None: l.append('--cb=%d' % self.cb)
      if self.cb_type is not None: l.append('--cb_type %s' % self.cb_type)
    if self.unique_id is not None: l.append('--unique_id=%d' % self.unique_id)
    if self.total is not None: l.append('--total=%d' % self.total)
    if self.node is not None: l.append('--node=%d' % self.node)
    if self.span_server is not None: l.append('--span_server=%s' % self.span_server)
    if self.mem is not None: l.append('--mem=%d' % self.mem)
    if self.audit: l.append('--audit')
    if self.bfgs: l.append('--bfgs')
    if self.adaptive: l.append('--adaptive')
    if self.invariant: l.append('--invariant')
    if self.normalized: l.append('--normalized')
    if self.sgd: l.append('--sgd')
    if self.ignore is not None: l.append('--ignore=%d' % self.ignore)
    if self.nn is not None: l.append('--nn=%d' % self.nn)
    if self.holdout_off is not None: l.append('--holdout_off')
    return ' '.join(l)

  def vw_train_command(self, cache_file):
    if os.path.exists(self.get_model_file()) and self.incremental:
      return self.vw_base_command([self.vw], True) + ' --passes %d -c -i %s' \
        % (self.passes, self.get_model_file())
    else:
      print 'No existing model file or not options.incremental'
      return self.vw_base_command([self.vw], True) + ' --passes %d -c' \
          % (self.passes)

  def vw_test_command(self, model_file, prediction_file):
    return self.vw_base_command([self.vw], False) + ' -t -i %s -r %s' % (model_file, prediction_file)

  def training(self, instances):    
    if type(instances) is str:
      self.start_training(instances)
      self.close_process()  
      return

    f = self.save_tmp_file(instances, True)
    self.start_training(f)
    self.close_process()
    self.del_file(f)

  def predicting(self, instances):
    if type(instances) is str:
      self.start_predicting(instances)
      self.close_process()  
      return
    f = self.save_tmp_file(instances, False)
    self.start_predicting(f)
    self.close_process()
    self.del_file(f)

  def save_tmp_file(self, instances, training=True):    
    f = self.tmpfile('_tmp_' + ('training' if training else 'testing') + '_file.vw.')
    with open(f, 'wb') as fs: fs.write('\n'.join(instances))    
    return f

  def tmpfile(self, suffix):
    _, f = tempfile.mkstemp(dir=self.tmpdir, suffix=suffix)
    os.close(_)
    return self.tmpdir + '/' + f.split('\\')[-1]

  def start_training(self, training_file):
    cache_file = self.tmpdir + '/' + self.handle + '.cache'
    model_file = self.get_model_file()

    # Remove the old cache and model files
    if not self.incremental:
      self.del_file(cache_file)      
      self.del_file(model_file)      

    # Run the actual training
    cmd = self.vw_train_command(cache_file)
    self.vw_process = self.make_subprocess(cmd, training_file)

  def close_process(self):
    # Close the process
    assert self.vw_process
    if self.vw_process.wait() != 0:
      raise Exception("vw_process %d (%s) exited abnormally with return code %d" % \
        (self.vw_process.pid, self.vw_process.command, self.vw_process.returncode))

  def start_predicting(self, testing_file):
    model_file = self.get_model_file()
    # Be sure that the prediction file has a unique filename, since many processes may try to
    # make predictions using the same model at the same time
    pred_file = self.handle + '.prediction'
    prediction_file = self.tmpfile(pred_file)    

    self.vw_process = self.make_subprocess(
      self.vw_test_command(model_file, prediction_file), testing_file)
    self.prediction_file = prediction_file    

  def parse_prediction(self, p):
    return map(float, p.split()) if self.lda else float(p.split()[0])

  def read_predictions_(self):
    for x in open(self.prediction_file):
      yield self.parse_prediction(x)
    self.del_file(self.prediction_file)

  def del_file(self, file):
    #try: os.remove(file)
    #except OSError: pass
    pass

  def make_subprocess(self, command, file):    
    stdout = open('nul', 'w')
    stderr = open('nul', 'w') if self.silent else sys.stderr
    
    commands = shlex.split(str(command))
    commands += ['-d', file]
    print 'Running command: "%s"' % str(commands)
    result = subprocess.Popen(commands, 
        stdout=stdout, stderr=stderr, 
        close_fds=sys.platform != "win32", 
        universal_newlines=True, cwd='.')
    result.command = command
    return result

  def get_model_file(self):
    if self.incremental:
      return self.filename
    else:
      return self.tmpdir + '/' + self.filename

  