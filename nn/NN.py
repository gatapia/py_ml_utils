import inspect
from keras.models import Graph
from keras.layers import *

from theano import function
from debug import to_str
from model_eval import fit_evaluate

class NN(Graph):
  def __init__(self, seed=0): 
    super(NN, self).__init__()
    np.random.seed(0)    
    self.branch = None
    self.last_name = None

  def add_branch(self, layers): 
    self.branch = None
    self.last_name = None

    print 'add_branch w/ %d layers' % len(layers)
    self._create_branch(layers[0].input_shape)
    for layer in layers: self._add_layer(layer)
    return self

  def _add_layer(self, layer, **kw):
    if len(self.outputs) > 0: raise Exception('NN already has an output layer')
    if not isinstance(layer, Layer): raise Exception('_add_layer expects a keras.layers.Layer as input')
    name = self.branch + ':' + type(layer).__name__ + ':' + str(len(self.nodes))
    print 'add_layer - layer: %s' % to_str(layer)
    self.add_node(layer, name, self.last_name, **kw)   
    self.last_name = name
    return self    

  def add_output_layers(self, layers, merge_mode='concat'):
    if len(self.get_output()) > 0: raise Exception('Only single output branch supported')
    print 'add_output_layers w/ %d layers' % len(layers)

    all_inputs = [n['input'] for n in self.node_config if n['input'] != '']
    all_finals = [n['name'] for n in self.node_config if n['name'] not in all_inputs]

    self.branch = 'output:0'    
    self._add_layer(layers[0], inputs=all_finals, merge_mode=merge_mode)
    for layer in layers[1:]: self._add_layer(layer)
    self._add_output_impl()    
    return self

  def _add_output_impl(self):
    self.add_output(name=str(len(self.outputs)) + '_output', input=self.last_name)

  def _create_branch(self, input_shape, dtype='float'):     
    if self.branch is None: self.branch = 'input:0'
    else: self.branch = 'input:' + str(int(self.branch.split(':')[1]) + 1)
    self.add_input(self.branch, input_shape=input_shape[1:], dtype=dtype)
    self.last_name = self.branch
    return self

  def eval(self, X, y, 
      final_hodlout_split=.05, validation_split=.075, 
      epochs=3, batch_size=128, 
      loss='binary_crossentropy', optimizer='rmsprop', 
      compile_args={}, fit_args={}):
    print 'evaluating...'
    if len(self.inputs) == 0: raise Exception('NN has no branches')
    if len(self.outputs) == 0: self._add_output_impl()

    if not type(X) is dict:
      inputs = self.inputs.keys()
      newx = {}
      if len(inputs) == 1: newx[inputs[0]] = X
      else:
        if len(inputs) != len(X): raise Exception('When NN has multiple inputs, the len(X) should be the same as the number of inputs.')
        newx = {}
        for i in range(len(X)): newx[inputs[i]] = X[i]
      X = newx

    return fit_evaluate(self, X, y, final_hodlout_split, validation_split, 
      epochs, batch_size, 
      loss, optimizer, compile_args, fit_args)
  
  def __str__(self): return print_self(self)
