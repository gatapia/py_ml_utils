import inspect
from keras.models import *
from keras.layers import *
from theano import function

class NN(self):
  def __init__(self, seed=0): 
    np.random.seed(0)
    
    self.branch = None
    self.last_name = None

  def add_layer(layer): 
    if type(layer) is not Layer: raise Exception('add_layer expects a keras.layers.Layer as input')    
    if self.branch is None: create_branch(layer.input_shape)
    return self._add_layer_impl(layer)

  def _add_layer_impl(layer, **kw):
    name = self.branch + ':' + type(layer).__name__ + ':' + str(len(self.nodes))
    self.add_node(layer, name, last_name, **kw)   
    last_name = name
    return self  

  def add_layers(layers): 
    for layer in layers: add_layer(layer)
    return self

  def add_output_layers(layers, merge_mode='concat'):
    if len(self.get_output()) > 0: raise Exception('Only single output branch supported')

    all_inputs = [n['input'] for n in self.node_config if n['input'] != '']
    all_finals = [n['name'] for n in self.node_config if n['name'] not in all_inputs]

    self.branch = 'output:0'    
    self._add_layer_impl(layers[0], inputs=all_finals, merge_mode=merge_mode)
    for layer in layers[1:]: self._add_layer_impl(layer)
    self.add_output(name=str(len(self.outputs)) + '_output', input=self.last_name)
    return self


  def create_branch(input_shape, dtype=int): 
    if self.branch is None: self.branch = 'input:0'
    else: self.branch = 'input:' + str(int(self.branch.split(':')[1]) + 1)
    self.add_input(self.branch, input_shape, dtype=dtype)
    last_name = self.branch
    return self
  
  def __str__(self): return print_self(self)
