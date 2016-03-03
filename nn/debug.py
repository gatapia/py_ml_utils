from keras import backend as K
from keras.layers import *
from theano import function
import inspect

def _call_f(inp, method, input_data):
  f = K.function([inp], [method]) if inp is not None \
      else K.function([], [method])
  return f([input_data])[0] if input_data is not None \
      else f([])[0]

def _simulate_layer_call(layer, input_data, train=True):
  np.random.seed(0)
  if not hasattr(layer, 'input'):
    if not hasattr(input_data, 'ndim'): input_data = np.array(input_data)
    layer.input = K.placeholder(ndim=input_data.ndim)
    layer.set_input_shape(input_data.shape)
  return _call_f(layer.input, layer.get_output(train=train), input_data)

def print_out(layer, input_data, train=True):
  output = _simulate_layer_call(layer, input_data, train)
  print output
  return output

def print_shape(layer, input_data, train=True):
  output = _simulate_layer_call(layer, input_data, train)
  print output.shape
  return output

def to_str(obj, print_shapes=True, indent=0, buffer=None):
  desc = buffer or ''

  if isinstance(obj, Layer):
    desc += type(obj).__name__ + '('
    args, _, _, defaults = inspect.getargspec(obj.__init__)  
    mandatories = len(args) if defaults is None else len(args) - len(defaults)
    arg_descs = []
    for i, arg in enumerate(args):
      if arg == 'self' or arg == 'layers' or not hasattr(obj, arg): continue        
      val = getattr(obj, arg)
      if inspect.isfunction(val): val = val.__name__
      default_idx = i - mandatories
      if default_idx < 0 or defaults[default_idx] != val:
        arg_descs.append(arg + '=' + (("'" + val + "'") if type(val) is str else str(val)))
    desc += ', '.join(arg_descs) + ')'
    if print_shapes: 
      try: desc += ' => ' + str(obj.output_shape)
      except: desc += ' => unknown'
    desc = ('    ' * indent) + desc
    if hasattr(obj, 'layers') and len(obj.layers) > 0:
      desc = to_str(obj.layers, print_shapes, indent+1, desc)
    return desc

  elif type(obj) is list and isinstance(obj[0], Layer):    
    for l in obj: 
      desc = to_str(l, print_shapes, indent, desc)
      desc += '\n'
    return desc

  elif type(obj) is Sequential: return to_str(obj.layers)
  else: raise Exception(type(obj).__name__ + ' is not supported')

def print_layer(layer, print_shapes=True, indent=0):  
  print to_str(layer, print_shapes, indent)  

def print_layers(layers, print_shapes=True, indent=0):
  print to_str(layers, print_shapes, indent)

def print_sequential(seq, print_shapes=True):
  print to_str(seq, print_shapes)
