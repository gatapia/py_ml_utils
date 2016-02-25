from keras import backend as K
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

def print_layer(layer, print_shapes=True, indent=0):  
  desc = type(layer).__name__ + '('
  args, _, _, defaults = inspect.getargspec(layer.__init__)  
  mandatories = len(args) if defaults is None else len(args) - len(defaults)
  arg_descs = []
  for i, arg in enumerate(args):
    if arg == 'self' or arg == 'layers' or not hasattr(layer, arg): continue        
    val = getattr(layer, arg)
    if inspect.isfunction(val): val = val.__name__
    default_idx = i - mandatories
    if default_idx < 0 or defaults[default_idx] != val:
      arg_descs.append(arg + '=' + (("'" + val + "'") if type(val) is str else str(val)))
  desc += ', '.join(arg_descs) + ')'
  if print_shapes: 
    try: desc += ' => ' + str(layer.output_shape)
    except: desc += ' => unknown'
  print ('    ' * indent) + desc
  if hasattr(layer, 'layers') and len(layer.layers) > 0:
    print_layers(layer.layers, print_shapes, indent+1)

def print_layers(layers, print_shapes=True, indent=0):
  for l in layers: print_layer(l, print_shapes, indent)

def print_sequential(seq, print_shapes=True):
  print_layers(seq.layers, print_shapes)
