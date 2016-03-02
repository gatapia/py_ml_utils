from keras.models import *

def build_sequential(layers, input_shape):
  if layers is None or len(layers) == 0: raise Exception('layers is empty')
  np.random.seed(0)
  seq = Sequential()
  first_layer = layers[0]
  first_layer.set_input_shape([None] + list(input_shape[1:]))
  seq.add(first_layer)
  for layer in layers[1:]: seq.add(layer)
  return seq

def build_graph(branches, trunk): 
  np.random.seed(0)
  graph = Graph()
  [_add_graph_branch(graph, layers) for layers in branches]
  _merge_branches_and_create_trunk(graph, trunk)
  return graph

def _add_graph_branch(graph, layers, dtype='int'):  
  if layers is None or len(layers) == 0: raise Exception('branch is empty')
  branch_idx = str(len(graph.inputs));
  name = branch_idx + 'input'
  if hasattr(layers[0], 'input_length') and layers[0].input_length > 0:
    graph.add_input(name, (layers[0].input_length,), dtype=dtype)
  elif hasattr(layers[0], 'input_dim') and layers[0].input_dim > 0:
    graph.add_input(name, (layers[0].input_dim,), dtype=dtype)
  else:
    graph.add_input(name, layers[0].input_shape, dtype=dtype)

  for layer_idx, layer in enumerate(layers):
    if layer is None: continue
    last_name = name
    name = branch_idx + '_' + str(layer_idx) + '_' + (layer.name or str(layer_idx))
    # print 'adding:', name, 'layer:', print_layer(layer)
    graph.add_node(layer, name, last_name)   
  return graph

def _merge_branches_and_create_trunk(graph, trunk_layers, merge_mode='concat'):
  if len(trunk_layers) == 0: raise Exception('no trunk layers provided')
  all_inputs = [n['input'] for n in graph.node_config if n['input'] != '']
  all_finals = [n['name'] for n in graph.node_config if n['name'] not in all_inputs]
  last_name = name = 'trunk_merger'
  graph.add_node(trunk_layers[0], name, inputs=all_finals, merge_mode=merge_mode)  
  for layer in trunk_layers[1:]:
    last_name = name
    name = branch_idx + (l.name or str(layer_idx))
    graph.add_node(layer, name, last_name)   
  graph.add_output(name=str(len(graph.outputs)) + '_output', input=last_name)
  return graph
