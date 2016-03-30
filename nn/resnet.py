from keras.models import *
from keras.layers import *
import sys

channels = 0

# imagenet - not working @ Flatten layer
def build_resnet(input_shape, depth=34, 
      shortcut_type=None, datatype='imagenet'):      
  if depth > 34 and sys.getrecursionlimit() <= 1000: sys.setrecursionlimit(5000)

  global channels
  if shortcut_type is None: shortcut_type = 'B' if datatype == 'imagenet' else 'A'
  
  model = Graph()  
  model.add_input('0', input_shape)
  def add(l, last_str=None):    
    last_l = 0 if len(model.nodes) == 0 else int(model.nodes.keys()[-1])
    last_str = last_str or str(last_l)
    last_l = last_l + 1
    model.add_node(l, str(last_l), last_str),
    return last_l

  def add_two_branches(b1, b2, merger, merge_mode):    
    start_l = model.nodes.keys()[-1]
    last_1 = add(b1[0], start_l)    
    for l in b1[1:]: last_1 = add(l)
    last_2 = add(b2[0], start_l)
    for l in b2[1:]: last_2 = add(l)

    last_l = max(last_1, last_2) + 1
    model.add_node(merger, str(last_l), inputs=[str(last_1), str(last_2)], merge_mode=merge_mode)

  def shortcut(input_plane, output_plane, stride):
    use_conv = shortcut_type == 'C' or (
        shortcut_type == 'B' and input_plane != output_plane)    
    if use_conv:     
      return [
        # hack to get stride 2 to work, why do we need this? Image size?
        ZeroPadding2D((1 if stride > 1 else 0, 1 if stride > 1 else 0)), 
        Convolution2D(output_plane, 1, 1, subsample=(stride, stride), init='normal'),
        BatchNormalization()
      ]
    elif input_plane != output_plane:
      return [
        # hack to get stride 2 to work, why do we need this? Image size?
        ZeroPadding2D((1 if stride > 1 else 0, 1 if stride > 1 else 0)),        
        AveragePooling2D((1, 1), (stride, stride)),
        IdentityAndMultZero()        
      ]
    else: 
      return [Identity()]

  def basic_block(n, stride):
    global channels
    '''
    The basic residual layer block for 18 and 34 layer network, and the
    CIFAR networks
    '''
    input_plane = channels
    channels = n
    b1 = [
      Convolution2D(n, 3, 3, subsample=(stride, stride), init='normal'),
      ZeroPadding2D((1, 1)),
      BatchNormalization(),
      Activation('relu'),
      Convolution2D(n, 3, 3, subsample=(1, 1), init='normal'),
      ZeroPadding2D((1, 1)),
      BatchNormalization()
    ]  
    b2 = shortcut(input_plane, n, stride)    
    add_two_branches(b1, b2, Activation('relu'), 'sum')    

  def bottleneck(n, stride):
    global channels
    '''
    The bottleneck residual layer for 50, 101, and 152 layer networks
    '''
    input_plane = channels
    channels = n * 4
    b1 = [
      Convolution2D(n, 1, 1, subsample=(1, 1), init='normal'),      
      BatchNormalization(),
      Activation('relu'),
      Convolution2D(n, 3, 3, subsample=(stride, stride), init='normal'),
      ZeroPadding2D((1, 1)),
      BatchNormalization(),      
      Activation('relu'),
      Convolution2D(n*4, 1, 1, subsample=(1, 1), init='normal'),
      BatchNormalization()
    ]
    b2 = shortcut(input_plane, n * 4, stride)
    add_two_branches(b1, b2, Activation('relu'), 'sum')
  
  def layer(block, features, count, stride):
    for i in range(count): 
      block(features, stride if i == 0 else 1)

  if datatype == 'imagenet':
    '''
    Configurations for ResNet:
    num. residual blocks, num features, residual block function
    '''
    cfg = {
      18: ((2, 2, 2, 2), 512, basic_block),
      34: ((3, 4, 6, 3), 512, basic_block),
      50: ((3, 4, 6, 3), 2048, bottleneck),
      101: ((3, 4, 23, 3), 2048, bottleneck),
      152: ((3, 8, 36, 3), 2048, bottleneck)
    }
    defn, n_features, block = cfg[depth]
    channels = 64
    add(Convolution2D(channels, 7, 7, subsample=(2,2), init='normal'))
    add(ZeroPadding2D((3, 3))),
    add(BatchNormalization())
    add(Activation('relu')),
    add(MaxPooling2D((3,3),(2,2)))
    layer(block, 64, defn[0], 1)
    layer(block, 128, defn[1], 2)
    layer(block, 256, defn[2], 2)
    layer(block, 512, defn[3], 2)
    add(AveragePooling2D((7, 7), (1, 1)))
    add(Flatten())
    # add(Dense(1000, activation='linear'))
  elif datatype == 'cifar10':
    assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 101, 1202')
    n = (depth - 2) / 6
    channels = 16
    add(Convolution2D(channels, 3, 3, subsample=(1, 1), init='normal'))    
    add(ZeroPadding2D((1, 1)))
    add(BatchNormalization())
    add(Activation('relu'))
    layer(basic_block, 16, n, 1)
    layer(basic_block, 32, n, 2)
    layer(basic_block, 64, n, 2)
    add(AveragePooling2D((8, 8), (1, 1)))
    add(Flatten())
    # add(Dense(10, activation='linear'))
  elif datatype == 'chalearn': 
    assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 101, 1202')
    n = (depth - 2) / 6
    channels = 32
    add(Convolution2D(channels, 3, 3, subsample=(1, 1), init='normal'))    
    add(ZeroPadding2D((1, 1)))
    add(BatchNormalization())
    add(Activation('relu'))
    add(MaxPooling2D((2, 2), (1, 1)))
    layer(basic_block, 32, n, 1)
    layer(basic_block, 64, n, 2)
    layer(basic_block, 128, n, 2)
    add(AveragePooling2D((8, 8), (1, 1)))
    add(Flatten())
    # add(Dense(10, activation='linear'))
  else: raise Exception('invalid dataset: ' + datatype)
  return model

class IdentityAndMultZero(Layer):  
  @property
  def output_shape(self):
    output_shape = list(self.input_shape)
    output_shape[1] += output_shape[1]    
    return tuple(output_shape)

  def get_output(self, train):
    X = self.get_input(train)
    return K.concatenate([X, X*0], 1)

class Identity(Layer):
  def get_output(self, train):
    return self.get_input(train)

'''
class MultZero(Layer):
  def get_output(self, train):
    return self.get_input(train) * 0
'''

if __name__ == '__main__':
  # shortcut_type=B for imagenet and A for cifar
  # nn = build_resnet(input_shape=(3, 224, 224), depth=152, shortcut_type='B', datatype='imagenet')
  for depth in [18, 34, 50, 101, 152]:
    print '\n\ntesting depth:', depth
    nn = build_resnet(input_shape=(3, 224, 224), depth=depth, shortcut_type='B', datatype='imagenet')
    nn = build_resnet(input_shape=(3, 32, 32), depth=depth, shortcut_type='A', datatype='cifar10')
