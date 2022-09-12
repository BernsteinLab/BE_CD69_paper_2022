import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import attention_tf as attention_module

SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896


class Enformer(tf.keras.Model):
  """Main model."""

  def __init__(self,
               channels: int = 1536,
               num_transformer_layers: int = 11,
               num_heads: int = 8,
               pooling_type: str = 'attention',
               name: str = 'enformer', num_list: int = 5): #int=6 is what deepmind used
    """Enformer model.
    Args:
      channels: Number of convolutional filters and the overall 'width' of the
        model.
      num_transformer_layers: Number of transformer layers.
      num_heads: Number of attention heads.
      pooling_type: Which pooling function to use. Options: 'attention' or max'.
      name: Name of sonnet module.
    """
    super().__init__(name=name)
    # pylint: disable=g-complex-comprehension,g-long-lambda,cell-var-from-loop
    heads_channels = {'human': 2}
    dropout_rate = 0.4
    assert channels % num_heads == 0, ('channels needs to be divisible '
                                       f'by {num_heads}')
    whole_attention_kwargs = {
        'attention_dropout_rate': 0.05,
        'initializer': None,
        'key_size': 64,
        'num_heads': num_heads,
        'num_relative_position_features': channels // num_heads,
        'positional_dropout_rate': 0.01,
        'relative_position_functions': [
            'positional_features_exponential',
            'positional_features_central_mask',
            'positional_features_gamma'
        ],
        'relative_positions': True,
        'scaling': True,
        'value_size': channels // num_heads,
        'zero_initialize': True
    }

    trunk_name_scope = tf.name_scope('trunk')
    trunk_name_scope.__enter__()
    
    def conv_block(filters, width=1, w_init='glorot_uniform', padding='same', name='conv_block', **kwargs):
      return tf.keras.Sequential([
          tf.keras.layers.BatchNormalization(),
          tfa.layers.GELU(),
          tf.keras.layers.Conv1D(filters, width, kernel_initializer=w_init, padding=padding, **kwargs)
      ], name=name)

    stem = tf.keras.Sequential([
        tf.keras.layers.Conv1D(channels // 2, 15, padding='same'),
        Residual(conv_block(channels // 2, 1, name='pointwise_conv_block')),
        pooling_module(pooling_type, pool_size=2),
    ], name='stem')

    #passing in hyperparam num_list
    filter_list = exponential_linspace_int(start=channels // 2, end=channels,
                                           num = num_list, divisible_by=128)
    
     
    
    conv_tower = tf.keras.Sequential([
        tf.keras.Sequential([
            conv_block(num_filters, 5, padding='same'),
            Residual(conv_block(num_filters, 1, name='pointwise_conv_block')),
            pooling_module(pooling_type, pool_size=2),
            ],
                   name=f'conv_tower_block_{i}')
        for i, num_filters in enumerate(filter_list)], name='conv_tower')

    # Transformer.
    def transformer_mlp():
      return tf.keras.Sequential([
           tf.keras.layers.LayerNormalization(
    axis=-1, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones'),
          tf.keras.layers.Dense(channels * 2, activation='relu'),
          tf.keras.layers.Dropout(rate= dropout_rate),
          tf.keras.layers.Dense(channels),
         tf.keras.layers.Dropout(rate= dropout_rate)], name='mlp')
      
    

    transformer = tf.keras.Sequential([
        tf.keras.Sequential([
            Residual(tf.keras.Sequential([
                tf.keras.layers.LayerNormalization(
    axis=-1, epsilon=0.001, center=True, scale=True,
    beta_initializer='zeros', gamma_initializer='ones'),
                attention_module.MultiheadAttention(**whole_attention_kwargs,
                                                    name=f'attention_{i}'),
                tf.keras.layers.Dropout(rate= dropout_rate)], name='mha')),
                 #Added the above extra projection
            Residual(transformer_mlp(), name=f'transformer_block_{i}')])
        for i in range(num_transformer_layers)])

        #TODO: Add in the names of the layers. 
    
    crop_final = TargetLengthCrop1D(TARGET_LENGTH, name='target_input')

    final_pointwise = tf.keras.Sequential([
        conv_block(channels * 2, 1, padding='same'),
        tf.keras.layers.Dropout(rate=dropout_rate / 8),
        tfa.layers.GELU()], name='final_pointwise')

    self._trunk = tf.keras.Sequential([stem,
                              conv_tower,
                              transformer,
                              crop_final,
                              final_pointwise],
                             name='trunk')
    trunk_name_scope.__exit__(None, None, None)
    
    
    #with tf.name_scope('heads'):
    self._new_heads = {
        head: tf.keras.Sequential(
          [tf.keras.layers.Dense(num_channels, activation='softplus')],
          name=f'new_head_{head}')
        for head, num_channels in heads_channels.items()
    }

    # pylint: enable=g-complex-comprehension,g-long-lambda,cell-var-from-loop

  @property
  def trunk(self):
    return self._trunk
    
  @property
  def heads(self):
    return self._new_heads

  def call(self, inputs: tf.Tensor,
               training: bool) -> Dict[str, tf.Tensor]:
    trunk_embedding = self.trunk(inputs, training=training)
    return {
        head: head_module(trunk_embedding, training=training)
        for head, head_module in self._new_heads.items()
    }

  @tf.function(input_signature=[
      tf.TensorSpec([None, SEQUENCE_LENGTH, 4], tf.float32)])
  def predict_on_batch(self, x):
    """Method for SavedModel."""
    return self(x, training=False)


class TargetLengthCrop1D(tf.keras.layers.Layer):
  """Crop sequence to match the desired target length."""

  def __init__(self, target_length: int, name='target_length_crop'):
    super().__init__(name=name)
    self._target_length = target_length

  def call(self, inputs):
    trim = (inputs.shape[-2] - self._target_length) // 2
    if trim < 0:
      raise ValueError('inputs longer than target length')

    return inputs[..., trim:-trim, :]


def pooling_module(kind, pool_size):
  """Pooling module wrapper."""
  if kind == 'attention':
    return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
  elif kind == 'max':
    return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')
  else:
    raise ValueError(f'Invalid pooling kind: {kind}.')


class SoftmaxPooling1D(tf.keras.layers.Layer):
  """Pooling operation with optional weights."""

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """Softmax pooling.
    Args:
      pool_size: Pooling size, same as in Max/AvgPooling.
      per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
      w_init_scale: When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
      name: Module name.
    """
    super().__init__(name=name)
    self._pool_size = pool_size
    self._per_channel = per_channel
    self._w_init_scale = w_init_scale
    self._logit_linear = None


  def _initialize(self, num_features):
    self._logit_linear = tf.keras.layers.Dense(
        num_features if self._per_channel else 1,
        use_bias=False,  # Softmax is agnostic to shifts.
       kernel_initializer = tf.keras.initializers.Identity(
    gain=1.0))

  def call(self, inputs):
    _, length, num_features = inputs.shape
    self._initialize(num_features)
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)


class Residual(tf.keras.layers.Layer):
  """Residual block."""

  def __init__(self, layer,  name='residual'):
    super().__init__(name=name)
    '''
    layer is some tf.keras layer
    '''
    self._layer = layer

  def call(self, inputs: tf.Tensor, training: bool, *args,
               **kwargs) -> tf.Tensor:
    print(inputs.shape, self._layer(inputs, training, *args, **kwargs).shape)
    return inputs + self._layer(inputs, training, *args, **kwargs)



def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]


def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]


if __name__ == '__main__':
  x = tf.random.uniform(shape=(1,18000,4))
  model = Enformer(channels=1536, num_heads=2, num_transformer_layers=2, pooling_type='max')
  print(model(x, training=True))


  y = tf.random.uniform(shape=(1,131072,4))
  model = Enformer(channels=1536, num_heads=2, num_transformer_layers=2, pooling_type='attention') 
  print(model(y, training=True))

  #TODO: CONV Tower crashes with pooling type 'attention