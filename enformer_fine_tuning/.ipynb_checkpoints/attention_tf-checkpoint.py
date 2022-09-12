from einops import rearrange, reduce
from typing import Any, Dict, List, Optional
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers 

class TransformerBlock(tf.keras.Model):

    def __init__(
        self,
        channels: int,
        dropout_rate: float,
        attention_kwargs: Dict[str, Any],
        name: str = 'transformer_block',
    ):
        super().__init__(name=name)
        self.mha_ln = layers.LayerNormalization(axis=-1,
                                            scale=True,
                                            center=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones")
        self.mha = MultiheadSelfAttention(**attention_kwargs)
        self.mha_dropout = layers.Dropout(dropout_rate)

        self.mlp_ln = layers.LayerNormalization(axis=-1,
                                            scale=True,
                                            center=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones")
        self.mlp_linear1 = layers.Dense(channels * 2)
        self.mlp_dropout1 = layers.Dropout(dropout_rate)
        self.mlp_linear2 = layers.Dense(channels)
        self.mlp_dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        x = self.mha_ln(inputs)
        x = self.mha(x, training=training)
        x = self.mha_dropout(x, training=training)
        x += inputs  # Residual
        mha_output = x

        # MLP.
        x = self.mlp_ln(mha_output)
        x = self.mlp_linear1(x)
        x = self.mlp_dropout1(x, training=training)
        x = tf.nn.relu(x)
        x = self.mlp_linear2(x)
        x = self.mlp_dropout2(x, training=training)
        return x + mha_output


class MultiheadAttention(tf.keras.layers.Layer):
    """Multi-head attention."""

    def __init__(self,
                 value_size: int,
                 key_size: int,
                 num_heads: int,
                 scaling: bool = True,
                 attention_dropout_rate: float = 0.1,
                 relative_positions: bool = False,
                 relative_position_symmetric: bool = False,
                 relative_position_functions: Optional[List[str]] = None,
                 num_relative_position_features: Optional[int] = None,
                 positional_dropout_rate: float = 0.1,
                 zero_initialize: bool = True,
                 use_projection_bias : bool = False,
                 initializer: Optional[tf.keras.initializers.Initializer] = None,
                 output_size = None,
                 name: str = None):
               
    
        super().__init__(name=name)
        self._value_size = value_size
        self._key_size = key_size
        self.num_heads = num_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._scaling = scaling
        self._relative_positions = relative_positions
        self._relative_position_symmetric = relative_position_symmetric
        self._relative_position_functions = relative_position_functions
        self.zero_initialize = zero_initialize 
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        if num_relative_position_features is None:
          # num_relative_position_features needs to be divisible by the number of
          # relative positional functions *2 (for symmetric & asymmetric version).
            divisible_by = 2 * len(self._relative_position_functions)
            self._num_relative_position_features = (
              (self._value_size // divisible_by) * divisible_by)
        else:
            self._num_relative_position_features = num_relative_position_features
        self._positional_dropout_rate = positional_dropout_rate

        self._initializer = initializer
    
        if self._initializer is None:
            self._initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

   
    def build(self, input_shape):

        num_query_features = input_shape[-1]
        num_key_features = input_shape[-1] #key dim = query dim = value dim
        num_value_features = (
            input_shape[-1] if len(input_shape) > 2 else num_key_features
        )
        
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )
        
        
        embedding_size = self._value_size * self.num_heads

        self._q_layer = layers.Dense(
            name='q_layer',
            units=self.num_heads*self._key_size,
            kernel_initializer=self._initializer, 
            use_bias = False)
        self._k_layer = layers.Dense(
            name='k_layer',
            units=self.num_heads*self._key_size,
            kernel_initializer=self._initializer, 
            use_bias = False)
        self._v_layer = layers.Dense(
            name='v_layer',
            units=self.num_heads*self._value_size,
            kernel_initializer=self._initializer, 
            use_bias = False)
        
        
        w_init = tf.keras.initializers.Zeros() if self.zero_initialize else self._initializer 
        
        self._embedding_layer = layers.Dense(
            name="embedding_layer",
            units = output_size,
            kernel_initializer=w_init, 
            use_bias = False)

        
        
        if self._relative_positions:
            self._r_k_layer = layers.Dense(
                self.num_heads*self._key_size,
                name='r_k_layer',
                use_bias=False,
                kernel_initializer=self._initializer)
            
            self._r_w_bias = tf.Variable(
                self._initializer([1, self.num_heads, 1, self._key_size],
                                dtype=tf.float32),
                name='r_w_bias')
            
            self._r_r_bias = tf.Variable(
                self._initializer([1, self.num_heads, 1, self._key_size],
                                dtype=tf.float32),
                name='r_r_bias')
        
    
        super().build(input_shape)
        
    def call(self, inputs,
               training=False):
        # Initialise the projection layers.
        embedding_size = self._value_size * self.num_heads
        seq_len = inputs.shape[1]

        # Compute q, k and v as multi-headed projections of the inputs.
        n, h = inputs.shape[-2], self.num_heads

        q = self._q_layer(inputs)
        k = self._k_layer(inputs)
        v = self._v_layer(inputs)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Scale the query by the square-root of key size.
        if self._scaling:
            q *= self._key_size**-0.5

        if self._relative_positions:
            # For relative positions, we project positions to form relative keys.
            distances = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
            positional_encodings = positional_features_all(
                positions=distances,
                feature_size=self._num_relative_position_features,
                seq_length=seq_len,
                feature_functions=self._relative_position_functions,
                symmetric=self._relative_position_symmetric)
              # [1, 2T-1, Cr]

            if training:
                positional_encodings = tf.nn.dropout(
                    positional_encodings, rate=self._positional_dropout_rate)

            # [1, H, 2T-1, K]
            r_k = self._r_k_layer(positional_encodings)
          #  (1,2*1024 -1, 512)
        

            # Add shifted relative logits to content logits.
            # [B, H, T', T]
            content_logits = tf.einsum('b h i d, b h j d -> b h i j', q + self._r_w_bias, k)
            # [B, H, T', 2T-1]
            r_k = rearrange(r_k, 'b n (h d) -> (b h) n d', h = h)

            relative_logits = tf.einsum('b h i d,  h j d -> b h i j', q + self._r_r_bias, r_k)
            #  [B, H, T', T]

            relative_logits = relative_shift(relative_logits)

            logits = content_logits + relative_logits
        else:
            # [B, H, T', T]
            logits = tf.matmul(q, k, transpose_b=True)

        weights = tf.nn.softmax(logits)

        # Dropout on the attention weights.
        if training:
            weights = tf.nn.dropout(weights, rate=self._attention_dropout_rate)

        # Transpose and reshape the output.
        out = tf.einsum('b h i j, b h j d -> b h i d', weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self._embedding_layer(out)

        return out

     


def relative_shift(x):
    """Shift the relative logits like in TransformerXL."""
    # We prepend zeros on the final timescale dimension.
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x


# Available feature functions:
def get_positional_feature_function(name):
    """Returns positional feature functions."""
    available = {
      'positional_features_exponential': positional_features_exponential,
      'positional_features_central_mask': positional_features_central_mask,
      'positional_features_gamma': positional_features_gamma,
      'positional_features_cosine': positional_features_cosine,
      'positional_features_linear_masks': positional_features_linear_masks,
      'positional_features_sin_cos': positional_features_sin_cos,
    }
    if name not in available:
        raise ValueError(f'Function {name} not available in {available.keys()}')
    return available[name]


def positional_features_all(positions: tf.Tensor,
                            feature_size: int,
                            seq_length: Optional[int] = None,
                            bin_size: Optional[int] = None,
                            feature_functions: Optional[List[str]] = None,
                            symmetric=False):

    if feature_functions is None:
        feature_functions = ['positional_features_exponential',
                             'positional_features_central_mask',
                             'positional_features_gamma']
    num_components = len(feature_functions)  # 1 per each basis function
    if not symmetric:
        num_components = 2 * num_components

  # For now, we do not allow odd sized embeddings.
    if feature_size % num_components != 0:
        raise ValueError(
            f'feature_size has to be divisible by {num_components}')

    feature_functions = [get_positional_feature_function(f)
                         for f in feature_functions]
    num_basis_per_class = feature_size // num_components
    embeddings = tf.concat([f(tf.abs(positions), num_basis_per_class,
                            seq_length, bin_size)
                          for f in feature_functions],
                         axis=-1)
    if not symmetric:
        embeddings = tf.concat([embeddings,
                            tf.sign(positions)[..., tf.newaxis] * embeddings],
                           axis=-1)
    tf.TensorShape(embeddings.shape).assert_is_compatible_with(
          positions.shape + [feature_size])
    return embeddings


def _prepend_dims(x, num_dims):
    return tf.reshape(x, shape=[1] * num_dims + x.shape)


def positional_features_exponential(positions: tf.Tensor,
                                    feature_size: int,
                                    seq_length: Optional[int] = None,
                                    bin_size: Optional[int] = None,
                                    min_half_life: Optional[float] = 3.0):
  



  
    del bin_size  # Unused.
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    # Grid of half lifes from [3, seq_length / 2] with feature_size
    # distributed on the log scale.
    seq_length = tf.cast(seq_length, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = _prepend_dims(half_life, positions.shape.rank)
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
    return outputs


def positional_features_central_mask(positions: tf.Tensor,
                                     feature_size: int,
                                     seq_length: Optional[int] = None,
                                     bin_size: Optional[int] = None):
    """Positional features using a central mask (allow only central features)."""
    del seq_length  # Unused.
    del bin_size  # Unused.
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32))
    center_widths = center_widths - 1
    center_widths = _prepend_dims(center_widths, positions.shape.rank)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis],
                    tf.float32)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
    return outputs


def gamma_pdf(x, concentration, rate):
    """Gamma probability distribution function: p(x|concentration, rate)."""
    log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
    log_normalization = (tf.math.lgamma(concentration) -
                       concentration * tf.math.log(rate))
    return tf.exp(log_unnormalized_prob - log_normalization)


def positional_features_gamma(positions: tf.Tensor,
                              feature_size: int,
                              seq_length: Optional[int] = None,
                              bin_size: Optional[int] = None,
                              stddev=None,
                              start_mean=None):
    """Positional features computed using the gamma distributions."""
    del bin_size  # Unused.
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    if stddev is None:
        stddev = seq_length / (2 * feature_size)
    if start_mean is None:
        start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = _prepend_dims(mean, positions.shape.rank)
    concentration = (mean / stddev)**2
    rate = mean / stddev**2
    probabilities = gamma_pdf(
      tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis],
      concentration, rate)
    probabilities += 1e-8  # To ensure numerical stability.
    outputs = probabilities / tf.reduce_max(probabilities)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
    return outputs


def positional_features_cosine(positions: tf.Tensor,
                               feature_size: int,
                               seq_length: Optional[int] = None,
                               bin_size: Optional[int] = None):
    """Cosine positional features."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    periodicity = 1.25 * tf.pow(2.0, tf.range(0, feature_size, dtype=tf.float32))
    periodicity = _prepend_dims(periodicity, positions.shape.rank)

    outputs = tf.math.cos(2 * tnp.pi * positions[..., tf.newaxis] / periodicity)
    tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
    return outputs


def positional_features_linear_masks(positions: tf.Tensor,
                                     feature_size: int,
                                     seq_length: Optional[int] = None,
                                     bin_size: Optional[int] = None):
    """Exponentially increasing point focuses."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    distances = tf.range(0, feature_size, dtype=tf.float32)
    distances = _prepend_dims(distances, positions.shape.rank)
    outputs = tf.cast(distances == tf.abs(positions[..., tf.newaxis]),
                    dtype=tf.float32)

    tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
    return outputs


def positional_features_sin_cos(positions: tf.Tensor,
                                feature_size: int,
                                seq_length: Optional[int] = None,
                                bin_size: Optional[int] = None,
                                max_time=10000.0):
    """Sine/cosine positional encodings."""
    del bin_size  # Unused.
    del seq_length  # Unused.
    if feature_size % 2 != 0:
        raise ValueError('feature_size needs to be divisible by 2.')
    i = tf.range(0, feature_size, 2, dtype=tf.float32)
    i = _prepend_dims(i, positions.shape.rank)

    # Concat sines and cosines and return.
    outputs = tf.concat([
      tf.sin(positions[..., tf.newaxis] / max_time**(i / feature_size)),
      tf.cos(positions[..., tf.newaxis] / max_time**(i / feature_size))], -1)

    tf.TensorShape(outputs.shape).assert_is_compatible_with(
      positions.shape + [feature_size])
    return outputs

